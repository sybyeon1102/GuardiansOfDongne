import os
import json
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from project_core.env import (
    load_env,
    env_str,
    env_int,
    env_float,
    env_bool,
    env_path,
)
from modeling.schemas import PoseWindowRequest
from modeling.pipeline.pose_features import features_from_pose_seq


# =========================================================
# .env 로딩 (agent 디렉터리 기준)
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_env(BASE_DIR)


# =========================================================
# 환경 변수 / 기본 설정
# =========================================================


INFERENCE_SERVER_URL_DEFAULT = "http://localhost:8000"
INFERENCE_SERVER_URL = env_str(
    "INFERENCE_SERVER_URL",
    INFERENCE_SERVER_URL_DEFAULT
).rstrip("/")

AGENT_CODE = env_str("AGENT_CODE", "agent-unknown")

CAMERA_CONFIG_PATH = env_path("CAMERA_CONFIG_PATH", BASE_DIR)

STREAM_HTTP_HOST = env_str("STREAM_HTTP_HOST", "0.0.0.0")
STREAM_HTTP_PORT = env_int("STREAM_HTTP_PORT", "8001")

MJPEG_ENABLED = env_bool("MJPEG_ENABLED", "true")
HLS_ENABLED = env_bool("HLS_ENABLED", "false")

HLS_ROOT = env_path("HLS_OUTPUT_DIR", BASE_DIR)
HLS_BASE_PATH = env_str("HLS_BASE_PATH", "/streams")

BEHAVIOR_TRAIN_FPS = env_float("BEHAVIOR_TRAIN_FPS", 10.0)
POSE_WINDOW_SIZE = env_int("BEHAVIOR_WINDOW_SIZE", 16)
POSE_WINDOW_STRIDE = env_int("BEHAVIOR_WINDOW_STRIDE", 4)

POSE_MAX_FPS = env_float("POSE_MAX_FPS", "30")

TRACKING_SEND_FPS_DEFAULT = env_float("TRACKING_SEND_FPS_DEFAULT", "10.0")
TRACKING_DOWNSCALE_MODE = env_str("TRACKING_DOWNSCALE_MODE", "half")  # none | half

LOG_LEVEL = env_str("LOG_LEVEL", "INFO")
REQUEST_TIMEOUT_SEC = env_float("REQUEST_TIMEOUT_SEC", "5")


# =========================================================
# 서버와 맞춘 요청/응답 스키마 (FastAPI 서버 behavior_inference_server.py 참고)
# =========================================================


class InferenceResult(BaseModel):
    # BehaviorResultResponse 와 최대한 맞춘다.
    agent_code: str
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

    is_anomaly: bool
    det_prob: float            # 현재 EMA 기준 이상 확률
    top_label: str | None = None
    top_prob: float
    prob: dict[str, float]


class CameraStatus(str):
    ONLINE = "online"
    OFFLINE = "offline"
    STARTING = "starting"
    ERROR = "error"


class CameraSourceType(str):
    RTSP = "rtsp"
    FILE = "file"
    WEBCAM = "webcam"


class CameraSummary(BaseModel):
    id: str
    display_name: str
    type: str
    status: str
    reason: str | None = None
    mjpeg_url: str | None = None
    hls_url: str | None = None
    width: int | None = None
    height: int | None = None
    expected_fps: float | None = None
    tracking_send_fps: float | None = None
    downscale: str | None = None


# =========================================================
# 설정/유틸
# =========================================================


def ensure_ffmpeg_available() -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        msg = (
            "[Agent] ffmpeg 실행 파일을 찾을 수 없습니다.\n"
            "에이전트 환경에 ffmpeg를 설치한 뒤 다시 실행해 주세요.\n"
            "예시)\n"
            "  - Windows (s.gcdkoop): scoop install ffmpeg\n"
            "  - macOS (brew):   brew install ffmpeg\n"
            "  - Ubuntu:         sudo apt install ffmpeg\n"
        )
        raise RuntimeError(msg)


def _expand_source_path(raw_src: str) -> str:
    """
    파일 경로일 때 ~, 환경변수, BASE_DIR 기준 상대경로 등을 모두 처리한다.
    스트림(rtsp/http/rtmp…)이면 그대로 반환.
    """
    if looks_live_src(raw_src):
        return raw_src

    expanded = os.path.expandvars(os.path.expanduser(raw_src))
    src_path = Path(expanded)
    if not src_path.is_absolute():
        src_path = (BASE_DIR / src_path).resolve()
    return str(src_path)


def looks_live_src(src: str) -> bool:
    """
    rtsp/http/rtmp 등 "라이브 스트림" 주소처럼 보이는지 간단히 판별.
    """
    import re as _re

    return bool(_re.match(r"^(rtsp|rtsps|rtmp|http|https|rtp)://", src, _re.I))


@dataclass
class CameraConfig:
    camera_id: str
    display_name: str
    type: str  # "rtsp" | "file" | "webcam"
    source: str
    expected_fps: float | None = None
    tracking_send_fps: float | None = None
    downscale: str = TRACKING_DOWNSCALE_MODE  # "none" | "half"
    enabled: bool = True
    file_mode: str | None = None  # "loop" | "once" (file 일 때)

    # Pose 윈도우 설정 (전역값을 기본으로 사용)
    window_size: int = POSE_WINDOW_SIZE
    window_stride: int = POSE_WINDOW_STRIDE
    resize_width: int = 640  # Pose 입력 해상도 가로 기준
    frame_skip: int = 1      # 1이면 매 프레임 처리

    @property
    def is_live(self) -> bool:
        # rtsp / webcam 은 live, file 은 offline 재생
        return self.type in {CameraSourceType.RTSP, CameraSourceType.WEBCAM}

    @property
    def source_id(self) -> str:
        # 지금은 source 문자열 전체를 source_id 로 사용
        return self.source

    @property
    def hls_dir(self) -> Path:
        return HLS_ROOT / self.camera_id

    @property
    def hls_playlist(self) -> Path:
        return self.hls_dir / "index.m3u8"

    @property
    def tracking_stride(self) -> int | None:
        """
        expected_fps / tracking_send_fps 를 정수 stride 로 변환.
        둘 중 하나라도 없으면 None (트래킹 비활성),
        정수로 딱 떨어지지 않으면 load 시점에서 에러 처리.
        """
        if self.expected_fps and self.tracking_send_fps and self.tracking_send_fps > 0:
            stride = self.expected_fps / self.tracking_send_fps
            if abs(round(stride) - stride) > 1e-6:
                return None
            return int(round(stride))
        return None

    def finalize_pose_timing(self, train_fps: float) -> None:
        """
        expected_fps / train_fps 를 기준으로 pose frame_skip 을 자동 설정한다.

        - expected_fps 가 지정되어 있고 train_fps > 0 이면:
          frame_skip = max(1, round(expected_fps / train_fps))
        - expected_fps 가 없으면: frame_skip 은 기존 값(기본 1)을 유지.
        """
        if self.expected_fps and self.expected_fps > 0 and train_fps > 0:
            ratio = self.expected_fps / train_fps
            self.frame_skip = max(1, int(round(ratio)))


@dataclass
class CameraRuntimeState:
    status: str = CameraStatus.STARTING
    reason: str | None = None
    width: int | None = None
    height: int | None = None


CAMERA_CONFIGS: dict[str, CameraConfig] = {}
CAMERA_STATES: dict[str, CameraRuntimeState] = {}

LATEST_FRAMES: dict[str, np.ndarray] = {}
LATEST_FRAMES_LOCK = threading.Lock()


def load_camera_configs() -> list[CameraConfig]:
    """
    CAMERA_CONFIG_PATH 에서 cameras.json 을 읽어서 CameraConfig 리스트로 변환.
    """
    cfg_path = Path(CAMERA_CONFIG_PATH)
    if not cfg_path.exists():
        raise RuntimeError(f"cameras.json not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "cameras" not in raw:
        raise RuntimeError("cameras.json format error: root must be an object with 'cameras'.")

    cams: list[CameraConfig] = []
    for item in raw.get("cameras", []):
        cam_id = str(item.get("id", "")).strip()
        if not cam_id:
            continue

        enabled = bool(item.get("enabled", True))
        if not enabled:
            state = CameraRuntimeState(status=CameraStatus.OFFLINE, reason="disabled in config")
            CAMERA_STATES[cam_id] = state
            continue

        src_type = str(item.get("type", "rtsp")).strip()
        source = item.get("source")
        if source is None:
            raise RuntimeError(f"camera '{cam_id}' has no 'source' field")

        display_name = str(item.get("display_name") or cam_id)
        expected_fps = item.get("expected_fps")
        tracking_send_fps = item.get("tracking_send_fps", TRACKING_SEND_FPS_DEFAULT)
        downscale = str(item.get("downscale") or TRACKING_DOWNSCALE_MODE).strip()
        file_mode = item.get("file_mode")

        if src_type == CameraSourceType.FILE:
            source_str = _expand_source_path(str(source))
        elif src_type == CameraSourceType.WEBCAM:
            source_str = str(source)
        else:
            source_str = str(source)

        cfg = CameraConfig(
            camera_id=cam_id,
            display_name=display_name,
            type=src_type,
            source=source_str,
            expected_fps=float(expected_fps) if expected_fps is not None else None,
            tracking_send_fps=float(tracking_send_fps) if tracking_send_fps is not None else None,
            downscale=downscale,
            enabled=True,
            file_mode=file_mode,
            frame_skip=round(expected_fps / BEHAVIOR_TRAIN_FPS)
        )

        stride = cfg.tracking_stride
        if cfg.expected_fps and cfg.tracking_send_fps and stride is None:
            raise RuntimeError(
                f"camera '{cam_id}': expected_fps={cfg.expected_fps}, "
                f"tracking_send_fps={cfg.tracking_send_fps} => non-integer stride"
            )

        cams.append(cfg)
        CAMERA_STATES.setdefault(cam_id, CameraRuntimeState())

    return cams


def register_kakao_if_available() -> None:
    token = os.getenv("KAKAO_ACCESS_TOKEN")
    if not token:
        return

    url = f"{INFERENCE_SERVER_URL}/agent/register_kakao"
    payload: dict[str, Any] = {
        "agent_code": AGENT_CODE,
        "kakao_access_token": token,
        "note": f"registered by agent '{AGENT_CODE}'",
    }
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        print(f"[AGENT] Kakao token registered for agent_code={AGENT_CODE}")
    except Exception as e:
        print(f"[AGENT] Kakao token register failed: {e}")


def _set_camera_state(cam_id: str, **updates: Any) -> None:
    state = CAMERA_STATES.setdefault(cam_id, CameraRuntimeState())
    for k, v in updates.items():
        setattr(state, k, v)


def _update_latest_frame(cam_id: str, frame: np.ndarray) -> None:
    with LATEST_FRAMES_LOCK:
        LATEST_FRAMES[cam_id] = frame.copy()


# =========================================================
# HLS 스트리머 (기존 구조 유지, 기본은 비활성)
# =========================================================


@dataclass
class HlsStreamer:
    camera: CameraConfig
    ffmpeg_proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not HLS_ENABLED:
            return

        HLS_ROOT.mkdir(parents=True, exist_ok=True)
        self.camera.hls_dir.mkdir(parents=True, exist_ok=True)

        for p in self.camera.hls_dir.glob("*"):
            if p.is_file():
                p.unlink()

        input_arg = self.camera.source

        cmd = ["ffmpeg", "-y"]

        if looks_live_src(input_arg):
            cmd += ["-rtsp_transport", "tcp"]
        else:
            cmd.append("-re")

        cmd += [
            "-i",
            input_arg,
            "-c:v",
            "copy",
            "-an",
            "-f",
            "hls",
            "-hls_time",
            "1",
            "-hls_list_size",
            "5",
            "-hls_flags",
            "delete_segments+program_date_time",
            str(self.camera.hls_playlist),
        ]

        print(f"[HLS:{self.camera.camera_id}] ffmpeg 시작: {' '.join(cmd)}")
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        proc = self.ffmpeg_proc
        if proc is None:
            return
        print(f"[HLS:{self.camera.camera_id}] stopping ffmpeg (pid={proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        self.ffmpeg_proc = None


# =========================================================
# Pose + 서버 연동 워커 (+ Tracking 프레임 전송)
# =========================================================


class PoseAgentWorker(threading.Thread):
    def __init__(
        self,
        config: CameraConfig,
        server_url: str,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.server_url = server_url.rstrip("/")
        self.session = session or requests.Session()

        self._stop_flag = threading.Event()
        self._buf: deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._ts_buf: deque[float] = deque(maxlen=self.config.window_size)

        self._http_timeout = REQUEST_TIMEOUT_SEC

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        cfg = self.config

        if cfg.type == CameraSourceType.WEBCAM:
            try:
                src: Any = int(cfg.source)
            except ValueError:
                src = cfg.source
        else:
            src = cfg.source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[{cfg.camera_id}] failed to open source: {cfg.source}")
            _set_camera_state(cfg.camera_id, status=CameraStatus.ERROR, reason="failed to open source")
            return

        is_live = cfg.is_live
        mono = time.monotonic

        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frame_idx = 0
        win_idx = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = cfg.expected_fps or 3.0
        if fps <= 1e-3:
            fps = 3.0

        tracking_stride = cfg.tracking_stride  # 정수 프레임 간격 또는 None

        print(
            f"[{cfg.camera_id}] start: src='{cfg.source}', is_live={is_live}, fps≈{fps:.2f}, "
            f"tracking_send_fps={cfg.tracking_send_fps}, tracking_stride={tracking_stride}, "
            f"downscale={cfg.downscale}"
        )
        _set_camera_state(cfg.camera_id, status=CameraStatus.STARTING, reason=None)

        while not self._stop_flag.is_set():
            loop_start = mono()
            ok, bgr = cap.read()
            now = mono()

            if not ok:
                if is_live:
                    time.sleep(0.1)
                    continue
                else:
                    if cfg.type == CameraSourceType.FILE and cfg.file_mode == "loop":
                        cap.release()
                        cap = cv2.VideoCapture(cfg.source)
                        continue
                    else:
                        print(f"[{cfg.camera_id}] end of file / no more frames")
                        _set_camera_state(cfg.camera_id, status=CameraStatus.OFFLINE, reason="eof")
                        break

            h, w, _ = bgr.shape
            _set_camera_state(cfg.camera_id, status=CameraStatus.ONLINE, width=w, height=h)
            _update_latest_frame(cfg.camera_id, bgr)

            frame_idx += 1

            if cfg.frame_skip > 1 and (frame_idx % cfg.frame_skip) != 0:
                # pose / tracking 둘 다 이 frame_skip 이후 프레임 기준으로 처리
                continue

            # Pose 입력 해상도 리사이즈
            scale = cfg.resize_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(bgr, (new_w, new_h))

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = mp_pose.process(rgb)
            rgb.flags.writeable = True

            if res and res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kpt = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in lm],
                    dtype=np.float32,
                )
            else:
                kpt = np.full((33, 4), np.nan, np.float32)

            self._buf.append(kpt)
            self._ts_buf.append(now)


            # 1) Pose 윈도우 -> /behavior/analyze_pose
            if len(self._buf) >= cfg.window_size:
                window_kpt = np.stack(list(self._buf), axis=0)  # (T, 33, 4)
                start_ts = self._ts_buf[0]
                end_ts = self._ts_buf[-1]

                has_pose = bool(np.isfinite(window_kpt).any())

                # 1) Prep v3 원형과 동일한 feature 추출 (ffill/bfill 포함)
                feat = features_from_pose_seq(window_kpt)  # (T, 169)

                # 2) 0-프레임-윈도우 판정
                #    - features_from_pose_seq는 NaN→ffill/bfill→NaN→0까지 처리된 상태라
                #      전체가 사실상 0이면 "의미 있는 포즈가 없다"고 간주할 수 있다.

                win_idx += 1
                self._post_window(
                    window_index=win_idx,
                    window_kpt=window_kpt,
                    window_feat=feat,
                    has_pose=has_pose,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )

                # 슬라이딩 윈도우 stride 만큼 앞으로 밀기
                for _ in range(cfg.window_stride):
                    if self._buf:
                        self._buf.popleft()
                    if self._ts_buf:
                        self._ts_buf.popleft()

            # 2) Tracking 프레임 -> /tracking (다운 샘플링 + 프레임 다운샘플링)
            if tracking_stride and tracking_stride > 0 and cfg.tracking_send_fps and cfg.tracking_send_fps > 0:
                if frame_idx % tracking_stride == 0:
                    track_frame = bgr
                    if cfg.downscale == "half":
                        # 가로/세로를 정확히 1/2로 줄여서 (픽셀 단위 정수 연산)
                        track_frame = cv2.resize(
                            bgr,
                            (w // 2, h // 2),
                            interpolation=cv2.INTER_AREA,
                        )
                    self._post_tracking_frame(track_frame, frame_idx)

            # 3) 라이브 소스면 FPS에 맞춰 슬립
            if is_live:
                loop_end = mono()
                elapsed = loop_end - loop_start
                target_dt = 1.0 / max(fps, 1e-3)
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        cap.release()
        mp_pose.close()
        print(f"[{cfg.camera_id}] worker stopped")


    def _post_window(
        self,
        window_index: int,
        window_kpt: np.ndarray,
        window_feat: np.ndarray,
        has_pose: bool,
        start_ts: float,
        end_ts: float,
    ) -> None:
        """
        Pose 윈도우와 대응되는 feature 윈도우 및 has_pose 플래그를 서버로 전송한다.
        - window_kpt: (T,33,4)
        - window_feat: (T,feat_dim)  예: feat_dim=169
        """
        url = f"{self.server_url}/behavior/analyze_pose"

        payload = PoseWindowRequest(
            agent_code=AGENT_CODE,
            camera_id=self.config.camera_id,
            source_id=self.config.source_id,
            window_index=window_index,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            features=window_feat.tolist(),
            # 새로 추가: 0-프레임-윈도우 여부
            has_pose=has_pose,
        )

        try:
            resp = self.session.post(
                url,
                json=payload.model_dump(),
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
        except ValueError as e:
            print(f"[{self.config.camera_id}] HTTP error: {e}")
            return
        except requests.RequestException as e:
            print(f"[{self.config.camera_id}] HTTP error: {e}")
            return

        try:
            data = resp.json()
            inf = InferenceResult(**data)
        except Exception as e:
            print(f"[{self.config.camera_id}] response parse error: {e}")
            return

        print(
            f"[{self.config.camera_id}] "
            f"win={inf.window_index} "
            f"is_anom={inf.is_anomaly} "
            f"det={inf.det_prob:.3f} "
            f"top={inf.top_label}({inf.top_prob:.3f})"
        )

    def _post_tracking_frame(self, frame_bgr: np.ndarray, frame_index: int) -> None:
        """
        /tracking 엔드포인트로 다운샘플링된 프레임을 전송한다.
        - frame_bgr: (H,W,3) BGR 프레임 (이미 downscale 적용된 상태일 수 있음)
        - frame_index: 원본 캡처 기준 프레임 인덱스
        """
        url = f"{self.server_url}/tracking"

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            print(f"[{self.config.camera_id}] tracking: JPEG encode failed")
            return

        files = {
            "file": ("frame.jpg", buf.tobytes(), "image/jpeg"),
        }
        data = {
            "agent_code": AGENT_CODE,
            "camera_id": self.config.camera_id,
            "timestamp": str(time.time()),
            "frame_index": str(frame_index),
        }

        try:
            resp = self.session.post(
                url,
                data=data,
                files=files,
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[{self.config.camera_id}] tracking HTTP error: {e}")
            return


# =========================================================
# FastAPI 앱 + MJPEG / HLS 서빙
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_from_env()
    yield
    stop_all()


app = FastAPI(
    title="Pose Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if HLS_ENABLED:
    app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")

_workers: list[PoseAgentWorker] = []
_hls_streamers: list[HlsStreamer] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cameras", response_model=list[CameraSummary])
def list_cameras() -> list[CameraSummary]:
    items: list[CameraSummary] = []
    for cam_id, cfg in CAMERA_CONFIGS.items():
        state = CAMERA_STATES.get(cam_id, CameraRuntimeState())
        mjpeg_url = None
        if MJPEG_ENABLED:
            mjpeg_url = f"/streams/{cam_id}.mjpeg"

        hls_url = None
        if HLS_ENABLED:
            hls_url = f"{HLS_BASE_PATH.rstrip('/')}/{cam_id}/index.m3u8"

        items.append(
            CameraSummary(
                id=cam_id,
                display_name=cfg.display_name,
                type=cfg.type,
                status=state.status,
                reason=state.reason,
                mjpeg_url=mjpeg_url,
                hls_url=hls_url,
                width=state.width,
                height=state.height,
                expected_fps=cfg.expected_fps,
                tracking_send_fps=cfg.tracking_send_fps,
                downscale=cfg.downscale,
            )
        )
    return items


_MJPEG_BOUNDARY = "frame"


def _mjpeg_generator(camera_id: str):
    if not MJPEG_ENABLED:
        return

    while True:
        with LATEST_FRAMES_LOCK:
            frame = LATEST_FRAMES.get(camera_id)

        if frame is None:
            time.sleep(0.1)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.1)
            continue

        data = jpeg.tobytes()
        yield (
            b"--"
            + _MJPEG_BOUNDARY.encode("ascii")
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + b"Content-Length: "
            + str(len(data)).encode("ascii")
            + b"\r\n\r\n"
            + data
            + b"\r\n"
        )

        time.sleep(0.04)


@app.get("/streams/{camera_id}.mjpeg")
def mjpeg_stream(camera_id: str):
    if not MJPEG_ENABLED:
        raise HTTPException(status_code=404, detail="MJPEG disabled")

    if camera_id not in CAMERA_CONFIGS:
        raise HTTPException(status_code=404, detail=f"unknown camera_id: {camera_id}")

    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
    )


def start_from_env() -> None:
    print(f"[AGENT] starting with INFERENCE_SERVER_URL={INFERENCE_SERVER_URL}, AGENT_CODE={AGENT_CODE}")
    ensure_ffmpeg_available()

    cams = load_camera_configs()
    if not cams:
        print("[AGENT] no cameras configured")
        return

    for cfg in cams:
        CAMERA_CONFIGS[cfg.camera_id] = cfg

    register_kakao_if_available()

    print("[AGENT] starting workers and (optional) HLS streamers for cameras:")
    for cfg in cams:
        print(f"  - {cfg.camera_id}: src={cfg.source}")

        streamer = HlsStreamer(cfg)
        streamer.start()
        _hls_streamers.append(streamer)

        worker = PoseAgentWorker(cfg, server_url=INFERENCE_SERVER_URL)
        worker.start()
        _workers.append(worker)


def stop_all() -> None:
    for w in _workers:
        w.stop()
    for s in _hls_streamers:
        s.stop()

import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import mediapipe as mp
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =========================================================
# .env 로딩 (agent 디렉터리 기준)
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# =========================================================
# 환경 변수 / 기본 설정
# =========================================================

INFERENCE_SERVER_URL_DEFAULT = "http://localhost:8000"
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", INFERENCE_SERVER_URL_DEFAULT)

SOURCES_ENV = os.getenv("AGENT_SOURCES", "videos/cam01.mp4,videos/cam02.mp4")
CAMERA_IDS_ENV = os.getenv("AGENT_CAMERA_IDS", "cam01,cam02")

DEFAULT_WINDOW_SIZE = int(os.getenv("AGENT_WINDOW_SIZE", "16"))
DEFAULT_STRIDE = int(os.getenv("AGENT_STRIDE", "4"))
DEFAULT_RESIZE_WIDTH = int(os.getenv("AGENT_RESIZE_WIDTH", "640"))
DEFAULT_FRAME_SKIP = int(os.getenv("AGENT_FRAME_SKIP", "1"))  # 1이면 매 프레임 처리

HLS_ROOT = Path(os.getenv("AGENT_HLS_ROOT", "./hls")).resolve()


# =========================================================
# 서버와 맞춘 요청/응답 스키마
# =========================================================

class PoseWindowRequest(BaseModel):
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None
    # keypoints: (T, 33, 4) 리스트
    #   [[ [x,y,z,vis], ...33 ], ...T ]
    keypoints: list[list[list[float]]]


class InferenceResult(BaseModel):
    camera_id: str
    source_id: str | None
    window_index: int
    window_start_ts: float | None
    window_end_ts: float | None
    is_anomaly: bool
    det_prob: float
    top_label: str | None
    top_prob: float
    prob: dict[str, float]


# =========================================================
# 설정/유틸
# =========================================================

def looks_live_src(src: str) -> bool:
    return bool(re.match(r"^(rtsp|rtsps|rtmp|http|https|rtp)://", src, re.I))


@dataclass
class CameraConfig:
    camera_id: str
    source: str            # mp4 경로 또는 rtsp://...
    source_id: str         # DB/로그에 찍을 식별자
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    resize_width: int = DEFAULT_RESIZE_WIDTH
    frame_skip: int = DEFAULT_FRAME_SKIP

    @property
    def is_live(self) -> bool:
        return looks_live_src(self.source)

    @property
    def hls_root(self) -> Path:
        return HLS_ROOT

    @property
    def hls_dir(self) -> Path:
        return self.hls_root / self.camera_id

    @property
    def hls_playlist(self) -> Path:
        return self.hls_dir / "index.m3u8"


def parse_camera_env() -> list[CameraConfig]:
    """
    AGENT_SOURCES / AGENT_CAMERA_IDS 환경변수를 파싱해서
    CameraConfig 리스트로 변환한다.

    - 스트림 주소(rtsp/http/rtmp 등)는 그대로 사용
    - 파일 경로는:
        * ~ (홈 경로) 확장
        * 환경변수($HOME 등) 확장
        * BASE_DIR(= agent_app.py 가 있는 디렉터리) 기준 절대경로로 변환
    """
    src_tokens = [s.strip() for s in SOURCES_ENV.split(",") if s.strip()]
    id_tokens = [s.strip() for s in CAMERA_IDS_ENV.split(",") if s.strip()]

    n = min(len(src_tokens), len(id_tokens))
    if n == 0:
        raise RuntimeError("AGENT_SOURCES / AGENT_CAMERA_IDS 설정이 비었습니다.")

    configs: list[CameraConfig] = []

    for i in range(n):
        raw_src = src_tokens[i]
        cam_id = id_tokens[i]

        # 1) 스트림 주소(rtsp/http/rtmp 등)라면 그대로 사용
        if looks_live_src(raw_src):
            source_str = raw_src
            source_id = raw_src
        else:
            # 2) 파일 경로라면:
            #   - ~ (홈) 확장
            #   - 환경변수($HOME 등) 확장
            #   - BASE_DIR 기준 절대 경로로 변환
            expanded = os.path.expandvars(os.path.expanduser(raw_src))
            src_path = Path(expanded)

            if not src_path.is_absolute():
                src_path = (BASE_DIR / src_path).resolve()

            source_str = str(src_path)
            source_id = source_str  # 필요하면 나중에 basename 등으로 바꿔도 됨

        cfg = CameraConfig(
            camera_id=cam_id,
            source=source_str,
            source_id=source_id,
        )
        configs.append(cfg)

    return configs


def ensure_ffmpeg_available() -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        msg = (
            "[Agent] ffmpeg 실행 파일을 찾을 수 없습니다.\n"
            "에이전트 환경에 ffmpeg를 설치한 뒤 다시 실행해 주세요.\n"
            "예시)\n"
            "  - Windows (scoop): scoop install ffmpeg\n"
            "  - macOS (brew):   brew install ffmpeg\n"
            "  - Ubuntu:         sudo apt install ffmpeg\n"
        )
        raise RuntimeError(msg)

    try:
        subprocess.run(
            [exe, "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise RuntimeError(
            f"[Agent] ffmpeg 호출에 실패했습니다: {e}\n"
            "ffmpeg 설치 및 PATH 설정을 확인해 주세요."
        )


# =========================================================
# HLS 스트리머
# =========================================================

@dataclass
class HlsStreamer:
    camera: CameraConfig
    ffmpeg_proc: subprocess.Popen | None = None

    def start(self) -> None:
        HLS_ROOT.mkdir(parents=True, exist_ok=True)
        self.camera.hls_dir.mkdir(parents=True, exist_ok=True)

        # 기존 세그먼트 정리
        for p in self.camera.hls_dir.glob("*"):
            if p.is_file():
                p.unlink()

        input_arg = self.camera.source

        # 공통 옵션
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "warning",
        ]

        # 라이브/파일에 따른 입력 옵션
        if looks_live_src(input_arg):
            # RTSP/HTTP 등 스트림 입력일 경우
            cmd += ["-rtsp_transport", "tcp"]
        else:
            # 파일 입력일 경우 실시간처럼 흘려보내기
            cmd.append("-re")

        # 공통 HLS 출력 옵션
        cmd += [
            "-i", input_arg,
            "-c:v", "copy",
            "-an",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments+omit_endlist",
            str(self.camera.hls_playlist),
        ]

        print(f"[HLS:{self.camera.camera_id}] starting ffmpeg: {' '.join(cmd)}")
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
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
# PoseAgentWorker: 한 카메라에 대한 캡처 + Mediapipe + 서버 POST
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

        self._http_timeout = 5.0

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        cfg = self.config
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            print(f"[{cfg.camera_id}] failed to open source: {cfg.source}")
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
            fps = 3.0  # fallback
        print(
            f"[{cfg.camera_id}] start: src='{cfg.source}', "
            f"is_live={is_live}, fps≈{fps:.2f}"
        )

        while not self._stop_flag.is_set():
            loop_start = mono()
            ok, bgr = cap.read()
            now = mono()

            if not ok:
                if is_live:
                    time.sleep(0.1)
                    continue
                else:
                    print(f"[{cfg.camera_id}] EOF")
                    break

            frame_idx += 1

            if cfg.frame_skip > 1 and (frame_idx % cfg.frame_skip) != 0:
                continue

            h, w, _ = bgr.shape
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
                # 포즈가 안 잡히는 프레임은 NaN으로 채움
                kpt = np.full((33, 4), np.nan, np.float32)

            self._buf.append(kpt)
            self._ts_buf.append(now)

            if len(self._buf) < cfg.window_size:
                # 윈도우가 다 안 찼으면 대기
                continue

            window_kpt = np.stack(list(self._buf), axis=0)  # (T, 33, 4)
            start_ts = self._ts_buf[0]
            end_ts = self._ts_buf[-1]

            win_idx += 1
            self._post_window(
                window_index=win_idx,
                window_kpt=window_kpt,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            # 슬라이딩 윈도우 stride 만큼 앞으로 밀기
            for _ in range(cfg.stride):
                if self._buf:
                    self._buf.popleft()
                if self._ts_buf:
                    self._ts_buf.popleft()

            # 라이브 소스면 FPS에 맞춰 슬립
            if is_live:
                loop_end = mono()
                elapsed = loop_end - loop_start
                target = 1.0 / fps
                if elapsed < target:
                    time.sleep(target - elapsed)

        cap.release()
        mp_pose.close()
        print(f"[{cfg.camera_id}] stopped.")

    def _post_window(
        self,
        window_index: int,
        window_kpt: np.ndarray,
        start_ts: float,
        end_ts: float,
    ) -> None:
        url = f"{self.server_url}/behavior/analyze_pose"

        # ⚠ 여기서는 여전히 NaN이 그대로 들어가면 JSON 직렬화 에러가 날 수 있음.
        #   (필요하면 np.nan_to_num 으로 치환하거나, NaN 포함 윈도우를 스킵하는 로직을
        #    추가로 넣을 수 있음)
        payload = PoseWindowRequest(
            camera_id=self.config.camera_id,
            source_id=self.config.source_id,
            window_index=window_index,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            keypoints=window_kpt.tolist(),
        )

        try:
            resp = self.session.post(
                url,
                json=payload.model_dump(),
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
        except ValueError as e:
            # NaN 등으로 인한 JSON 직렬화 에러
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


# =========================================================
# FastAPI 앱
# =========================================================

app = FastAPI(title="Pose Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")

_workers: list[PoseAgentWorker] = []
_hls_streamers: list[HlsStreamer] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def start_from_env() -> None:
    ensure_ffmpeg_available()
    configs = parse_camera_env()

    print("[AGENT] starting workers and HLS streamers for cameras:")
    for cfg in configs:
        print(f"  - {cfg.camera_id}: src={cfg.source}")

        # HLS 스트리머
        streamer = HlsStreamer(cfg)
        streamer.start()
        _hls_streamers.append(streamer)

        # Pose 워커
        worker = PoseAgentWorker(cfg, server_url=INFERENCE_SERVER_URL)
        worker.start()
        _workers.append(worker)


def stop_all() -> None:
    for w in _workers:
        w.stop()
    for s in _hls_streamers:
        s.stop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    start_from_env()
    yield
    # shutdown
    stop_all()


app.router.lifespan_context = lifespan

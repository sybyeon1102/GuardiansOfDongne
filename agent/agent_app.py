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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =========================================================
# .env 로딩 (client 디렉터리 기준)
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
    # (T,33,4): [[ [x,y,z,vis], ...33 ], ...T ]
    keypoints: list[list[list[float]]]


class BehaviorResultResponse(BaseModel):
    camera_id: str
    source_id: str | None = None
    window_index: int
    window_start_ts: float | None = None
    window_end_ts: float | None = None

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

    hls_root: Path = HLS_ROOT

    @property
    def hls_dir(self) -> Path:
        return self.hls_root / self.camera_id

    @property
    def hls_playlist(self) -> Path:
        return self.hls_dir / "index.m3u8"


def parse_camera_env() -> list[CameraConfig]:
    src_tokens = [s.strip() for s in SOURCES_ENV.split(",") if s.strip()]
    id_tokens = [s.strip() for s in CAMERA_IDS_ENV.split(",") if s.strip()]

    n = min(len(src_tokens), len(id_tokens))
    if n == 0:
        raise RuntimeError("AGENT_SOURCES / AGENT_CAMERA_IDS 설정이 비었습니다.")

    configs: list[CameraConfig] = []
    for i in range(n):
        src = src_tokens[i]
        cam_id = id_tokens[i]
        cfg = CameraConfig(
            camera_id=cam_id,
            source=src,
            source_id=src,
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

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", input_arg,
            "-c:v", "copy",
            "-c:a", "copy",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments+omit_endlist",
            str(self.camera.hls_playlist),
        ]

        # 파일 소스일 경우 실시간처럼 흘려보내기
        if not looks_live_src(input_arg):
            cmd.insert(1, "-re")

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
        http_timeout: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.server_url = server_url.rstrip("/")
        self.http_timeout = float(http_timeout)
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def _make_pose_estimator(self) -> mp.solutions.pose.Pose:
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # legacy 기본값
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _post_window(
        self,
        window_kpt: np.ndarray,
        window_index: int,
        start_ts: float | None,
        end_ts: float | None,
    ) -> None:
        if window_kpt.ndim != 3 or window_kpt.shape[1:] != (33, 4):
            print(
                f"[{self.config.camera_id}] invalid keypoints shape: "
                f"{window_kpt.shape}, skip"
            )
            return

        payload = PoseWindowRequest(
            camera_id=self.config.camera_id,
            source_id=self.config.source_id,
            window_index=window_index,
            window_start_ts=start_ts,
            window_end_ts=end_ts,
            keypoints=window_kpt.tolist(),
        )

        url = f"{self.server_url}/behavior/analyze_pose"
        try:
            resp = requests.post(
                url,
                json=payload.model_dump(),
                timeout=self.http_timeout,
            )
        except Exception as e:
            print(f"[{self.config.camera_id}] HTTP error: {e}")
            return

        if resp.status_code != 200:
            print(
                f"[{self.config.camera_id}] server HTTP {resp.status_code}: "
                f"{resp.text}"
            )
            return

        try:
            data = resp.json()
            result = BehaviorResultResponse(**data)
            print(
                f"[{self.config.camera_id}] "
                f"win={result.window_index} "
                f"is_anom={result.is_anomaly} "
                f"det={result.det_prob:.3f} "
                f"top={result.top_label}({result.top_prob:.3f})"
            )
        except Exception:
            # 응답 스키마가 조금 달라도 에이전트 입장에선 치명적이지 않다.
            pass

    def run(self) -> None:
        cfg = self.config
        src = cfg.source

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{cfg.camera_id}] failed to open source: {src!r}")
            return

        is_live = looks_live_src(src)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0.0:
            fps = 25.0
        frame_dt = 1.0 / fps if fps > 0 else 0.0

        print(
            f"[{cfg.camera_id}] start: src={src!r}, "
            f"is_live={is_live}, fps≈{fps:.2f}"
        )

        pose = self._make_pose_estimator()
        buf: deque[np.ndarray] = deque(maxlen=cfg.window_size)
        frame_idx = 0
        window_index = 0

        mono = time.monotonic
        t0 = mono()

        while not self._stop_flag.is_set():
            loop_start = time.perf_counter()
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
            real_ts = now - t0  # 필요시 window_start_ts/end_ts로 활용 가능

            # frame_skip 적용
            if cfg.frame_skip > 1 and (frame_idx % cfg.frame_skip != 0):
                if not is_live and frame_dt > 0:
                    spent = time.perf_counter() - loop_start
                    to_sleep = frame_dt - spent
                    if to_sleep > 0:
                        time.sleep(to_sleep)
                continue

            # 리사이즈
            if cfg.resize_width > 0:
                h, w = bgr.shape[:2]
                s = cfg.resize_width / float(w)
                bgr_proc = cv2.resize(bgr, (cfg.resize_width, int(h * s)))
            else:
                bgr_proc = bgr

            rgb = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res and res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kpt = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in lm],
                    dtype=np.float32,
                )
            else:
                kpt = np.full((33, 4), np.nan, np.float32)

            buf.append(kpt)

            do_infer = (len(buf) == cfg.window_size) and (
                frame_idx % cfg.stride == 0
            )
            if do_infer:
                window_index += 1
                window_kpt = np.stack(list(buf), axis=0)  # (T,33,4)
                self._post_window(
                    window_kpt,
                    window_index=window_index,
                    start_ts=None,
                    end_ts=None,
                )

            if not is_live and frame_dt > 0:
                spent = time.perf_counter() - loop_start
                to_sleep = frame_dt - spent
                if to_sleep > 0:
                    time.sleep(to_sleep)

        cap.release()
        print(f"[{cfg.camera_id}] stopped.")


# =========================================================
# 전역 워커 / 스트리머 + lifespan
# =========================================================

_workers: list[PoseAgentWorker] = []
_hls_streamers: list[HlsStreamer] = []


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
    try:
        yield
    finally:
        # shutdown
        stop_all()


# =========================================================
# FastAPI 앱 정의
# =========================================================

app = FastAPI(title="GoD Pose Agent", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount(
    "/hls",
    StaticFiles(directory=str(HLS_ROOT), html=False, check_dir=False),
    name="hls",
)

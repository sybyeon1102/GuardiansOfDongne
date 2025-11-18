"""
RTSP 카메라 + backend /behavior/analyze 를 연결해 주는
간단한 스트리밍 서버입니다.

Simple streaming server that connects an RTSP camera to the backend
/behavior/analyze API and exposes /stream and /probs endpoints for the
existing index.html UI.
"""

import contextlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Deque, Iterable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ========= 환경 설정 / Environment =========

ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# RTSP URL (노트북 카메라 사용 시 "0")
RTSP_URL_RAW = os.getenv("RTSP_URL", "0")
RTSP_URL: str | int
if RTSP_URL_RAW == "0":
    RTSP_URL = 0
else:
    RTSP_URL = RTSP_URL_RAW

# Backend HTTP endpoint
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
BACKEND_ANALYZE_PATH = os.getenv("BACKEND_ANALYZE_PATH", "/behavior/analyze")

# 스트리밍 설정
STREAM_WINDOW_SIZE = int(os.getenv("STREAM_WINDOW_SIZE", "16"))
STREAM_FPS = float(os.getenv("STREAM_FPS", "8.0"))
STREAM_RESIZE_WIDTH_RAW = os.getenv("STREAM_RESIZE_WIDTH")
STREAM_RESIZE_WIDTH = int(STREAM_RESIZE_WIDTH_RAW) if STREAM_RESIZE_WIDTH_RAW else None
JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "80"))
OVERLAY_STATS = os.getenv("STREAM_OVERLAY_STATS", "0") == "1"

# ========= 전역 상태 / Global state =========

lock = threading.Lock()
stop_event = threading.Event()

latest_frame_bgr: np.ndarray | None = None
last_probs: list[float] | None = None
last_events: list[str] | None = None

pose: mp.solutions.pose.Pose | None = None


# ========= 유틸 함수 / Utilities =========


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """
    NaN 값을 앞/뒤 방향으로 보간합니다.

    Forward/backward fill NaN values in a 2D array.
    """
    T, D = arr.shape
    out = arr.copy()

    last = np.zeros(D, np.float32)
    has = np.zeros(D, bool)

    for t in range(T):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]

    last[:] = 0.0
    has[:] = False

    for t in range(T - 1, -1, -1):
        nz = ~np.isnan(out[t])
        last[nz] = out[t, nz]
        has |= nz
        miss = np.isnan(out[t]) & has
        out[t, miss] = last[miss]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def features_from_buf(buf: Iterable[np.ndarray]) -> np.ndarray:
    """
    legacy stream_lstm_app.py 와 동일한 방식으로
    (T, 33, 4) 키포인트 버퍼에서 (T, 169) 피처를 생성합니다.

    Build (T, 169) LSTM input features from a buffer of (33, 4) keypoints.
    """
    kpts = np.stack(list(buf), axis=0)  # (T, 33, 4)
    T = kpts.shape[0]

    xy = kpts[:, :, :2].reshape(T, -1)  # (T, 66)
    vis = kpts[:, :, 3:4].reshape(T, -1)  # (T, 33)

    xy = _ffill_bfill(xy).reshape(T, 33, 2)
    vis = _ffill_bfill(vis).reshape(T, 33, 1)

    hip = np.mean(xy[:, [23, 24], :], axis=1)
    sh = np.mean(xy[:, [11, 12], :], axis=1)
    sc = np.linalg.norm(sh - hip, axis=1, keepdims=True)
    sc[sc < 1e-3] = 1.0

    xy_n = (xy - hip[:, None, :]) / sc[:, None, :]
    vel = np.diff(xy_n, axis=0, prepend=xy_n[:1])

    def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        n1[n1 == 0.0] = 1e-6
        n2[n2 == 0.0] = 1e-6
        cos = (v1 * v2).sum(-1) / (n1 * n2)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    def pick(i: int) -> np.ndarray:
        return xy_n[:, i, :]

    angs = np.stack(
        [
            ang(pick(11), pick(13), pick(15)),
            ang(pick(12), pick(14), pick(16)),
            ang(pick(23), pick(25), pick(27)),
            ang(pick(24), pick(26), pick(28)),
        ],
        axis=1,
    )  # (T, 4)

    feat = np.concatenate(
        [xy_n.reshape(T, -1), vel.reshape(T, -1), angs, vis.reshape(T, -1)],
        axis=1,
    )  # (T, 169)

    return np.clip(feat.astype(np.float32), -10.0, 10.0)


def draw_pose(
    img: np.ndarray,
    landmarks,
    visibility_thr: float = 0.5,
    r: int = 3,
    th: int = 2,
) -> None:
    """
    MediaPipe 포즈 랜드마크를 이미지 위에 그립니다.

    Draw pose landmarks from MediaPipe on the image.
    """
    h, w = img.shape[:2]

    def to_px(lm) -> tuple[int, int]:
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        return x, y

    connections = mp.solutions.pose.POSE_CONNECTIONS

    for a, b in connections:
        la, lb = landmarks[a], landmarks[b]
        if (la.visibility >= visibility_thr) and (lb.visibility >= visibility_thr):
            xa, ya = to_px(la)
            xb, yb = to_px(lb)
            cv2.line(img, (xa, ya), (xb, yb), (0, 255, 255), th, cv2.LINE_AA)

    for lm in landmarks:
        if lm.visibility >= visibility_thr:
            x, y = to_px(lm)
            cv2.circle(img, (x, y), r, (255, 200, 0), -1, cv2.LINE_AA)


def draw_prob_bars(
    img: np.ndarray,
    events: Iterable[str],
    probs: Iterable[float],
    *,
    x: int = 20,
    y: int = 20,
    w: int = 260,
    h: int = 18,
    gap: int = 6,
) -> None:
    """
    이벤트별 확률 막대를 오버레이합니다.

    Draw probability bars overlay for each event.
    """
    for i, (ev, p) in enumerate(zip(events, probs)):
        y0 = y + i * (h + gap)
        cv2.rectangle(img, (x, y0), (x + w, y0 + h), (60, 60, 60), 1)
        width = int(w * float(p))
        cv2.rectangle(
            img,
            (x, y0),
            (x + width, y0 + h),
            (0, 0, 255) if p >= 0.5 else (0, 180, 0),
            -1,
        )
        cv2.putText(
            img,
            f"{ev[:14]} {p:.2f}",
            (x + w + 10, y0 + h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _make_pose() -> mp.solutions.pose.Pose:
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=(0 if not os.environ.get("POSE_COMPLEX") else int(os.environ["POSE_COMPLEX"])),
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _call_backend_analyze(features: np.ndarray) -> tuple[list[str] | None, list[float] | None]:
    """
    backend /behavior/analyze 엔드포인트를 호출하여
    이벤트 라벨과 확률 벡터를 가져옵니다.

    Call backend /behavior/analyze and return (events, probs).
    """
    url = BACKEND_BASE_URL.rstrip("/") + BACKEND_ANALYZE_PATH
    frames = [
        {
            "index": int(i),
            "features": [float(x) for x in features[i]],
        }
        for i in range(features.shape[0])
    ]

    payload = {"frames": frames}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read()
        result = json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None, None

    events = result.get("events")
    scores = result.get("scores")

    if not isinstance(events, list) or not isinstance(scores, list):
        return None, None

    return [str(ev) for ev in events], [float(s) for s in scores]


def _capture_and_infer_loop() -> None:
    """
    RTSP 스트림에서 프레임을 읽고, pose → feature → backend 분석 → 오버레이를 수행합니다.

    Capture frames from RTSP, run pose → features → backend analyze,
    and update global frame/probability state.
    """
    global latest_frame_bgr, last_probs, last_events, pose

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    pose = _make_pose()

    buf: Deque[np.ndarray] = deque(maxlen=STREAM_WINDOW_SIZE)

    if isinstance(RTSP_URL, int):
        cap = cv2.VideoCapture(RTSP_URL)
    else:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("[ERROR] cannot open RTSP/camera:", RTSP_URL)
        return

    tick = 0.0
    min_interval = max(1.0 / STREAM_FPS, 0.01)

    while not stop_event.is_set():
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        if STREAM_RESIZE_WIDTH:
            h, w = bgr.shape[:2]
            scale = STREAM_RESIZE_WIDTH / float(w)
            bgr = cv2.resize(bgr, (STREAM_RESIZE_WIDTH, int(h * scale)))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb) if pose else None

        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            draw_pose(bgr, lm, visibility_thr=0.5, r=3, th=2)
            kpt = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float32)
        else:
            kpt = np.full((33, 4), np.nan, np.float32)

        buf.append(kpt)

        events: list[str] | None = None
        probs: list[float] | None = None

        now = time.time()
        if len(buf) == STREAM_WINDOW_SIZE and (now - tick) >= min_interval:
            tick = now
            feat = features_from_buf(buf)
            events, probs = _call_backend_analyze(feat)

        if probs is not None and events is not None:
            if OVERLAY_STATS:
                draw_prob_bars(bgr, events, probs, x=20, y=20, w=260, h=18, gap=6)
            with lock:
                last_probs = list(probs)
                last_events = list(events)

        with lock:
            latest_frame_bgr = bgr

    cap.release()
    if pose is not None:
        pose.close()


def mjpeg_generator():
    boundary = b"--frame"
    while not stop_event.is_set():
        with lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
        if frame is None:
            time.sleep(0.03)
            continue
        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            continue
        data = jpeg.tobytes()
        yield b"%b\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (boundary, len(data))
        yield data + b"\r\n"
        time.sleep(0.01)


# ========= FastAPI 앱 / FastAPI app =========


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """
    FastAPI lifespan 훅에서 캡처/추론 스레드를 시작/종료합니다.

    Start/stop the capture/inference worker thread using FastAPI lifespan.
    """
    worker = threading.Thread(target=_capture_and_infer_loop, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        # 캡처 루프가 종료될 시간을 잠깐 줍니다.
        # Give the worker a moment to shut down cleanly.
        worker.join(timeout=2.0)


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    index_path = Path(__file__).resolve().parents[3] / "web" / "index.html"
    return FileResponse(index_path)


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/probs")
def probs() -> JSONResponse:
    with lock:
        events = list(last_events) if last_events else []
        probs = [float(x) for x in last_probs] if last_probs is not None else None
    return JSONResponse({"events": events, "probs": probs, "ok": probs is not None})

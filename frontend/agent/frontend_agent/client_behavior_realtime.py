"""
실시간 웹캠/RTSP 영상을 읽어 포즈를 추출하고,
백엔드 /behavior/analyze API를 호출해 정상/이상 행동을 분석하는 클라이언트입니다.

This script reads webcam/RTSP video, extracts pose keypoints with MediaPipe,
builds LSTM input features, and sends them to the backend /behavior/analyze API.
Optionally, it also sends PROBS-style event logs to /event/payload.
"""

import argparse
import os
import time
from collections import deque
from collections.abc import Deque

import cv2
import mediapipe as mp
import numpy as np
import requests


# ----------------------------
# 전처리 유틸 (modeling.preprocessing.dataset_common 로직을 독립 복제)
# Preprocessing utilities (logic copied from modeling.preprocessing.dataset_common)
# ----------------------------


def fill_missing_values(features: np.ndarray) -> np.ndarray:
    """NaN을 앞/뒤 방향으로 채운 뒤, 남은 NaN/inf는 0으로 채웁니다.
    Forward/backward fill NaNs and replace remaining NaN/inf with zeros.
    """
    time_steps, dim = features.shape
    output = features.copy()

    last_values = np.zeros(dim, np.float32)
    has_value = np.zeros(dim, bool)

    # forward fill
    for t in range(time_steps):
        not_nan = ~np.isnan(output[t])
        last_values[not_nan] = output[t, not_nan]
        has_value |= not_nan
        missing = np.isnan(output[t]) & has_value
        output[t, missing] = last_values[missing]

    # backward fill
    last_values[:] = 0.0
    has_value[:] = False

    for t in range(time_steps - 1, -1, -1):
        not_nan = ~np.isnan(output[t])
        last_values[not_nan] = output[t, not_nan]
        has_value |= not_nan
        missing = np.isnan(output[t]) & has_value
        output[t, missing] = last_values[missing]

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def build_features_from_keypoints(sequence_keypoints: np.ndarray) -> np.ndarray:
    """(T, 33, 4) 형태의 키포인트 배열에서 LSTM 입력용 피처를 구성합니다.
    Build LSTM input features from pose keypoints (T, 33, 4).

    - T: 시간 축 길이 (프레임 수) / number of frames
    - 33: MediaPipe Pose keypoints
    - 4: (x, y, z, visibility)
    """
    time_steps = sequence_keypoints.shape[0]

    xy = sequence_keypoints[:, :, :2].reshape(time_steps, -1)
    visibility = sequence_keypoints[:, :, 3:4].reshape(time_steps, -1)

    xy = fill_missing_values(xy).reshape(time_steps, 33, 2)
    visibility = fill_missing_values(visibility).reshape(time_steps, 33, 1)

    hip = np.mean(xy[:, [23, 24], :], axis=1)
    shoulders = np.mean(xy[:, [11, 12], :], axis=1)
    scale = np.linalg.norm(shoulders - hip, axis=1, keepdims=True)
    scale[scale < 1e-3] = 1.0

    xy_normalized = (xy - hip[:, None, :]) / scale[:, None, :]
    velocity = np.diff(xy_normalized, axis=0, prepend=xy_normalized[:1])

    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """벡터 a-b, c-b 사이의 각도를 계산합니다.
        Compute angle between vectors a-b and c-b.
        """
        v1 = a - b
        v2 = c - b
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        norm1[norm1 == 0.0] = 1e-6
        norm2[norm2 == 0.0] = 1e-6
        cos = (v1 * v2).sum(-1) / (norm1 * norm2)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    def pick(index: int) -> np.ndarray:
        return xy_normalized[:, index, :]

    angles = np.stack(
        [
            compute_angle(pick(11), pick(13), pick(15)),
            compute_angle(pick(12), pick(14), pick(16)),
            compute_angle(pick(23), pick(25), pick(27)),
            compute_angle(pick(24), pick(26), pick(28)),
        ],
        axis=1,
    )

    features = np.concatenate(
        [
            xy_normalized.reshape(time_steps, -1),
            velocity.reshape(time_steps, -1),
            angles,
            visibility.reshape(time_steps, -1),
        ],
        axis=1,
    )

    return np.clip(features.astype(np.float32), -10.0, 10.0)


# ----------------------------
# MediaPipe / 비디오 캡처
# MediaPipe / video capture
# ----------------------------


def parse_arguments_realtime_client() -> argparse.Namespace:
    """실시간 클라이언트용 명령행 인자를 파싱합니다.
    Parse command-line arguments for the realtime client.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Realtime pose-based anomaly client using backend /behavior/analyze.\n"
            "웹캠/RTSP에서 포즈를 추출하고 백엔드 API로 이상행동을 분석합니다."
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (camera index or RTSP/file URL). 기본값: '0' (웹캠)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
        help="Backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="LSTM 윈도우 크기 (프레임 수) / Window size in frames (default: 60)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="프레임 건너뛰기 간격 / Frame stride (default: 5)",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="리사이즈 기준 너비 (픽셀) / Resize width in pixels (default: 640)",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="OpenCV 창에 결과를 시각화합니다. Show OpenCV window.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model_complexity (0/1/2). 기본: 1",
    )
    parser.add_argument(
        "--no-event-log",
        action="store_true",
        help="/event/payload 로 PROBS 이벤트를 보내지 않습니다.",
    )

    return parser.parse_args()


def open_video_capture(source: str) -> cv2.VideoCapture:
    """비디오 캡처를 엽니다 (카메라 인덱스 or URL).
    Open video capture from a camera index or URL.
    """
    if source.isdigit():
        index = int(source)
        capture = cv2.VideoCapture(index)
    else:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    return capture


def create_pose_estimator(model_complexity: int = 1) -> mp.solutions.pose.Pose:
    """MediaPipe Pose 추정기를 생성합니다.
    Create a MediaPipe Pose estimator.
    """
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_keypoints_from_frame(
    frame_bgr: np.ndarray,
    pose_estimator: mp.solutions.pose.Pose,
) -> np.ndarray:
    """단일 프레임에서 포즈 키포인트를 추출합니다.
    Extract pose keypoints (33, 4) from a single frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_estimator.process(frame_rgb)

    if not result.pose_landmarks:
        return np.full((33, 4), np.nan, np.float32)

    landmarks = result.pose_landmarks.landmark
    keypoints = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    )
    return keypoints


# ----------------------------
# 백엔드 호출 / 이벤트 로깅
# Backend calls / event logging
# ----------------------------


def call_backend_behavior_analyze(
    backend_url: str,
    features: np.ndarray,
) -> dict:
    """백엔드 /behavior/analyze 엔드포인트를 호출합니다.
    Call backend /behavior/analyze endpoint.
    """
    url = backend_url.rstrip("/") + "/behavior/analyze"

    frames_payload = [
        {
            "index": int(i),
            "features": [float(x) for x in features[i]],
        }
        for i in range(features.shape[0])
    ]
    payload = {"frames": frames_payload}

    response = requests.post(url, json=payload, timeout=2.0)
    response.raise_for_status()
    return response.json()


def send_event_payload(
    backend_url: str,
    event_type: str,
    events: list[str],
    probs: list[float],
) -> None:
    """백엔드 /event/payload 엔드포인트로 PROBS 스타일 이벤트를 전송합니다.
    Send PROBS-style event payload to backend /event/payload.
    """
    url = backend_url.rstrip("/") + "/event/payload"

    payload = {
        "type": event_type,
        "events": events,
        "probs": probs,
    }

    try:
        response = requests.post(url, json=payload, timeout=2.0)
        response.raise_for_status()
    except requests.RequestException as exc:  # noqa: PERF203
        print(f"[WARN] failed to send event payload: {exc}")


def draw_predictions_on_frame(
    frame_bgr: np.ndarray,
    events: list[str],
    probs: list[float],
) -> None:
    """프레임에 이벤트/확률 정보를 오버레이합니다.
    Draw predicted events and probabilities on the frame.
    """
    h, w = frame_bgr.shape[:2]

    y0 = 30
    dy = 25

    for index, (event, probability) in enumerate(zip(events, probs)):
        text = f"{event}: {probability:.2f}"
        position = (10, y0 + index * dy)
        color = (0, 0, 255) if probability >= 0.5 else (0, 200, 0)
        cv2.putText(
            frame_bgr,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


# ----------------------------
# 메인 루프
# Main loop
# ----------------------------


def run_client_realtime() -> None:
    """실시간 클라이언트 메인 루프.
    Main loop for realtime client.
    """
    arguments = parse_arguments_realtime_client()

    backend_url = arguments.backend_url
    window_size = max(1, arguments.window_size)
    stride = max(1, arguments.stride)

    print(f"[INFO] Using backend: {backend_url}")
    print(f"[INFO] window_size={window_size}, stride={stride}")

    capture = open_video_capture(arguments.source)

    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=arguments.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoints_buffer: Deque[np.ndarray] = deque(maxlen=window_size)

    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[WARN] failed to read frame. Retrying...")
                time.sleep(0.03)
                continue

            if arguments.resize_width > 0:
                height, width = frame.shape[:2]
                scale = arguments.resize_width / float(width)
                frame = cv2.resize(
                    frame,
                    (arguments.resize_width, int(height * scale)),
                )

            keypoints = extract_keypoints_from_frame(frame, pose_estimator)
            keypoints_buffer.append(keypoints)

            if len(keypoints_buffer) == window_size and (frame_index % stride == 0):
                sequence_keypoints = np.stack(keypoints_buffer, axis=0)
                features = build_features_from_keypoints(sequence_keypoints)

                try:
                    result = call_backend_behavior_analyze(backend_url, features)
                except requests.RequestException as exc:  # noqa: PERF203
                    print(f"[WARN] backend analyze failed: {exc}")
                    frame_index += 1
                    continue

                events = result.get("events", [])
                probs = result.get("scores", [])

                if arguments.show_window:
                    draw_predictions_on_frame(frame, events, probs)

                if not arguments.no_event_log and events and probs:
                    send_event_payload(
                        backend_url=backend_url,
                        event_type="PROBS",
                        events=events,
                        probs=probs,
                    )

            if arguments.show_window:
                cv2.imshow("Realtime Behavior Client", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            frame_index += 1

    finally:
        capture.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("[INFO] finished.")


if __name__ == "__main__":
    # Tensorflow 등이 섞여 있을 수 있는 환경을 조용히 만들기 위한 예전 습관을 유지.
    # Keep some legacy-style env tweaks for quiet logs if TF is present in the env.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    run_client_realtime()

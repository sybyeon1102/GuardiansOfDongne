# agent_main.py
"""
에이전트 실행 엔트리 포인트.
Entry point for RTSP → Mediapipe → Backend agent.
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from rtsp_reader import RtspReader, RtspReaderConfig
from pose_extractor import PoseExtractor, PoseExtractorConfig
from window_builder import WindowBuilder, WindowConfig
from sender_http import BehaviorSender, SenderConfig


@dataclass
class AgentConfig:
    """
    에이전트 전체 설정.
    Overall configuration for the agent.
    """

    camera_id: str
    rtsp_url: str
    backend_url: str
    window_size: int = 16
    stride: int = 4


def load_agent_config() -> AgentConfig:
    """
    .env 에서 에이전트 설정을 읽어온다.
    Load agent configuration from .env file.
    """
    load_dotenv()

    camera_id = os.getenv("AGENT_CAMERA_ID", "cam01")
    rtsp_url = os.getenv("AGENT_RTSP_URL")
    backend_url = os.getenv(
        "BACKEND_BEHAVIOR_URL",
        "http://localhost:8000/behavior/analyze_pose",
    )

    if not rtsp_url:
        raise RuntimeError("AGENT_RTSP_URL 환경 변수가 필요합니다. (RTSP 카메라 주소)")

    window_size = int(os.getenv("AGENT_WINDOW_SIZE", "16"))
    stride = int(os.getenv("AGENT_WINDOW_STRIDE", "4"))

    return AgentConfig(
        camera_id=camera_id,
        rtsp_url=rtsp_url,
        backend_url=backend_url,
        window_size=window_size,
        stride=stride,
    )


def main() -> None:
    """
    RTSP 카메라 → Mediapipe Pose → 슬라이딩 윈도우 → FastAPI POST 메인 루프.
    Main loop: RTSP camera → Mediapipe Pose → sliding window → FastAPI POST.
    """
    config = load_agent_config()

    print("==============================================")
    print("[AGENT] starting...")
    print(f"[AGENT] camera_id   = {config.camera_id}")
    print(f"[AGENT] rtsp_url    = {config.rtsp_url}")
    print(f"[AGENT] backend_url = {config.backend_url}")
    print(f"[AGENT] window_size = {config.window_size}")
    print(f"[AGENT] stride      = {config.stride}")
    print("==============================================")

    rtsp = RtspReader(RtspReaderConfig(rtsp_url=config.rtsp_url))
    pose = PoseExtractor(PoseExtractorConfig())
    window_builder = WindowBuilder(
        WindowConfig(window_size=config.window_size, stride=config.stride)
    )
    sender = BehaviorSender(
        SenderConfig(
            camera_id=config.camera_id,
            backend_url=config.backend_url,
            source_id=None,  # 실시간 카메라이므로 None / None for live camera
        )
    )

    rtsp.open()

    try:
        while True:
            ok, frame = rtsp.read_frame()
            if not ok or frame is None:
                print("[AGENT] 프레임 읽기 실패, 0.5초 대기 후 재시도.")
                time.sleep(0.5)
                continue

            # Mediapipe Pose 추론 및 keypoints 추출
            keypoints, annotated = pose.process_frame(frame)

            # (선택) 디버그용 랜드마크 이미지 확인이 필요하면 사용
            # For debugging, you can show annotated frame:
            # cv2.imshow("pose", annotated)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

            # 슬라이딩 윈도우 업데이트
            ready, window_arr, idx = window_builder.add_keypoints(keypoints)
            if not ready or window_arr is None:
                continue

            # 백엔드로 전송
            response = sender.send_window(
                window_index=idx,
                keypoints_window=window_arr,
                window_start_ts=None,
                window_end_ts=None,
            )

            if response is None:
                continue

            top_label = response.get("top_label")
            top_prob = response.get("top_prob")
            is_anomaly = response.get("is_anomaly")

            if isinstance(top_prob, (int, float)):
                top_prob_str = f"{top_prob:.3f}"
            else:
                top_prob_str = str(top_prob)

            print(
                f"[AGENT] idx={idx}, label={top_label}, "
                f"prob={top_prob_str}, is_anomaly={is_anomaly}"
            )

    finally:
        print("[AGENT] cleaning up...")
        rtsp.release()
        pose.close()
        sender.close()
        # cv2.destroyAllWindows()  # 디버그용 창을 사용했다면 주석 해제


if __name__ == "__main__":
    main()

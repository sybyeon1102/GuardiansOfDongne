# sender_http.py
"""
FastAPI /behavior/analyze_pose 엔드포인트로 윈도우 데이터를 전송하는 모듈.
Module for sending window data to FastAPI /behavior/analyze_pose endpoint.
"""

from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np


@dataclass
class SenderConfig:
    """백엔드 전송 설정 / Backend sending configuration."""
    camera_id: str
    backend_url: str
    source_id: str | None = None  # 실시간 카메라면 None / None for live camera


class BehaviorSender:
    """
    keypoints 윈도우를 백엔드로 전송하는 헬퍼.
    Helper that sends keypoints windows to backend.
    """

    def __init__(self, config: SenderConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=3.0)

    def close(self) -> None:
        """HTTP 클라이언트를 닫는다 / Close HTTP client."""
        self._client.close()

    def send_window(
        self,
        window_index: int,
        keypoints_window: np.ndarray,
        window_start_ts: float | None = None,
        window_end_ts: float | None = None,
    ) -> dict[str, Any] | None:
        """
        백엔드에 윈도우를 전송하고 JSON 응답을 반환한다.
        Send window to backend and return JSON response.

        실패 시 None 을 반환하고 에러를 출력만 한다.
        On failure, returns None and only logs error.
        """
        payload = {
            "camera_id": self.config.camera_id,
            "source_id": self.config.source_id,
            "window_index": window_index,
            "window_start_ts": window_start_ts,
            "window_end_ts": window_end_ts,
            "keypoints": keypoints_window.tolist(),
        }

        try:
            resp = self._client.post(self.config.backend_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data
        except Exception as exc:
            print(f"[SENDER] backend POST 실패: {exc}")
            return None

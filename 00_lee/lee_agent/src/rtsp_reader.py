# rtsp_reader.py
"""
RTSP 카메라에서 프레임을 읽어오는 모듈.
Module for reading frames from an RTSP camera.
"""

from dataclasses import dataclass

import cv2


@dataclass
class RtspReaderConfig:
    """RTSP 리더 설정 / RTSP reader config."""
    rtsp_url: str


class RtspReader:
    """
    RTSP 스트림을 열고 프레임을 반복적으로 읽어오는 클래스.
    Class that opens an RTSP stream and reads frames continuously.
    """

    def __init__(self, config: RtspReaderConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """RTSP 스트림을 연다 / Open RTSP stream."""
        self.cap = cv2.VideoCapture(self.config.rtsp_url)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"RTSP 카메라를 열 수 없습니다: {self.config.rtsp_url}"
            )

    def read_frame(self) -> tuple[bool, "cv2.Mat | None"]:
        """
        한 프레임을 읽는다.
        Read a single frame.

        Returns
        -------
        ok : bool
            성공 여부 / success flag.
        frame : cv2.Mat | None
            읽은 프레임 (BGR) / frame in BGR.
        """
        if self.cap is None:
            raise RuntimeError("카메라가 아직 열리지 않았습니다. call open() first.")

        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        """리소스를 정리한다 / Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

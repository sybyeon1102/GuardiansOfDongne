# window_builder.py
"""
슬라이딩 윈도우 형태로 keypoints 시퀀스를 관리하는 모듈.
Module for managing keypoints sequence as a sliding window.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

import numpy as np


@dataclass
class WindowConfig:
    """슬라이딩 윈도우 설정 / Sliding window configuration."""
    window_size: int = 16
    stride: int = 4


class WindowBuilder:
    """
    (T,33,4) 시퀀스를 슬라이딩 윈도우로 잘라주는 클래스.
    Class that builds sliding windows from (T,33,4) sequences.
    """

    def __init__(self, config: WindowConfig) -> None:
        self.config = config
        self._buffer: Deque[np.ndarray] = deque(maxlen=config.window_size)
        self._frame_counter: int = 0
        self._window_index: int = 0

    @property
    def window_index(self) -> int:
        """현재까지 생성된 윈도우 인덱스 / Current generated window index."""
        return self._window_index

    def add_keypoints(self, kpts: np.ndarray) -> tuple[bool, np.ndarray | None, int]:
        """
        새 keypoints 프레임을 추가하고, stride 에 맞을 경우 윈도우를 반환한다.
        Add new keypoints frame and return window when stride condition is met.

        Returns
        -------
        ready : bool
            윈도우가 준비되었는지 여부 / whether window is ready.
        window : np.ndarray | None
            (T,33,4) 윈도우 배열, 준비되지 않았으면 None.
        index : int
            이 윈도우의 window_index (준비되지 않았으면 이전 인덱스).
        """
        self._buffer.append(kpts)
        self._frame_counter += 1

        if len(self._buffer) < self.config.window_size:
            return False, None, self._window_index

        if self._frame_counter % self.config.stride != 0:
            return False, None, self._window_index

        window_arr = np.stack(list(self._buffer), axis=0)  # (T,33,4)
        idx = self._window_index
        self._window_index += 1

        return True, window_arr, idx

    def reset(self) -> None:
        """버퍼와 카운터를 초기화한다 / Reset buffer and counters."""
        self._buffer.clear()
        self._frame_counter = 0
        self._window_index = 0


def iter_windows(
    keypoints_stream: Iterable[np.ndarray],
    config: WindowConfig,
) -> Iterable[tuple[int, np.ndarray]]:
    """
    제너레이터 버전: keypoints 스트림에서 (index, window) 를 생성한다.
    Generator version: yields (index, window) from keypoints stream.
    """
    builder = WindowBuilder(config)

    for kpts in keypoints_stream:
        ready, window_arr, idx = builder.add_keypoints(kpts)
        if ready and window_arr is not None:
            yield idx, window_arr

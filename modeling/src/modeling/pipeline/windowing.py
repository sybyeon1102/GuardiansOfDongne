from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from modeling.pipeline.types import PoseArray


def slice_pose_windows(
    kpts: PoseArray,
    win: int,
    stride: int,
) -> PoseArray:
    """
    포즈 시퀀스 전체에서 고정 길이 슬라이딩 윈도우들을 잘라낸다.

    - 입력 포즈는 (T, 33, 4) 형태여야 한다.
    - 출력은 (N, win, 33, 4) 형태로,
      각 윈도우는 [s, s+win) 구간을 그대로 복사한 뷰/복사본이다.

    이 함수는 fps 보정이나 라벨 생성은 다루지 않고,
    순수하게 프레임 인덱스 기반 윈도우링만 담당한다.

    Slice fixed-length sliding windows from a pose sequence.

    - Input shape: (T, 33, 4)
    - Output shape: (N, win, 33, 4)
    - Each window is taken from [s, s+win) with stride between starts.

    This function does not perform any fps correction or labeling; it only
    handles index-based windowing.
    """
    k = np.asarray(kpts, dtype=np.float32)

    if k.ndim != 3 or k.shape[1:] != (33, 4):
        raise ValueError(
            f"pose sequence must have shape (T, 33, 4), got {k.shape}"
        )

    if win <= 0:
        raise ValueError(f"win must be > 0, got {win}")

    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    T = k.shape[0]
    if T < win:
        # 윈도우를 하나도 만들 수 없으면 빈 배열 반환
        return np.zeros((0, win, 33, 4), np.float32)

    starts = list(range(0, T - win + 1, stride))
    N = len(starts)

    out = np.empty((N, win, 33, 4), np.float32)
    for i, s in enumerate(starts):
        out[i] = k[s : s + win]

    return out


def iter_pose_windows(
    kpts: PoseArray,
    win: int,
    stride: int,
) -> Iterable[tuple[int, PoseArray]]:
    """
    포즈 시퀀스에서 슬라이딩 윈도우를 하나씩 생성하는 제너레이터.

    - 각 항목은 (start_index, window_kpts) 튜플이다.
    - window_kpts 의 shape 는 (win, 33, 4) 이다.
    - slice_pose_windows 와 동일한 규칙을 따르되,
      메모리를 아끼고 싶을 때 사용할 수 있다.

    Iterate over sliding windows of a pose sequence.

    - Each item is (start_index, window_kpts)
    - window_kpts has shape (win, 33, 4)
    - Follows the same rules as slice_pose_windows but yields windows
      one by one.
    """
    k = np.asarray(kpts, dtype=np.float32)

    if k.ndim != 3 or k.shape[1:] != (33, 4):
        raise ValueError(
            f"pose sequence must have shape (T, 33, 4), got {k.shape}"
        )

    if win <= 0:
        raise ValueError(f"win must be > 0, got {win}")

    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    T = k.shape[0]
    if T < win:
        return

    for s in range(0, T - win + 1, stride):
        yield s, k[s : s + win]

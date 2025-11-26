from typing import Any

import numpy as np
from numpy.typing import NDArray

from modeling.pipeline.types import PoseArray, FeatureArray


def _ffill_bfill(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    NaN 값을 앞/뒤 방향으로 보간한 뒤, 남은 NaN/inf 를 0으로 치환한다.

    - 입력: (T, D) 배열. 일부 값이 NaN 일 수 있다.
    - 처리:
      1) 앞 방향(forward)으로 마지막 관측값을 채움
      2) 뒤 방향(backward)으로도 한 번 더 채움
      3) 남은 NaN / ±inf 는 0으로 치환
    - 출력: 같은 shape 의 float32 배열.

    Fill NaNs forward and backward, then replace any remaining NaN/inf with 0.
    """
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ffill_bfill expects 2D array (T, D), got {x.shape}")

    T, D = x.shape
    out = x.copy()

    last = np.zeros(D, np.float32)
    has = np.zeros(D, bool)

    # forward pass
    for t in range(T):
        nz = ~np.isnan(out[t])
        if np.any(nz):
            last[nz] = out[t, nz]
            has |= nz
        miss = np.isnan(out[t]) & has
        if np.any(miss):
            out[t, miss] = last[miss]

    # backward pass
    last[:] = 0.0
    has[:] = False
    for t in range(T - 1, -1, -1):
        nz = ~np.isnan(out[t])
        if np.any(nz):
            last[nz] = out[t, nz]
            has |= nz
        miss = np.isnan(out[t]) & has
        if np.any(miss):
            out[t, miss] = last[miss]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def features_from_pose_seq(kpts: PoseArray) -> FeatureArray:
    """
    포즈 시퀀스에서 동일한 169차원 feature 를 생성한다.

    입력
    ----
    kpts :
        (T, 33, 4) 형태의 포즈 시퀀스.
        각 관절은 [x, y, z, visibility] 순서의 값을 가진다고 가정한다.

    출력
    ----
    feat :
        (T, 169) 형태의 float32 feature 시퀀스.

    - 169 차원 구성:
      - 66: 정규화된 좌표 xy_n (33 * 2)
      - 66: 정규화된 속도 vel (33 * 2)
      -  4: 네 지점 관절 각도 (양팔/양다리)
      - 33: visibility

    Generate 169-dim features per frame from pose sequence.
    """
    k = np.asarray(kpts, dtype=np.float32)

    if k.size == 0:
        return np.zeros((0, 169), np.float32)

    if k.ndim != 3 or k.shape[1:] != (33, 4):
        raise ValueError(
            f"pose sequence must have shape (T, 33, 4), got {k.shape}"
        )

    T = k.shape[0]

    # (T, 33, 4) -> (T, 33*2) xy  + (T, 33) visibility
    xy = k[:, :, :2].reshape(T, -1)       # (T, 66)
    vis = k[:, :, 3:4].reshape(T, -1)     # (T, 33)

    # NaN 보간 (앞/뒤 방향)
    xy = _ffill_bfill(xy).reshape(T, 33, 2)
    vis = _ffill_bfill(vis).reshape(T, 33, 1)

    # 엉덩이(23,24) / 어깨(11,12) 중심과 skeleton scale
    hip = np.mean(xy[:, [23, 24], :], axis=1)  # (T, 2)
    sh = np.mean(xy[:, [11, 12], :], axis=1)   # (T, 2)
    sc = np.linalg.norm(sh - hip, axis=1, keepdims=True)  # (T, 1)
    sc[sc < 1e-3] = 1.0

    # 위치 정규화: 엉덩이 기준, skeleton scale 로 나눔
    xy_n = (xy - hip[:, None, :]) / sc[:, None, :]  # (T, 33, 2)

    # 프레임 간 속도 (첫 프레임은 자기 자신과의 차이)
    vel = np.diff(xy_n, axis=0, prepend=xy_n[:1])   # (T, 33, 2)

    def ang(a: NDArray[np.floating],
            b: NDArray[np.floating],
            c: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        세 점 a,b,c 에 대해 ∠ABC (라디안)를 계산.

        Compute angle ABC (in radians) for three points a,b,c.
        """
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        n1[n1 == 0] = 1e-6
        n2[n2 == 0] = 1e-6
        cos = (v1 * v2).sum(-1) / (n1 * n2)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    def pick(i: int) -> NDArray[np.floating]:
        return xy_n[:, i, :]

    # 어깨-팔꿈치-손목, 엉덩이-무릎-발목 관절 각도 4개
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
        [
            xy_n.reshape(T, -1),       # 33 * 2 = 66
            vel.reshape(T, -1),        # 33 * 2 = 66
            angs,                      # 4
            vis.reshape(T, -1),        # 33
        ],
        axis=1,
    )  # (T, 169)

    return np.clip(feat.astype(np.float32), -10.0, 10.0)

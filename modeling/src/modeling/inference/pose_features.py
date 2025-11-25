import numpy as np
from numpy.typing import NDArray


def _ffill_bfill(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    NaN이 아닌 값들을 앞에서부터 전달해서 NaN을 채운다 (forward-fill).
    - arr: (T, D)
    - return: (T, D)  (복사본)
    """
    if arr.ndim != 2:
        raise ValueError(f"_ffill_bfill expects 2D array, got shape={arr.shape}")

    T, D = arr.shape
    out = arr.copy()
    last = np.zeros(D, dtype=np.float32)
    has = np.zeros(D, dtype=bool)

    for t in range(T):
        nz = ~np.isnan(out[t])
        if np.any(nz):
            last[nz] = out[t, nz]
            has |= nz

        miss = np.isnan(out[t]) & has
        if np.any(miss):
            out[t, miss] = last[miss]

    # backward-fill(bfill)은 하지 않는다 (legacy 구현 그대로).
    return out


def features_from_pose_seq(kpt_seq: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Mediapipe Pose 시퀀스 (T,33,4) -> feature 시퀀스 (T,169) 변환.

    - kpt_seq: (T, 33, 4) [x, y, z, visibility]
    - return: (T, 169) float32

    구성:
    - xy_n: hip 기준 정규화된 좌표          (T, 33*2 = 66)
    - vel: xy_n의 1-step 차분               (T, 66)
    - angs: 네 개 관절 각도                 (T, 4)
      * 양쪽 팔꿈치, 무릎
    - vis: visibility forward-fill 후       (T, 33)
    => 총 66 + 66 + 4 + 33 = 169 차원
    """
    kpt_seq = np.asarray(kpt_seq, dtype=np.float32)

    if kpt_seq.ndim != 3 or kpt_seq.shape[1:] != (33, 4):
        raise ValueError(f"kpt_seq must have shape (T,33,4), got {kpt_seq.shape}")

    T = kpt_seq.shape[0]

    # xy, visibility 분리
    xy = kpt_seq[:, :, :2]          # (T,33,2)
    vis = kpt_seq[:, :, 3:4]        # (T,33,1)

    # NaN forward-fill (joint별)
    xy_flat = xy.reshape(T, -1)     # (T,66)
    vis_flat = vis.reshape(T, -1)   # (T,33)

    xy_filled = _ffill_bfill(xy_flat).reshape(T, 33, 2)   # (T,33,2)
    vis_filled = _ffill_bfill(vis_flat).reshape(T, 33, 1) # (T,33,1)

    # hip / shoulder 중심 정규화
    hip = np.mean(xy_filled[:, [23, 24], :], axis=1)  # (T,2)
    sh = np.mean(xy_filled[:, [11, 12], :], axis=1)   # (T,2)

    sc = np.linalg.norm(sh - hip, axis=1, keepdims=True)  # (T,1)
    sc[sc < 1e-3] = 1.0

    # hip 기준, scale 로 정규화
    xy_n = (xy_filled - hip[:, None, :]) / sc[:, None, :]    # (T,33,2)

    # 한 스텝 차분 (속도)
    vel = np.diff(xy_n, axis=0, prepend=xy_n[:1])            # (T,33,2)

    # 관절 각도 계산
    def ang(
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        c: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """
        a, b, c: (..., 2)
        각도 ∈ [0, 1], π로 나눈 값
        """
        ba = a - b
        bc = c - b
        na = np.linalg.norm(ba, axis=-1, keepdims=True)
        nc = np.linalg.norm(bc, axis=-1, keepdims=True)
        na[na < 1e-6] = 1.0
        nc[nc < 1e-6] = 1.0
        cos = np.sum(ba * bc, axis=-1, keepdims=True) / (na * nc)
        cos = np.clip(cos, -1.0, 1.0)
        return (np.arccos(cos) / np.pi).astype(np.float32)   # (T,1)

    def pick(i: int) -> NDArray[np.floating]:
        # xy_n: (T,33,2)
        return xy_n[:, i, :]  # (T,2)

    # 각도들: (T,1) 네 개 → (T,4)
    angs = np.concatenate(
        [
            ang(pick(11), pick(13), pick(15)),  # 오른쪽 팔꿈치
            ang(pick(12), pick(14), pick(16)),  # 왼쪽 팔꿈치
            ang(pick(23), pick(25), pick(27)),  # 오른쪽 무릎
            ang(pick(24), pick(26), pick(28)),  # 왼쪽 무릎
        ],
        axis=1,
    )  # (T,4)

    # 최종 feature 구성
    xy_feat = xy_n.reshape(T, -1)           # (T,66)
    vel_feat = vel.reshape(T, -1)          # (T,66)
    vis_feat = vis_filled.reshape(T, -1)   # (T,33)

    feat = np.concatenate(
        [xy_feat, vel_feat, angs, vis_feat],
        axis=1,
    ).astype(np.float32)  # (T,169)

    if feat.shape != (T, 169):
        raise RuntimeError(f"feature shape mismatch, expected (T,169), got {feat.shape}")

    return feat


def features_from_buf(buf: list[NDArray[np.floating]]) -> NDArray[np.float32]:
    """
    legacy 스타일: (33,4) 프레임 리스트 -> feature 시퀀스.

    - buf: list of (33,4) 배열, 길이 T
    - return: (T,169)
    """
    if not buf:
        raise ValueError("buf must not be empty")

    kpt_seq = np.stack(buf, axis=0)  # (T,33,4)
    return features_from_pose_seq(kpt_seq)

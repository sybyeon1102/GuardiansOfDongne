from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from modeling.device import choose_torch_device
from modeling.pipeline.pose_features import features_from_pose_seq
from modeling.inference.types import LSTMAnom
from modeling.pipeline import PoseArray, FeatureArray


@dataclass
class BehaviorDetectorConfig:
    """
    이상 탐지기 설정.

    ckpt_path: Prep_v3 형식의 Model_1 .pt 체크포인트 경로.
    device   : 명시적으로 사용할 torch.device. None이면 choose_torch_device() 사용.

    Configuration for BehaviorDetector.
    """

    ckpt_path: Path | str
    device: torch.device | None = None


class BehaviorDetector:
    """
    체크포인트를 사용하는 이상 탐지기 래퍼.

    - 입력: 정규화되지 않은 포즈 시퀀스 (T, 33, 4)
      (예: Mediapipe 등으로 얻은 [x, y, z, visibility] 33개 관절)
    - 내부:
      1) features_from_pose_seq 로 (T, feat_dim) feature 생성
      2) ckpt.meta["norm_mean"], ["norm_std"] 로 feature 정규화
      3) LSTMAnom 모델로 logits 계산 후 sigmoid → p_anom
    - 출력: 이상 확률 p_anom ∈ [0, 1]

    Wrapper around anomaly detector.

    - Input: unnormalized pose sequence (T, 33, 4)
    - Internal:
      1) features_from_pose_seq → (T, feat_dim)
      2) normalize via ckpt.meta["norm_mean"/"norm_std"]
      3) LSTMAnom → sigmoid → p_anom
    - Output: anomaly probability in [0, 1]
    """

    def __init__(self, cfg: BehaviorDetectorConfig) -> None:
        ckpt_path = Path(cfg.ckpt_path)

        device = cfg.device if cfg.device is not None else choose_torch_device()
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)

        for key in ("model", "feat_dim", "num_out", "meta"):
            if key not in ckpt:
                raise ValueError(
                    f"unexpected checkpoint format: missing key '{key}' in {ckpt_path}"
                )

        feat_dim = int(ckpt["feat_dim"])
        num_out = int(ckpt["num_out"])
        if num_out != 1:
            raise ValueError(
                f"BehaviorDetector expects num_out=1 (binary), got num_out={num_out}"
            )

        model = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
        state_dict = ckpt["model"]
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"ckpt['model'] must be a state_dict dict, got {type(state_dict)!r}"
            )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        self.feat_dim = feat_dim

        meta = ckpt["meta"]
        self.meta: dict[str, Any] = meta
        self.win = int(meta.get("win", 16))

        norm_mean = np.asarray(meta["norm_mean"], dtype=np.float32)
        norm_std = np.asarray(meta["norm_std"], dtype=np.float32)

        if norm_mean.shape != (feat_dim,) or norm_std.shape != (feat_dim,):
            raise ValueError(
                "norm_mean/std shape mismatch: "
                f"feat_dim={feat_dim}, mean={norm_mean.shape}, std={norm_std.shape}"
            )

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.threshold = float(ckpt.get("threshold", 0.5))
        self._eps = 1e-6

    # -------- internal helpers ------------------------------------------------

    def _pose_to_array(self, kpt_seq: PoseArray | torch.Tensor) -> PoseArray:
        """
        입력 포즈 시퀀스를 (T, 33, 4) float32 numpy 배열로 변환하고 검증한다.

        Convert pose sequence to a float32 numpy array of shape (T, 33, 4).
        """
        if isinstance(kpt_seq, torch.Tensor):
            arr = kpt_seq.detach().cpu().numpy()
        else:
            arr = np.asarray(kpt_seq)

        if arr.ndim != 3 or arr.shape[1:] != (33, 4):
            raise ValueError(
                f"pose sequence must have shape (T, 33, 4), got {arr.shape}"
            )

        if arr.shape[0] != self.win:
            raise ValueError(
                f"window length mismatch: expected T={self.win}, got T={arr.shape[0]}"
            )

        return arr.astype(np.float32, copy=False)

    def _feature_from_pose(self, pose: PoseArray) -> FeatureArray:
        """
        포즈 시퀀스에서 raw feature 시퀀스를 생성한다.

        Build raw feature sequence (T, feat_dim) from pose sequence.
        """
        feat = features_from_pose_seq(pose)
        x = np.asarray(feat, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != self.feat_dim:
            raise ValueError(
                f"features_from_pose_seq must return shape (T,{self.feat_dim}), got {x.shape}"
            )

        if x.shape[0] != self.win:
            raise ValueError(
                f"window length mismatch after feature extraction: "
                f"expected T={self.win}, got T={x.shape[0]}"
            )

        return x

    def _preprocess(self, feat: FeatureArray) -> torch.Tensor:
        """
        raw feature 시퀀스를 정규화하고 (1, T, feat_dim) 텐서로 변환한다.

        Normalize raw features and convert to tensor (1, T, feat_dim).
        """
        x = np.asarray(feat, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.feat_dim:
            raise ValueError(
                f"feat must have shape (T,{self.feat_dim}), got {x.shape}"
            )

        x = (x - self.norm_mean[None, :]) / (self.norm_std[None, :] + self._eps)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    # -------- public API ------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, kpt_seq: PoseArray | torch.Tensor) -> float:
        """
        포즈 시퀀스 한 윈도우에 대한 이상 확률 p_anom ∈ [0,1] 을 계산한다.

        Compute anomaly probability p_anom ∈ [0, 1] for a single pose window.
        """
        pose = self._pose_to_array(kpt_seq)
        feat = self._feature_from_pose(pose)
        x_t = self._preprocess(feat)
        logits = self.model(x_t)       # (1, 1)
        p = torch.sigmoid(logits).item()
        return float(p)

    @torch.no_grad()
    def predict_is_anomaly(
        self,
        kpt_seq: PoseArray | torch.Tensor,
        threshold: float | None = None,
    ) -> bool:
        """
        threshold 기준으로 이상(True) / 정상(False) 여부를 판정한다.

        Predict whether the window is anomalous using a threshold.
        """
        thr = float(threshold) if threshold is not None else self.threshold
        p = self.predict_proba(kpt_seq)
        return p >= thr

    @torch.no_grad()
    def predict_anomaly_proba(self, kpt_seq: PoseArray | torch.Tensor) -> float:
        """
        이상 확률 p_anom ∈ [0,1] 을 계산하는 편의 메서드.

        Convenience alias for anomaly probability (server-facing).
        """
        return self.predict_proba(kpt_seq)

    def __call__(self, kpt_seq: PoseArray | torch.Tensor) -> float:
        """
        인스턴스를 함수처럼 호출하면 predict_proba 와 동일하게 동작한다.

        Calling the instance behaves like predict_proba(kpt_seq).
        """
        return self.predict_proba(kpt_seq)

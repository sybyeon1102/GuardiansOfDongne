from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from modeling.device import choose_torch_device
from modeling.inference.types import LSTMAnom
from modeling.pipeline.types import FeatureArray


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

      0) ckpt.meta["norm_mean"], ["norm_std"] 로 feature 정규화
      1) LSTMAnom 모델로 logits 계산 후 sigmoid → p_anom
    - 출력: 이상 확률 p_anom ∈ [0, 1]

    Wrapper around anomaly detector.

      0) normalize via ckpt.meta["norm_mean"/"norm_std"]
      1) LSTMAnom → sigmoid → p_anom
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
        if not isinstance(meta, dict):
            raise TypeError(
                f"ckpt['meta'] must be a dict, got {type(meta)!r}"
            )
        self.meta: dict[str, Any] = meta

        # win은 체크포인트 meta에서 반드시 가져와야 한다.
        win = meta.get("win")
        if win is None:
            raise ValueError(
                f"checkpoint meta is missing required key 'win' in {ckpt_path}"
            )
        self.win = int(win)

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

    def _preprocess(self, feat: FeatureArray | np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        raw feature 시퀀스를 정규화하고 (1, T, feat_dim) 텐서로 변환한다.

        Normalize raw features and convert to tensor (1, T, feat_dim).
        """
        if isinstance(feat, torch.Tensor):
            x = feat.detach().cpu().numpy()
        else:
            x = np.asarray(feat, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != self.feat_dim:
            raise ValueError(
                f"feat must have shape (T,{self.feat_dim}), got {x.shape}"
            )
        if x.shape[0] != self.win:
            raise ValueError(
                f"window length mismatch: expected T={self.win}, got T={x.shape[0]}"
            )

        x = (x - self.norm_mean[None, :]) / (self.norm_std[None, :] + self._eps)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    # -------- public API ------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, feat_seq: FeatureArray | np.ndarray | torch.Tensor) -> float:
        """
        feature 시퀀스 한 윈도우에 대한 이상 확률 p_anom ∈ [0,1] 을 계산한다.

        Compute anomaly probability p_anom ∈ [0, 1] for a single feature window.

        입력:
          - feat_seq: (T, feat_dim) feature sequence. (예: T=win, feat_dim=169)
        """
        x_t = self._preprocess(feat_seq)
        logits = self.model(x_t)  # (1, 1)
        p = torch.sigmoid(logits).item()
        return float(p)

    @torch.no_grad()
    def predict_is_anomaly(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
        threshold: float | None = None,
    ) -> bool:
        """
        threshold 기준으로 이상(True) / 정상(False) 여부를 판정한다.

        Predict whether the window is anomalous using a threshold.
        """
        thr = float(threshold) if threshold is not None else self.threshold
        p = self.predict_proba(feat_seq)
        return p >= thr

    @torch.no_grad()
    def predict_anomaly_proba(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
    ) -> float:
        """
        이상 확률 p_anom ∈ [0,1] 을 계산하는 편의 메서드.

        Convenience alias for anomaly probability (server-facing).
        """
        return self.predict_proba(feat_seq)

    def __call__(self, feat_seq: FeatureArray | np.ndarray | torch.Tensor) -> float:
        """
        인스턴스를 함수처럼 호출하면 predict_proba 와 동일하게 동작한다.

        Calling the instance behaves like predict_proba(feat_seq).
        """
        return self.predict_proba(feat_seq)

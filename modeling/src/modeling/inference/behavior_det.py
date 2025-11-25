from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


Tensor = torch.Tensor


class AttPool(nn.Module):
    """
    Prep_v3 Model_1/2에서 사용한 attention pooling 레이어.
    시퀀스 (B,T,D) -> context 벡터 (B,D)
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, T, D)
        a = self.w(h).squeeze(-1)                 # (B, T)
        w = torch.softmax(a, dim=1).unsqueeze(-1) # (B, T, 1)
        return (h * w).sum(1)                     # (B, D)


class LSTMAnom(nn.Module):
    """
    Prep_v3 Model_1에서 사용한 LSTM 기반 이진 탐지 모델.

    입력:  (B, T, F)   = (배치, 시퀀스 길이, feature_dim=169)
    출력:  (B, 1)      = 이상 로그릿 (sigmoid 이전)
    """

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 128,
        layers: int = 2,
        num_out: int = 1,
        bidir: bool = True,
    ) -> None:
        super().__init__()
        # Conv1d: (B, F, T) -> (B, F, T)
        self.pre = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=0.1,
        )
        d = hidden * (2 if bidir else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, F)
        return: (B, num_out) logits
        """
        # pre: Conv1d 는 (B, F, T)를 기대하므로 transpose 두 번
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)  # (B, T, F)
        h, _ = self.lstm(z)                              # (B, T, D)
        g = self.pool(h)                                 # (B, D)
        return self.head(g)                              # (B, num_out)


@dataclass
class BehaviorDetectorConfig:
    """
    Prep_v3 Model_1 기반 이상/정상 탐지기 설정.
    """
    ckpt_path: Path                  # Model_1.pt 경로 (Prep_v3 포맷)
    device: torch.device | None = None
    # 필요하면 나중에 hidden/layers 등을 override 옵션으로 추가할 수도 있음


class BehaviorDetector:
    """
    Prep_v3 Model_1.pt 체크포인트를 로드해서
    (T,169) feature 시퀀스를 받아 이상 확률 p_anom 을 돌려주는 래퍼.

    사용 예:
        cfg = BehaviorDetectorConfig(ckpt_path=Path("modeling/weights/behavior_det.pt"))
        det = BehaviorDetector(cfg)
        p = det(feat)  # feat: (T,169) numpy -> float (0~1)
    """

    def __init__(self, cfg: BehaviorDetectorConfig) -> None:
        ckpt_path = Path(cfg.ckpt_path)
        device = cfg.device or torch.device("cpu")
        self.device = device

        # --- 체크포인트 로드 ---
        ckpt = torch.load(ckpt_path, map_location=device)

        # 필수 필드 확인
        for k in ("model", "feat_dim", "num_out", "meta"):
            if k not in ckpt:
                raise ValueError(f"unexpected checkpoint format: missing key '{k}' in {ckpt_path}")

        feat_dim: int = int(ckpt["feat_dim"])
        num_out: int = int(ckpt["num_out"])
        if num_out != 1:
            raise ValueError(f"BehaviorDetector expects num_out=1, got {num_out}")

        # --- 모델 구성 + 파라미터 로드 ---
        model = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
        state_dict = ckpt["model"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"ckpt['model'] must be a state_dict dict, got {type(state_dict)!r}")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        self.feat_dim = feat_dim

        # --- meta 정보 (윈도우/정규화) ---
        meta: dict[str, Any] = ckpt["meta"]
        self.win: int = int(meta.get("win", 16))
        self.stride: int = int(meta.get("stride", 4))
        self.overlap: float = float(meta.get("overlap", 0.0))
        self.resize_w: int = int(meta.get("resize_w", 640))
        self.model_complexity: int = int(meta.get("model_complexity", 0))

        norm_mean = np.asarray(meta["norm_mean"], dtype=np.float32)
        norm_std = np.asarray(meta["norm_std"], dtype=np.float32)

        if norm_mean.shape != (feat_dim,) or norm_std.shape != (feat_dim,):
            raise ValueError(
                f"norm_mean/std shape mismatch: "
                f"feat_dim={feat_dim}, mean={norm_mean.shape}, std={norm_std.shape}"
            )

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.threshold: float = float(ckpt.get("threshold", 0.5))

        # eps to avoid division by zero
        self._eps = 1e-6

    def _preprocess(self, feat: NDArray[np.floating]) -> Tensor:
        """
        feat: (T, feat_dim) numpy → 정규화된 torch.Tensor (1, T, feat_dim)
        """
        x = np.asarray(feat, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.feat_dim:
            raise ValueError(f"feat must have shape (T,{self.feat_dim}), got {x.shape}")

        # (T,F) 에 대해 feature-wise 정규화
        x = (x - self.norm_mean[None, :]) / (self.norm_std[None, :] + self._eps)  # (T,F)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,T,F)
        return x_t

    @torch.no_grad()
    def predict_proba(self, feat: NDArray[np.floating]) -> float:
        """
        윈도우 하나에 대한 이상 확률 p_anom ∈ [0,1] 을 반환.
        """
        x_t = self._preprocess(feat)     # (1,T,F)
        logits: Tensor = self.model(x_t) # (1,1)
        p = torch.sigmoid(logits).item()
        return float(p)

    @torch.no_grad()
    def predict_is_anomaly(self, feat: NDArray[np.floating], threshold: float | None = None) -> bool:
        """
        threshold를 기준으로 True(이상) / False(정상) 판정.
        threshold를 지정하지 않으면 ckpt에 들어 있는 기본 threshold 사용.
        """
        thr = float(threshold) if threshold is not None else self.threshold
        p = self.predict_proba(feat)
        return p >= thr

    # 편의상 __call__을 predict_proba에 alias
    def __call__(self, feat: NDArray[np.floating]) -> float:  # type: ignore[override]
        return self.predict_proba(feat)

# modeling/src/modeling/inference/behavior_cls.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from .behavior_det import LSTMAnom  # Prep_v3와 동일 구조 재사용


Tensor = torch.Tensor


@dataclass
class BehaviorClassifierConfig:
    """
    Prep_v3 Model_2 기반 이상 행동 분류기 설정.
    """
    ckpt_path: Path              # Model_2.pt 경로 (Prep_v3 포맷)
    device: torch.device | None = None


class BehaviorClassifier:
    """
    Prep_v3 Model_2.pt 체크포인트를 로드해서
    (T,169) feature 시퀀스를 받아 라벨별 확률 분포를 돌려주는 래퍼.

    사용 예:
        cfg = BehaviorClassifierConfig(ckpt_path=Path("modeling/weights/behavior_cls.pt"))
        clsf = BehaviorClassifier(cfg)
        prob = clsf(feat)  # feat: (T,169) numpy -> dict[label -> prob]

        top_label, top_prob = clsf.top1(feat)
    """

    def __init__(self, cfg: BehaviorClassifierConfig) -> None:
        ckpt_path = Path(cfg.ckpt_path)
        device = cfg.device or torch.device("cpu")
        self.device = device

        # --- 체크포인트 로드 ---
        ckpt = torch.load(ckpt_path, map_location=device)

        for k in ("model", "feat_dim", "num_out", "meta"):
            if k not in ckpt:
                raise ValueError(f"unexpected checkpoint format: missing key '{k}' in {ckpt_path}")

        feat_dim: int = int(ckpt["feat_dim"])
        num_out: int = int(ckpt["num_out"])
        if num_out <= 1:
            raise ValueError(f"BehaviorClassifier expects num_out>1, got {num_out}")

        # --- 모델 구성 + 파라미터 로드 ---
        # 탐지 모델과 같은 LSTM+AttPool 구조를 사용하되, 출력 크기만 num_out
        model = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
        state_dict = ckpt["model"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"ckpt['model'] must be a state_dict dict, got {type(state_dict)!r}")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        self.feat_dim = feat_dim
        self.num_out = num_out

        # --- meta 정보 (윈도우/정규화/라벨) ---
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
        self._eps = 1e-6

        events = meta.get("events")
        if not isinstance(events, (list, tuple)) or len(events) != num_out:
            raise ValueError(
                f"meta['events'] must be a list of length num_out={num_out}, "
                f"got {type(events)!r} with len={len(events) if events is not None else 'None'}"
            )

        # 라벨 이름 목록 (softmax 인덱스와 1:1 대응)
        self.events: list[str] = [str(e) for e in events]

    def _preprocess(self, feat: NDArray[np.floating]) -> Tensor:
        """
        feat: (T, feat_dim) numpy → 정규화된 torch.Tensor (1, T, feat_dim)
        """
        x = np.asarray(feat, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.feat_dim:
            raise ValueError(f"feat must have shape (T,{self.feat_dim}), got {x.shape}")

        x = (x - self.norm_mean[None, :]) / (self.norm_std[None, :] + self._eps)  # (T,F)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,T,F)
        return x_t

    @torch.no_grad()
    def predict_proba(self, feat: NDArray[np.floating]) -> dict[str, float]:
        """
        윈도우 하나에 대한 클래스별 확률 분포를 반환.
        return: {label: prob}
        """
        x_t = self._preprocess(feat)      # (1,T,F)
        logits: Tensor = self.model(x_t)  # (1,num_out)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)  # (num_out,)

        # 라벨 이름과 확률을 매핑
        return {
            label: float(p)
            for label, p in zip(self.events, probs.tolist())
        }

    @torch.no_grad()
    def top1(self, feat: NDArray[np.floating]) -> tuple[str, float]:
        """
        가장 확률이 높은 라벨과 그 확률을 반환.
        """
        prob = self.predict_proba(feat)
        # prob이 비어 있지 않다고 가정 (num_out>1 체크를 이미 했음)
        label, p = max(prob.items(), key=lambda kv: kv[1])
        return label, p

    # 편의상 __call__을 predict_proba에 alias
    def __call__(self, feat: NDArray[np.floating]) -> dict[str, float]:  # type: ignore[override]
        return self.predict_proba(feat)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from modeling.device import choose_torch_device
from modeling.inference.types import LSTMAnom
from modeling.pipeline.types import FeatureArray


@dataclass
class BehaviorClassifierConfig:
    """
    행동 클래스 분류기 설정.

    ckpt_path: 체크포인트 경로.
    device   : 명시적으로 사용할 torch.device. None이면 choose_torch_device() 사용.

    Configuration for BehaviorClassifier.

    ckpt_path: Path to a checkpoint.
    device   : Explicit torch.device. If None, choose_torch_device() is used.
    """

    ckpt_path: Path | str
    device: torch.device | None = None


class BehaviorClassifier:
    """
    체크포인트를 사용하는 행동 클래스 분류기 래퍼.

      0) ckpt.meta["norm_mean"], ["norm_std"] 로 feature 정규화
      1) LSTMAnom (num_out = 클래스 수) 로 logits 계산
      2) softmax 로 클래스별 확률 계산
    - 출력:
      - dict[label → prob] 형태의 확률 맵
      - (events, probs) 벡터 형태 등의 편의 메서드 제공

    Wrapper around behavior classifier.
    """

    def __init__(self, cfg: BehaviorClassifierConfig) -> None:
        ckpt_path = Path(cfg.ckpt_path)

        device = cfg.device if cfg.device is not None else choose_torch_device()
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)

        for key in ("model", "feat_dim", "num_out", "events", "meta"):
            if key not in ckpt:
                raise ValueError(
                    f"unexpected checkpoint format: missing key '{key}' in {ckpt_path}"
                )

        feat_dim = int(ckpt["feat_dim"])
        num_out = int(ckpt["num_out"])

        events = ckpt["events"]
        if not isinstance(events, (list, tuple)) or not all(
            isinstance(e, str) for e in events
        ):
            raise ValueError(
                f"ckpt['events'] must be a list[str], got {type(events)!r}"
            )
        if len(events) != num_out:
            raise ValueError(
                f"len(events) must match num_out. "
                f"len(events)={len(events)}, num_out={num_out}"
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
        self.num_out = num_out
        self.events: list[str] = list(events)

        meta = ckpt["meta"]
        if not isinstance(meta, dict):
            raise TypeError(
                f"ckpt['meta'] must be a dict, got {type(meta)!r}"
            )
        self.meta: dict[str, Any] = meta

        # win 은 meta 에 반드시 있어야 한다 (하드코딩 금지)
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
        self._eps = 1e-6

    # -------- internal helpers ------------------------------------------------

    def _preprocess(
        self,
        feat: FeatureArray | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
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
    def predict_proba(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """
        feature 시퀀스 한 윈도우에 대한 클래스별 확률 맵(label → prob)을 계산한다.

        Compute per-class probability map (label → prob) for a single feature window.
        """
        x_t = self._preprocess(feat_seq)

        logits = self.model(x_t)          # (1, C)
        probs = F.softmax(logits, dim=1)  # (1, C)
        prob_vec = probs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        result: dict[str, float] = {}
        for label, p in zip(self.events, prob_vec):
            result[label] = float(p)

        return result

    @torch.no_grad()
    def predict_proba_vector(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
    ) -> tuple[list[str], NDArray[np.float32]]:
        """
        (labels, probs) 형태로 확률 벡터를 반환한다.

        Return (labels, probs_vector) for a single window.
        """
        x_t = self._preprocess(feat_seq)

        logits = self.model(x_t)          # (1, C)
        probs = F.softmax(logits, dim=1)  # (1, C)
        prob_vec = probs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        return list(self.events), prob_vec  # (C,)

    @torch.no_grad()
    def predict_top1(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
    ) -> tuple[str, float]:
        """
        가장 확률이 높은 클래스(label, prob)를 반환한다.

        Return the most probable class (label, prob).
        """
        labels, probs = self.predict_proba_vector(feat_seq)
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])

    def __call__(
        self,
        feat_seq: FeatureArray | np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """
        인스턴스를 함수처럼 호출하면 predict_proba 와 동일하게 동작한다.

        Calling the instance behaves like predict_proba(feat_seq).
        """
        return self.predict_proba(feat_seq)

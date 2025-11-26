from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from modeling.device import choose_torch_device
from modeling.pipeline.pose_features import features_from_pose_seq
from modeling.inference.types import LSTMAnom
from modeling.pipeline.types import PoseArray, FeatureArray


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

    - 입력: 정규화되지 않은 포즈 시퀀스 (T, 33, 4)
    - 내부:
      1) features_from_pose_seq 로 (T, feat_dim) feature 생성
      2) ckpt.meta["norm_mean"], ["norm_std"] 로 feature 정규화
      3) LSTMAnom (num_out = 클래스 수) 로 logits 계산
      4) softmax 로 클래스별 확률 계산
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

        meta: dict[str, Any] = ckpt["meta"]
        self.meta = meta
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
    def predict_proba(
        self,
        kpt_seq: PoseArray | torch.Tensor,
    ) -> dict[str, float]:
        """
        포즈 시퀀스 한 윈도우에 대한 클래스별 확률 맵(label → prob)을 계산한다.

        Compute per-class probability map (label → prob) for a single pose window.
        """
        pose = self._pose_to_array(kpt_seq)
        feat = self._feature_from_pose(pose)
        x_t = self._preprocess(feat)

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
        kpt_seq: PoseArray | torch.Tensor,
    ) -> tuple[list[str], NDArray[np.float32]]:
        """
        (labels, probs) 형태로 확률 벡터를 반환한다.

        Return (labels, probs_vector) for a single window.
        """
        pose = self._pose_to_array(kpt_seq)
        feat = self._feature_from_pose(pose)
        x_t = self._preprocess(feat)

        logits = self.model(x_t)          # (1, C)
        probs = F.softmax(logits, dim=1)  # (1, C)
        prob_vec = probs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        return list(self.events), prob_vec  # (C,)

    @torch.no_grad()
    def predict_top1(
        self,
        kpt_seq: PoseArray | torch.Tensor,
    ) -> tuple[str, float]:
        """
        가장 확률이 높은 클래스(label, prob)를 반환한다.

        Return the most probable class (label, prob).
        """
        labels, probs = self.predict_proba_vector(kpt_seq)
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])

    def __call__(self, kpt_seq: PoseArray | torch.Tensor) -> dict[str, float]:
        """
        인스턴스를 함수처럼 호출하면 predict_proba 와 동일하게 동작한다.

        Calling the instance behaves like predict_proba(kpt_seq).
        """
        return self.predict_proba(kpt_seq)

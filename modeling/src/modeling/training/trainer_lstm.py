"""
LSTM 기반 이상/정상 행동 멀티라벨 분류 모델의 학습 루틴을 제공합니다.

Provides training utilities for the LSTM-based multilabel classifier
used for anomaly/normal behavior detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from project_core import Err, Ok, Result


@dataclass(slots=True)
class TrainConfig:
    """
    LSTM 학습에 필요한 설정값입니다.

    Configuration for LSTM training.
    """

    data_dir: Path
    """
    전처리 결과(X.npy, Y.npy, meta.json)가 저장된 디렉터리입니다.

    Directory containing X.npy, Y.npy and meta.json produced by preprocessing.
    """

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-3
    sampler_pos_boost: float = 4.0
    validation_ratio: float = 0.2
    num_workers: int = 2
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42
    save_path: Path = Path("lstm_multilabel.pt")


@dataclass(slots=True)
class TrainMetrics:
    """
    학습 결과 요약 정보입니다.

    Summary of training results.
    """

    best_valid_loss: float
    best_thresholds: list[float]
    num_classes: int
    device: str


type TrainResult = Result[TrainMetrics, str]


class WindowDataset(Dataset):
    """
    (윈도우, 라벨) 쌍을 제공하는 Dataset 래퍼입니다.

    Dataset wrapper for (window, label) pairs.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]


class AttPool(nn.Module):
    """
    프레임 차원에 대한 단순 어텐션 풀링 레이어입니다.

    Simple attention pooling layer over the time dimension.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D)
        a = self.w(h).squeeze(-1)                  # (B, T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (h * w).sum(1)                      # (B, D)


class LstmAnomalyModel(nn.Module):
    """
    이상/정상 행동 멀티라벨 분류용 LSTM 모델입니다.

    LSTM-based multilabel classifier for anomaly/normal behavior.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_out: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1,
        )

        d = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(z)
        g = self.pool(h)
        return self.head(g)  # logits: (B, num_out)


def _select_device(preference: Literal["auto", "cpu", "cuda"]) -> str:
    """
    선호 옵션에 따라 학습에 사용할 디바이스 문자열을 선택합니다.

    Select device string ("cpu" or "cuda") from the preference.
    """
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model_lstm(config: TrainConfig) -> TrainResult:
    """
    LSTM 멀티라벨 모델을 학습하고, 최적 검증 손실 및 임계값을 반환합니다.

    Train the LSTM multilabel model and return the best validation loss
    and per-class decision thresholds.
    """
    try:
        if not config.data_dir.is_dir():
            return Err(f"data_dir not found: {config.data_dir}")

        data_dir = config.data_dir

        X_path = data_dir / "X.npy"
        Y_path = data_dir / "Y.npy"
        meta_path = data_dir / "meta.json"

        if not X_path.is_file() or not Y_path.is_file() or not meta_path.is_file():
            return Err(
                f"missing dataset files under {data_dir} "
                "(expected X.npy, Y.npy, meta.json)"
            )

        # ----------------- Load Data -----------------
        X = np.load(X_path)
        Y = np.load(Y_path)
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        events = meta.get("events", [])
        print(f"[INFO] X={X.shape} Y={Y.shape} classes={len(events)}")

        pos = Y.sum(0)
        tot = len(Y)
        for i, name in enumerate(events):
            print(f"{name:16s} pos={int(pos[i])} rate={pos[i] / tot:.6f}")

        # ----------------- Train / Valid split -----------------
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.RandomState(config.seed)
        rng.shuffle(idx)

        cut = int(n * (1.0 - config.validation_ratio))
        tr_idx = idx[:cut]
        va_idx = idx[cut:]

        print(f"[SPLIT] train={len(tr_idx)} valid={len(va_idx)}")

        any_pos = (Y[tr_idx] > 0).any(axis=1).astype(float)
        weights = 1.0 + any_pos * (config.sampler_pos_boost - 1.0)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(tr_idx),
            replacement=True,
        )

        dl_tr = DataLoader(
            WindowDataset(X[tr_idx], Y[tr_idx]),
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
        )

        dl_va = DataLoader(
            WindowDataset(X[va_idx], Y[va_idx]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        device = _select_device(config.device)
        print(f"[device] {device}")

        model = LstmAnomalyModel(
            feat_dim=X.shape[2],
            num_out=Y.shape[1],
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        pos_c = np.clip(Y.sum(0), 1.0, None)
        neg_c = Y.shape[0] - pos_c
        pos_weight = torch.tensor(
            neg_c / pos_c,
            dtype=torch.float32,
        ).clamp_(1.0, 100.0).to(device)

        print("[pos_weight]", pos_weight.cpu().numpy().round(3).tolist())

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        best_thr = [0.5] * Y.shape[1]

        start_time = time.time()

        for ep in range(1, config.epochs + 1):
            # ----------------- Train -----------------
            model.train()
            s, tot_samples = 0.0, 0

            for x, y in dl_tr:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                s += float(loss) * x.size(0)
                tot_samples += x.size(0)

            tr_loss = s / tot_samples if tot_samples > 0 else float("inf")

            # ----------------- Validate -----------------
            model.eval()
            s, tot_samples = 0.0, 0
            va_logits_list: list[np.ndarray] = []
            va_true_list: list[np.ndarray] = []

            with torch.no_grad():
                for x, y in dl_va:
                    x = x.to(device)
                    y = y.to(device)

                    logits = model(x)
                    loss = criterion(logits, y)

                    s += float(loss) * x.size(0)
                    tot_samples += x.size(0)

                    va_logits_list.append(logits.cpu().numpy())
                    va_true_list.append(y.cpu().numpy())

            va_loss = s / tot_samples if tot_samples > 0 else float("inf")
            print(f"[{ep:03d}] train {tr_loss:.4f}  valid {va_loss:.4f}")

            va_logits = np.concatenate(va_logits_list, axis=0)
            va_true = np.concatenate(va_true_list, axis=0)
            va_prob = 1.0 / (1.0 + np.exp(-va_logits))

            thr: list[float] = []
            for c in range(va_true.shape[1]):
                y_true = va_true[:, c]
                y_score = va_prob[:, c]

                if y_true.sum() == 0:
                    thr.append(0.5)
                    continue

                ps, rs, ts = precision_recall_curve(y_true, y_score)
                f1 = 2 * ps * rs / (ps + rs + 1e-12)
                best_idx = int(np.nanargmax(f1))
                t = ts[best_idx - 1] if 0 < best_idx < len(ts) + 1 else 0.5
                thr.append(float(t))

            print("[thr]", [round(t, 3) for t in thr])

            if va_loss < best_loss:
                best_loss = va_loss
                best_thr = thr

                config.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "meta": meta,
                        "thresholds": best_thr,
                    },
                    config.save_path,
                )
                print("  ↳ saved:", config.save_path)

        elapsed = time.time() - start_time
        print(f"[TIME] Elapsed time: {elapsed:.2f} seconds")

        metrics = TrainMetrics(
            best_valid_loss=best_loss,
            best_thresholds=best_thr,
            num_classes=Y.shape[1],
            device=device,
        )
        return Ok(metrics)
    except Exception as exc:
        return Err(f"failed to train LSTM model: {exc}")


def main() -> None:
    """
    CLI 진입점입니다.

    Command-line entry point to train the LSTM model.
    """
    parser = argparse.ArgumentParser(
        description="Train LSTM multilabel model on preprocessed window dataset."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sampler_pos_boost", type=float, default=4.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=Path("lstm_multilabel.pt"))

    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        sampler_pos_boost=args.sampler_pos_boost,
        validation_ratio=args.val_ratio,
        num_workers=args.num_workers,
        device=args.device,  # type: ignore[arg-type]
        seed=args.seed,
        save_path=args.save,
    )

    result = train_model_lstm(config)

    if isinstance(result, Err):
        print(f"[ERROR] {result.error}")
        raise SystemExit(1)

    metrics = result.value
    print(
        f"[DONE] best_valid_loss={metrics.best_valid_loss:.4f} "
        f"classes={metrics.num_classes} device={metrics.device}"
    )
    print("[DONE] thresholds:", [round(t, 3) for t in metrics.best_thresholds])


if __name__ == "__main__":
    main()

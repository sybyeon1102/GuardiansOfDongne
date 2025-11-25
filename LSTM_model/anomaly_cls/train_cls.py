# train_cls.py
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ---------------- Dataset ----------------
class WinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()   # (N, T, F)
        self.Y = torch.from_numpy(Y).long()    # (N,) 정수 라벨
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ---------------- Model ----------------
class AttPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1)
    def forward(self, h):
        a = self.w(h).squeeze(-1)          # (B, T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (h * w).sum(1)              # (B, D)


class LSTMAnom(nn.Module):
    def __init__(self, feat_dim, hidden=128, layers=2, num_out=8, bidir=True):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
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
        self.head = nn.Linear(d, num_out)   # 멀티클래스 logits
    def forward(self, x):
        # x: (B, T, F)
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)  # (B, T, F)
        h, _ = self.lstm(z)                              # (B, T, D)
        g = self.pool(h)                                 # (B, D)
        return self.head(g)                              # (B, C) logits


# ---------------- Train ----------------
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/elicer/data/Prep_v3/Model_2/Train")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save", default="/home/elicer/data/Prep_v3/Model_2/Train/weight/Model_2.pt")
    args = ap.parse_args()

    # --------- Load Data ----------
    X = np.load(os.path.join(args.data_dir, "X.npy"))         # (N, T, F)
    Y = np.load(os.path.join(args.data_dir, "Y.npy"))         # (N, C) or (N,)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8"))
    ev = meta["events"]                                       # 클래스 이름 리스트

    #tmp
    print(f"X dtype: {X.dtype}, Y dtype: {Y.dtype}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Y 형식 정리: (N, C) one-hot -> (N,) class index
    if Y.ndim == 2:
        Y = np.argmax(Y, axis=1).astype(np.int64)
    else:
        Y = Y.astype(np.int64)

    num_classes = len(ev)
    assert num_classes == len(np.unique(Y)), \
        f"meta['events'] 개수({num_classes})와 Y 라벨 종류({len(np.unique(Y))})가 다를 수 있음. 확인 필요."

    print(f"[INFO] X={X.shape} Y={Y.shape} classes={num_classes}")

    # 클래스 분포 출력
    counts = np.bincount(Y, minlength=num_classes)
    tot = len(Y)
    print("[INFO] class distribution:")
    for i, name in enumerate(ev):
        print(f"  {i:2d} {name:16s} count={counts[i]:5d} rate={counts[i]/tot:.6f}")

    # --------- Train / Valid split ----------
    n = X.shape[0]
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    k = int(n * 0.8)
    tr_idx, va_idx = idx[:k], idx[k:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[va_idx], Y[va_idx]

    dl_tr = DataLoader(
        WinDataset(X_tr, Y_tr),
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
    )
    dl_va = DataLoader(
        WinDataset(X_va, Y_va),
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    m = LSTMAnom(feat_dim=X.shape[2], num_out=num_classes).to(device)

    # 클래스 불균형 있으면 weight로 보정 (inverse freq)
    class_counts = np.bincount(Y, minlength=num_classes).astype(np.float32)
    class_weights = (class_counts.sum() / (class_counts + 1e-6))  # tot / count
    class_weights = class_weights / class_weights.mean()          # 평균 1로 스케일링
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("[class_weights]", class_weights.round(3).tolist())

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)

    best = np.inf
    history_tr_loss = []
    history_va_loss = []

    # --------- Epoch loop ----------
    for ep in range(1, args.epochs + 1):
        # ---- Train ----
        m.train()
        s, tot = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)   # y: (B,)
            opt.zero_grad()
            logits = m(x)                       # (B, C)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            s += float(loss) * x.size(0)
            tot += x.size(0)
        tr_loss = s / tot

        # ---- Valid ----
        m.eval()
        s, tot = 0.0, 0
        all_logits, all_true = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = m(x)
                loss = loss_fn(logits, y)
                s += float(loss) * x.size(0)
                tot += x.size(0)
                all_logits.append(logits.cpu().numpy())
                all_true.append(y.cpu().numpy())
        va_loss = s / tot

        print(f"[{ep:03d}] train {tr_loss:.4f}  valid {va_loss:.4f}")

        # ---- Validation metrics ----
        all_logits = np.concatenate(all_logits, axis=0)  # (Nv, C)
        all_true = np.concatenate(all_true, axis=0)      # (Nv,)
        probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
        pred = probs.argmax(axis=1)

        acc = accuracy_score(all_true, pred)
        macro_f1 = f1_score(all_true, pred, average="macro")
        micro_f1 = f1_score(all_true, pred, average="micro")

        print(f"  ACC={acc:.4f}  macro-F1={macro_f1:.4f}  micro-F1={micro_f1:.4f}")

        # 클래스별 리포트 (간단 버전)
        print("  [per-class F1]")
        f1_per_class = f1_score(all_true, pred, average=None, labels=np.arange(num_classes))
        for i, name in enumerate(ev):
            print(f"    {i:2d} {name:16s} F1={f1_per_class[i]:.4f}")

        # ---- loss history & plot ----
        history_tr_loss.append(tr_loss)
        history_va_loss.append(va_loss)

        epochs = np.arange(1, len(history_tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, history_tr_loss, label="train_loss")
        plt.plot(epochs, history_va_loss, label="valid_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss Curve (Epoch {ep})")
        plt.grid(True)
        plt.legend()

        save_dir = os.path.dirname(args.save)
        loss_dir = os.path.join(save_dir, "loss") if save_dir else "loss"
        os.makedirs(loss_dir, exist_ok=True)

        png_path = os.path.join(loss_dir, f"loss_curve_epoch_{ep:03d}.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f"[INFO] loss curve saved: {png_path}")

        # ---- 모델 저장 (val_loss 기준) ----
        if va_loss < best:
            best = va_loss
            ckpt = {
                "model": m.state_dict(),
                "feat_dim": X.shape[2],
                "num_out": num_classes,
                "events": ev,
                "class_weights": class_weights.tolist(),
                "epoch": ep,
                "batch": args.batch,
                "lr": args.lr,
                "meta": meta,
            }
            torch.save(ckpt, args.save)
            print("  ↳ saved:", args.save)


if __name__ == "__main__":
    run()

#train detection_v2.py
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve, roc_auc_score


class WinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


class AttPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1)
    def forward(self, h):
        a = self.w(h).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h * w).sum(1)


class LSTMAnom(nn.Module):
    def __init__(self, feat_dim, hidden=128, layers=2, num_out=1, bidir=True):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidir, dropout=0.1)
        d = hidden * (2 if bidir else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)
    def forward(self, x):
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(z)
        g = self.pool(h)
        return self.head(g)  # logits


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/elicer/data/Prep_v3/Model_1/Train/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save", default="./Model_1.pt")
    ap.add_argument("--sampler_pos_boost", type=float, default=1.0)
    args = ap.parse_args()

    # ----------------- Load Data -----------------
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    Y = np.load(os.path.join(args.data_dir, "Y.npy")).astype(np.float32)
    if Y.ndim == 1:
        Y = Y[:, None] # (N,) -> (N,1)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8"))
    print(f"[INFO] X={X.shape} Y={Y.shape}")


    pos = int(Y.sum())
    tot = len(Y)
    print(f"pos={pos} rate={pos/tot:.6f}")


    n = X.shape[0]
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    k = int(n * 0.8)
    tr_idx, va_idx = idx[:k], idx[k:]

    # ---------------- 샘플 가중치 ------------------
    tr_y = Y[tr_idx].reshape(-1)
    weights = 1.0 + tr_y * (args.sampler_pos_boost - 1.0)
    sampler = WeightedRandomSampler(
        weights=weights, 
        num_samples=len(tr_idx), 
        replacement=True
    )

    dl_tr = DataLoader(WinDataset(X[tr_idx], Y[tr_idx]), batch_size=args.batch, sampler=sampler, num_workers=2)
    dl_va = DataLoader(WinDataset(X[va_idx], Y[va_idx]), batch_size=args.batch, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = LSTMAnom(feat_dim=X.shape[2], num_out=1).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)

    # ---------------- Pos Weight 계산 --------------
    pos = float(Y.sum())        # 스칼라
    neg = float(len(Y) - pos)   # 스칼라
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    print("[pos_weight]", pos_weight.cpu().numpy().round(3).tolist())

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = np.inf
    history_tr_loss = []
    history_va_loss = []

    for ep in range(1, args.epochs + 1):
        # ---------------- Training --------------
        m.train()
        s, tot = 0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = m(x)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            s += float(loss) * x.size(0)
            tot += x.size(0)
        tr_loss = s / tot

        # ---------------- Validation --------------
        m.eval()
        s, tot = 0, 0
        va_logits, va_true = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logit = m(x)
                loss = bce(logit, y)
                s += float(loss) * x.size(0)
                tot += x.size(0)
                va_logits.append(logit.cpu().numpy())
                va_true.append(y.cpu().numpy())
        va_loss = s / tot
        print(f"[{ep:03d}] train {tr_loss:.4f}  valid {va_loss:.4f}")

        # ---------- Validation 지표 계산 ----------
        va_logits = np.concatenate(va_logits, 0)   # (N,1)
        va_true = np.concatenate(va_true, 0)       # (N,1)

        logits_1d = va_logits.reshape(-1)          # (N,)
        y_true = va_true.reshape(-1)               # (N,)

        # sigmoid(logit) → 확률
        va_prob = 1.0 / (1.0 + np.exp(-logits_1d)) # (N,)

        # 고정 threshold=0.5 기준 예측
        pred = (va_prob >= 0.5).astype(np.float32)

        # accuracy
        acc = (pred == y_true).mean()

        # precision / recall / F1@0.5
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1_fixed  = 2 * precision * recall / (precision + recall + 1e-12)

        # ROC-AUC
        roc_auc = roc_auc_score(y_true, va_prob)

        print(
            f"  ACC={acc:.4f}  P={precision:.4f}  R={recall:.4f}  "
            f"F1@0.5={f1_fixed:.4f}  ROC-AUC={roc_auc:.4f}"
        )

        # F1-curve 기반 best threshold 튜닝
        ps, rs, ts = precision_recall_curve(y_true, va_prob)
        f1_curve = 2 * ps * rs / (ps + rs + 1e-12)
        best_thr = float(ts[np.nanargmax(f1_curve)]) if len(ts) > 0 else 0.5

        print(f"[F1-thr] best_thr={best_thr:.3f}")

        
        # ---------- epoch별 실시간 loss 그래프 저장 ----------
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

        png_path = os.path.join(
            loss_dir,
            f"loss_curve_epoch_{ep:03d}.png"
        )
        plt.savefig(png_path, dpi=120)
        plt.close()

        print(f"[INFO] loss curve saved: {png_path}")

        # ------------------ 모델 저장 ------------------
        if va_loss < best:
            best = va_loss
            torch.save({
                "model": m.state_dict(),
                "feat_dim": X.shape[2],
                "num_out": 1,       
                "threshold": best_thr,
                "epoch": ep,
                "batch": args.batch,
                "lr": args.lr,
                "sampler_pos_boost": args.sampler_pos_boost,
                "meta": meta
            }, args.save)
            print("  ↳ saved:", args.save)


if __name__ == "__main__":
    run()
# save as: train_lstm.py
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve

import time


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
    start_time = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ds_lstm_all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save", default="lstm_multilabel.pt")
    ap.add_argument("--sampler_pos_boost", type=float, default=4.0)
    args = ap.parse_args()

    # ----------------- Load Data -----------------
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    Y = np.load(os.path.join(args.data_dir, "Y.npy"))
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8"))
    ev = meta["events"]

    print(f"[INFO] X={X.shape} Y={Y.shape} classes={len(ev)}")

    pos = Y.sum(0)
    tot = len(Y)
    for i, n in enumerate(ev):
        print(f"{n:16s} pos={int(pos[i])} rate={pos[i]/tot:.6f}")

    n = X.shape[0]
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    k = int(n * 0.8)
    tr_idx, va_idx = idx[:k], idx[k:]

    # 샘플 가중치
    tr_y = Y[tr_idx]
    any_pos = (tr_y.sum(1) > 0).astype(np.float32)
    weights = 1.0 + any_pos * (args.sampler_pos_boost - 1.0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(tr_idx), replacement=True)

    dl_tr = DataLoader(WinDataset(X[tr_idx], Y[tr_idx]), batch_size=args.batch, sampler=sampler, num_workers=2)
    dl_va = DataLoader(WinDataset(X[va_idx], Y[va_idx]), batch_size=args.batch, shuffle=False, num_workers=2)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[device]", device)

    m = LSTMAnom(feat_dim=X.shape[2], num_out=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)

    pos_c = np.clip(Y.sum(0), 1.0, None)
    neg_c = Y.shape[0] - pos_c
    pos_weight = torch.tensor(neg_c / pos_c, dtype=torch.float32).clamp_(1.0, 100.0).to(device)
    print("[pos_weight]", pos_weight.cpu().numpy().round(3).tolist())

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = np.inf
    best_thr = [0.5] * Y.shape[1]

    for ep in range(1, args.epochs + 1):
        # Train
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

        # Validate
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

        va_logits = np.concatenate(va_logits, 0)
        va_true = np.concatenate(va_true, 0)
        va_prob = 1 / (1 + np.exp(-va_logits))
        thr = []
        for c in range(va_true.shape[1]):
            y_true = va_true[:, c]
            y_score = va_prob[:, c]
            if y_true.sum() == 0:
                thr.append(0.5)
                continue
            ps, rs, ts = precision_recall_curve(y_true, y_score)
            f1 = 2 * ps * rs / (ps + rs + 1e-12)
            bi = int(np.nanargmax(f1))
            t = ts[bi - 1] if 0 < bi < len(ts) + 1 else 0.5
            thr.append(float(t))
        print("[thr]", [round(t, 3) for t in thr])

        if va_loss < best:
            best = va_loss
            best_thr = thr
            torch.save(
                {
                    "model": m.state_dict(),
                    "feat_dim": X.shape[2],
                    "num_out": Y.shape[1],
                    "meta": meta,
                    "thresholds": best_thr,
                },
                args.save,
            )
            print("  ↳ saved:", args.save)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[TIME] Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run()

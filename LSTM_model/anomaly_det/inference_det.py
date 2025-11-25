# inference_det.py
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


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
    ap.add_argument("--model_path", default="./Prep_v3/Train/weight/Model_1.pt")
    ap.add_argument("--test_dir", default="./Prep_v3/Test")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # 모델 로드
    print(f"[INFO] Loading model from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu")
    
    # 모델 설정 정보 출력
    print(f"[INFO] Model configuration:")
    print(f"  feat_dim: {ckpt['feat_dim']}")
    print(f"  num_out: {ckpt['num_out']}")
    print(f"  threshold: {ckpt['threshold']:.4f}")
    if 'epoch' in ckpt:
        print(f"  saved_epoch: {ckpt['epoch']}")
    if 'batch' in ckpt:
        print(f"  train_batch: {ckpt['batch']}")
    if 'lr' in ckpt:
        print(f"  train_lr: {ckpt['lr']}")
    if 'sampler_pos_boost' in ckpt:
        print(f"  sampler_pos_boost: {ckpt['sampler_pos_boost']}")
    if 'meta' in ckpt:
        for meta_key, meta_value in ckpt['meta'].items():
            if meta_key not in ("norm_mean", "norm_std"):
                print(f"  meta {meta_key}: {meta_value}")
            else:
                print(f"  meta {meta_key}: (length {len(meta_value)}), {meta_value[:3]}...")

    # 테스트 데이터 로드
    X_test = np.load(os.path.join(args.test_dir, "X.npy"))
    Y_test = np.load(os.path.join(args.test_dir, "Y.npy")).astype(np.float32)
    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]  # (N,) -> (N,1)
    
    print(f"[INFO] Test data: X={X_test.shape}, Y={Y_test.shape}")
    print(f"[INFO] Positive samples: {int(Y_test.sum())}/{len(Y_test)} ({Y_test.mean():.4f})")

    # 모델 초기화 및 가중치 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    model = LSTMAnom(feat_dim=ckpt['feat_dim'], num_out=ckpt['num_out']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    threshold = ckpt['threshold']

    # 추론 수행
    print(f"\n[INFO] Running inference...")
    X_tensor = torch.from_numpy(X_test).float().to(device)
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), args.batch_size):
            batch = X_tensor[i:i+args.batch_size]
            logits = model(batch)
            all_logits.append(logits.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)  # (N, 1)
    logits_1d = logits.reshape(-1)  # (N,)
    
    # 확률 계산 (sigmoid)
    probs = 1.0 / (1.0 + np.exp(-logits_1d))
    
    # threshold 기반 예측
    preds = (probs >= threshold).astype(np.float32)
    y_true = Y_test.reshape(-1)

    # 평가 지표 계산
    print(f"\n[INFO] Evaluation Results (threshold={threshold:.4f}):")
    
    # Accuracy
    acc = (preds == y_true).mean()
    print(f"  Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, probs)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # ROC Curve 그래프 저장
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    # 저장 디렉토리 생성
    os.makedirs(args.test_dir, exist_ok=True)
    roc_path = os.path.join(args.test_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curve saved to: {roc_path}")
    
    # Best threshold from PR curve
    ps, rs, ts = precision_recall_curve(y_true, probs)
    f1_curve = 2 * ps * rs / (ps + rs + 1e-12)
    best_thr_idx = np.nanargmax(f1_curve)
    best_thr = float(ts[best_thr_idx]) if len(ts) > 0 else 0.5
    best_f1 = f1_curve[best_thr_idx]
    
    print(f"\n[INFO] Best threshold from test data: {best_thr:.4f} (F1={best_f1:.4f})")
    
    # Best threshold로 재평가
    preds_best = (probs >= best_thr).astype(np.float32)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_best).ravel()
    precision_best = tp / (tp + fp + 1e-12)
    recall_best = tp / (tp + fn + 1e-12)
    
    print(f"  Precision: {precision_best:.4f}")
    print(f"  Recall: {recall_best:.4f}")
    print(f"  F1-Score: {best_f1:.4f}")


if __name__ == "__main__":
    run()

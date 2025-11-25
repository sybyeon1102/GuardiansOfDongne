# inference_cls.py
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class AttPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1)
    def forward(self, h):
        a = self.w(h).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h * w).sum(1)


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
        self.head = nn.Linear(d, num_out)
    def forward(self, x):
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(z)
        g = self.pool(h)
        return self.head(g)  # logits


def print_model_config(ckpt):
    """모델 설정 정보 출력"""
    print(f"[INFO] Model configuration:")
    print(f"  feat_dim: {ckpt['feat_dim']}")
    print(f"  num_out: {ckpt['num_out']}")
    if 'epoch' in ckpt:
        print(f"  saved_epoch: {ckpt['epoch']}")
    if 'batch' in ckpt:
        print(f"  train_batch: {ckpt['batch']}")
    if 'lr' in ckpt:
        print(f"  train_lr: {ckpt['lr']}")
    if 'events' in ckpt:
        print(f"  events: {ckpt['events']}")
    if 'class_weights' in ckpt:
        print(f"  class_weights: {[round(w, 3) for w in ckpt['class_weights']]}")


def evaluate_multiclass(y_true, y_pred, probs, events, test_dir):
    """Multi-class 분류 평가 및 시각화"""
    print(f"\n[INFO] Evaluation Results:")
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    
    # Per-class F1
    print("\n[INFO] Per-class F1 scores:")
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(len(events)))
    for i, name in enumerate(events):
        print(f"  {i:2d} {name:16s} F1={f1_per_class[i]:.4f}")
    
    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=events, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=events, yticklabels=events)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    # 저장
    inference_dir = os.path.join(test_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    cm_path = os.path.join(inference_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Confusion matrix saved to: {cm_path}")
    
    # Per-class probability distribution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(events):
        if i < len(axes):
            class_probs = probs[:, i]
            axes[i].hist(class_probs[y_true == i], bins=30, alpha=0.7, label=f'True {name}')
            axes[i].hist(class_probs[y_true != i], bins=30, alpha=0.7, label=f'Other classes')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{name}')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    prob_dist_path = os.path.join(inference_dir, "probability_distribution.png")
    plt.savefig(prob_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Probability distribution saved to: {prob_dist_path}")


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./Model_2_rescaling/Train/weight/Model_2.pt")
    ap.add_argument("--test_dir", default="./Model_2_rescaling/Test")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # 모델 로드
    print(f"[INFO] Loading model from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu")
    
    # 모델 설정 정보 출력
    print_model_config(ckpt)

    # 테스트 데이터 로드
    X_test = np.load(os.path.join(args.test_dir, "X.npy"))
    Y_test = np.load(os.path.join(args.test_dir, "Y.npy"))
    
    # Y 형식 정리: (N, C) one-hot -> (N,) class index
    if Y_test.ndim == 2:
        Y_test = np.argmax(Y_test, axis=1).astype(np.int64)
    else:
        Y_test = Y_test.astype(np.int64)
    
    events = ckpt.get('events', ckpt.get('meta', {}).get('events', []))
    num_classes = len(events)
    
    print(f"[INFO] Test data: X={X_test.shape}, Y={Y_test.shape}")
    print(f"[INFO] Number of classes: {num_classes}")
    
    # 클래스 분포 출력
    counts = np.bincount(Y_test, minlength=num_classes)
    tot = len(Y_test)
    print("[INFO] Test data class distribution:")
    for i, name in enumerate(events):
        print(f"  {i:2d} {name:16s} count={counts[i]:5d} rate={counts[i]/tot:.6f}")

    # 모델 초기화 및 가중치 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    model = LSTMAnom(feat_dim=ckpt['feat_dim'], num_out=ckpt['num_out']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 추론 수행
    print(f"\n[INFO] Running inference...")
    X_tensor = torch.from_numpy(X_test).float().to(device)
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), args.batch_size):
            batch = X_tensor[i:i+args.batch_size]
            logits = model(batch)
            all_logits.append(logits.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)  # (N, C)
    
    # 확률 계산 (softmax)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    # 예측 클래스
    preds = probs.argmax(axis=1)
    
    # 평가 지표 계산 및 시각화
    evaluate_multiclass(Y_test, preds, probs, events, args.test_dir)


if __name__ == "__main__":
    run()

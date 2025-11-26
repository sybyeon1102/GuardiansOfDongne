from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


class AttPool(nn.Module):
    """
    시간 축(T)에 대해 attention 기반 가중합을 수행하는 풀링 레이어.

    입력:  h (B, T, D)
    출력: (B, D)

    Attention pooling over the time dimension.
    Input:  h (B, T, D)
    Output: (B, D)
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.w(h).squeeze(-1)             # (B, T)
        weights = torch.softmax(scores, dim=1)     # (B, T)
        return (h * weights.unsqueeze(-1)).sum(1)  # (B, D)


class LSTMAnom(nn.Module):
    """
    Bi-LSTM 기반 이진 이상 탐지 모델.

    - Conv1d pre-net: (B, F, T) → (B, F, T)
    - Bi-LSTM: (B, T, F) → (B, T, hidden*dir)
    - Attention pooling: 시간 축 가중합
    - Linear head: (B, D) → (B, num_out) logits

    Bi-directional LSTM anomaly detector
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) feature 시퀀스 → logits: (B, num_out).

        x: feature sequence of shape (B, T, F) → logits of shape (B, num_out).
        """
        z = self.pre(x.transpose(1, 2)).transpose(1, 2)  # (B, T, F)
        h, _ = self.lstm(z)                              # (B, T, D)
        g = self.pool(h)                                 # (B, D)
        return self.head(g)                              # (B, num_out)

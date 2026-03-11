"""
PyTorch 제스처 분류 모델.
손 랜드마크(21개 x 3좌표 = 63차원) 입력 → 제스처 클래스 출력.
"""

import torch
import torch.nn as nn


class GestureNet(nn.Module):
    """경량 MLP 기반 제스처 분류기. 실시간 추론에 최적화."""

    GESTURE_CLASSES = [
        "open_palm",
        "fist",
        "peace",
        "point",
        "thumbs_up",
        "wave",
        "grab",
        "none",
    ]

    def __init__(self, input_dim: int = 63, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_classes(self) -> int:
        return self.net[-1].out_features

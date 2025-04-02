import torch
import torch.nn as nn
import torch.nn.functional as F


class NonContrastive(nn.Module):
    """非对比学习模块，进行像素级对齐"""

    def __init__(self, in_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, in_dim, 1)
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, in_dim, 1)
        )
        self.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(x)
        return z, p

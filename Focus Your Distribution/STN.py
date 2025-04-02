import torch
import torch.nn as nn
import torch.nn.functional as F


class ICA(nn.Module):
    """图像对齐网络ICA实现图像级对齐"""

    def __init__(self, device):
        super().__init__()
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(128 * 56 * 56, 512), nn.ReLU(),   # 假设输入图像是 224 * 224
            nn.Linear(512, 6)
        )
        self.device = device
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.to(self.device)

    def forward(self, x):
        theta = self.loc_net(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class FCA(nn.Module):
    """特征对齐网络FCA"""

    def __init__(self, in_channels, device, h, w):
        super().__init__()
        self.device = device
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear((h // 2 // 2) * (w // 2 // 2) * 128, 512), nn.ReLU(),
            nn.Linear(512, 6)
        )
        self.to(self.device)

    def forward(self, x):
        theta = self.loc_net(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

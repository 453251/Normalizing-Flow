import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        # 耦合层网络结构（与CFLOW-AD一致）
        self.net = nn.Sequential(
            nn.Linear(in_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (in_dim - in_dim // 2) * 2)  # 输出缩放因子和偏移量
        )
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)  # 可学习尺度因子

    def forward(self, x, invert=False):
        """
        输入: x [B, D]
        输出: z [B, D], log_det
        """
        x1, x2 = torch.chunk(x, 2, dim=1)  # 沿通道拆分
        st = self.net(x1)  # [B, (D - D//2)*2]
        s, t = torch.chunk(st, 2, dim=1)  # 缩放和偏移
        s = self.scale * torch.tanh(s)  # 限制缩放范围

        if not invert:
            z2 = x2 * torch.exp(s) + t  # 前向变换
            log_det = torch.sum(s, dim=1)
        else:
            z2 = (x2 - t) * torch.exp(-s)  # 逆向变换
            log_det = -torch.sum(s, dim=1)

        z = torch.cat([x1, z2], dim=1)
        return z, log_det
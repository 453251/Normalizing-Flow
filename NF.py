import torch
import torch.nn as nn
from AffineCoupling import AffineCoupling


class NormalizingFlowModel(nn.Module):
    def __init__(self, dim=256, num_layers=8):
        super().__init__()
        # 交替排列的耦合层和通道置换
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AffineCoupling(dim))
            self.layers.append(Permute(dim))  # 通道置换层

        # 基础分布为标准高斯
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def forward(self, x, invert=False):
        log_det = 0
        z = x
        for layer in reversed(self.layers) if invert else self.layers:
            if isinstance(layer, AffineCoupling):
                z, ld = layer(z, invert=invert)
            else:  # Permute层
                z, ld = layer(z, invert=invert)
            log_det += ld
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z) + log_det
        return log_prob


class Permute(nn.Module):
    """ 通道随机置换层（增强表达能力） """

    def __init__(self, dim):
        super().__init__()
        self.perm = torch.randperm(dim)
        self.inv_perm = torch.argsort(self.perm)

    def forward(self, x, invert=False):
        if not invert:
            return x[:, self.perm], 0  # 置换时log_det为0
        else:
            return x[:, self.inv_perm], 0

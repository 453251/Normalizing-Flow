import torch
import torch.nn as nn
import torch.nn.functional as F
from NF import NormalizingFlowModel


class MultiScaleFlow(nn.Module):
    def __init__(self, feat_dims={'layer1': 64, 'layer2': 128, 'layer3': 256}):
        super().__init__()
        # 为每个尺度构建独立的流模型（与CFLOW-AD层级对应）
        self.flows = nn.ModuleDict({
            'layer1': NormalizingFlowModel(dim=feat_dims['layer1'] * 9),  # 3x3块 → 通道数x9
            'layer2': NormalizingFlowModel(dim=feat_dims['layer2'] * 9),
            'layer3': NormalizingFlowModel(dim=feat_dims['layer3'] * 9)
        })
        self.patch_size = 3  # 与CFLOW-AD一致

    def forward(self, feat_pyramid):
        """
        输入: feat_pyramid = {'layer1': [B,C1,H1,W1], 'layer2': [B,C2,H2,W2], ...}
        输出: score_maps = {'layer1': [B,1,H,W], ...} (上采样到原图尺寸)
        """
        score_maps = {}
        for scale, feat in feat_pyramid.items():
            B, C, H, W = feat.shape
            # 提取局部块 (与CFLOW-AD相同)
            patches = F.unfold(feat, kernel_size=self.patch_size, stride=1)  # [B, C*9, L]
            patches = patches.permute(0, 2, 1).reshape(-1, C * 9)  # [B*L, C*9]

            # 计算负对数似然 (NLL)
            with torch.no_grad():  # 流模型在训练阶段已冻结（仅推理）
                log_prob = self.flows[scale].log_prob(patches)
            nll = -log_prob.view(B, -1)  # [B, L]

            # 转换为空间得分图
            L = (H - self.patch_size + 1) * (W - self.patch_size + 1)
            score_map = nll.view(B, H - self.patch_size + 1, W - self.patch_size + 1)

            # 上采样到原特征图尺寸（填充边缘）
            score_maps[scale] = F.interpolate(
                score_map.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        return score_maps

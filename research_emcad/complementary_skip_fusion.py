"""LGAG 后的互补跳连融合；输入、输出均为同尺寸解码特征。"""

import torch
from torch import nn
from torch.nn import functional as F


class ComplementarySkipFusion(nn.Module):
    """分解已门控 skip 的共享/新增响应，再学习小幅逐通道修正。"""

    def __init__(self, channels):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        # 每个通道只看主路、skip 与其差值；跨通道相关性由下面的投影显式估计。
        self.gate = nn.Conv2d(3 * channels, channels, 1, groups=channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        # 零初始化保证启用时的第一步前向也与原始相加严格相同。
        self.shared_gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.novel_gain = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, decoder, gated_skip):
        if (decoder.ndim != 4 or decoder.shape != gated_skip.shape
                or decoder.shape[1] != self.channels):
            raise ValueError("decoder and gated_skip must share [B,C,H,W]")
        direction = F.normalize(decoder, dim=1, eps=1e-6)
        # 只把与主路同向的 skip 分量视为共享；反向证据保留在互补分量中。
        shared = (gated_skip * direction).sum(dim=1, keepdim=True).relu() * direction
        novel = gated_skip - shared
        triplets = torch.stack((decoder, gated_skip, (decoder - gated_skip).abs()), dim=2)
        gate = torch.sigmoid(self.gate(triplets.flatten(1, 2)))
        correction = gate * (torch.tanh(self.novel_gain) * novel
                             - torch.tanh(self.shared_gain) * shared)
        return decoder + gated_skip + correction

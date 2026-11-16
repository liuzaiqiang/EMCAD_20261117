import torch
from torch import nn
from torch.nn import functional as F

class ProgressiveDeepSemantic(nn.Module):
    """Pass compact deep context through the three decoder skip stages."""

    def __init__(self, channels, width=32):
        super().__init__()
        self.compress = nn.Conv2d(channels[0], width, 1, bias=False)
        self.updates = nn.ModuleList(
            nn.Conv2d(channel, width, 1, bias=False) for channel in channels[1:3]
        )
        self.skip_projections = nn.ModuleList(
            nn.Conv2d(channel, width, 1, bias=False) for channel in channels[1:]
        )
        self.spatial_gates = nn.ModuleList(
            nn.Conv2d(width, 1, 3, padding=1) for _ in range(3)
        )
        self.strength = nn.Parameter(torch.full((3,), 0.1))

    def start(self, deep):
        return self.compress(deep)

    def filter_skip(self, semantic, skip, stage):
        semantic = F.interpolate(semantic, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        agreement = torch.tanh(semantic + self.skip_projections[stage](skip))
        gate = torch.tanh(self.spatial_gates[stage](agreement))
        return semantic, skip * (1.0 + torch.tanh(self.strength[stage]) * gate)

    def update(self, semantic, decoded, stage):
        return semantic + self.updates[stage](decoded)

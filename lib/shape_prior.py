"""Explicit shape priors for 2D EMCAD skip features.

You et al., IEEE TMI 2025, SUB/CUB equations (6)-(12), adapted from the
authors' 3D implementation to EMCAD's 2D skip maps.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _SelfUpdate(nn.Module):
    """Two class-token self-attention blocks, following the author's SUB."""

    def __init__(self, classes, spatial_size, heads=4):
        super().__init__()
        if spatial_size % heads:
            raise ValueError("SPM spatial token size must be divisible by heads")
        self.heads = heads
        self.head_size = spatial_size // heads
        self.position = nn.Parameter(torch.randn(1, heads, classes, classes))
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(spatial_size, eps=1e-6),
                "query": nn.Linear(spatial_size, spatial_size),
                "key": nn.Linear(spatial_size, spatial_size),
                "value": nn.Linear(spatial_size, spatial_size),
                "out": nn.Linear(spatial_size, spatial_size),
                "norm2": nn.LayerNorm(spatial_size, eps=1e-6),
                "mlp": nn.Sequential(nn.Linear(spatial_size, 2 * spatial_size),
                                     nn.GELU(), nn.Linear(2 * spatial_size, spatial_size)),
            }))
        self.final_norm = nn.LayerNorm(spatial_size, eps=1e-6)

    def forward(self, prior):
        batch, classes, width = prior.shape
        for block in self.blocks:
            x = block["norm1"](prior)
            q = block["query"](x).reshape(batch, classes, self.heads, self.head_size).transpose(1, 2)
            k = block["key"](x).reshape(batch, classes, self.heads, self.head_size).transpose(1, 2)
            v = block["value"](x).reshape(batch, classes, self.heads, self.head_size).transpose(1, 2)
            affinity = (q @ k.transpose(-1, -2) + self.position) / math.sqrt(self.head_size)
            updated = (affinity.softmax(-1) @ v).transpose(1, 2).reshape(batch, classes, width)
            prior = prior + block["out"](updated)
            prior = prior + block["mlp"](block["norm2"](prior))
        return self.final_norm(prior)


class _ResidualConv(nn.Module):
    """Author's 1x1/3x3 residual convolution, translated from 3D to 2D."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                      nn.BatchNorm2d(channels))

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class ShapePriorStage(nn.Module):
    """One SUB+CUB stage with residual local-prior update."""

    def __init__(self, channels, classes, grid):
        super().__init__()
        self.classes = classes
        self.grid = grid
        self.sub = _SelfUpdate(classes, grid * grid)
        self.resblock1 = _ResidualConv(channels)
        self.resblock2 = _ResidualConv(channels)
        self.local = nn.Sequential(nn.Conv2d(channels, classes, 1, bias=False),
                                   nn.BatchNorm2d(classes), nn.ReLU(inplace=True),
                                   nn.Conv2d(classes, classes, 3, padding=1))
        # The paper uses a residual CUB. A small learned layer scale keeps
        # uncommon/small foreground structures from being abruptly suppressed.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature, prior):
        batch = feature.shape[0]
        global_prior = self.sub(prior.flatten(2)).reshape(batch, self.classes, self.grid, self.grid)
        resized = F.interpolate(global_prior, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        encoded = self.resblock2(self.resblock1(feature))
        affinity = torch.matmul(encoded.flatten(2).float(), resized.flatten(2).float().transpose(1, 2))
        affinity = (affinity / math.sqrt(self.classes)).softmax(-1).to(encoded.dtype)
        context = torch.einsum("bcn,bnhw->bchw", affinity, resized)
        enhanced = encoded + context
        local = F.interpolate(self.local(enhanced), size=(self.grid, self.grid),
                              mode="bilinear", align_corners=False)
        # Retain the original EMCAD skip with an additive path.
        output = feature + self.residual_scale * (enhanced - feature)
        return output, global_prior + local


class ShapePriorPyramid(nn.Module):
    """Shared learnable class maps, updated across EMCAD's three skips."""

    def __init__(self, channels, classes, grid=16):
        super().__init__()
        self.initial_prior = nn.Parameter(torch.randn(1, classes, grid, grid))
        self.stages = nn.ModuleList(ShapePriorStage(c, classes, grid) for c in channels)

    def initial(self, batch):
        return self.initial_prior.expand(batch, -1, -1, -1)

    def forward_stage(self, stage, feature, prior):
        return self.stages[stage](feature, prior)

"""Bounded lattice warp before a depthwise convolution branch.

Each scale learns its own two-dimensional offset field, shared across channels.
The offset is bounded to one feature-grid pixel and starts at zero. Warping
once per scale avoids the large sampled tensor of per-kernel-point deformable
convolution while still adapting the spatial neighborhood to local shape.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BoundedDeformableDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size, max_offset=1.0):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("The bounded deformable branch needs an odd kernel >= 3")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.max_offset = float(max_offset)
        if self.max_offset <= 0:
            raise ValueError("max_offset must be positive")
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size, kernel_size))
        nn.init.normal_(self.weight, std=0.02)
        # One 2-D offset per spatial location, shared by channels.
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1,
                      groups=self.channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 2, 1),
        )
        # The initial forward is exactly the regular depthwise sampling grid.
        self.offset_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError("Unexpected channel count in deformable depthwise convolution")
        offsets = self.offset_predictor(x).float().permute(0, 2, 3, 1)
        offsets = offsets.tanh() * (self.max_offset * self.offset_gain.tanh())

        # Compute in float32 so grid_sample also works under mixed precision.
        y, x_coord = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32),
            torch.arange(width, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        sample_x = x_coord[None] + offsets[..., 0]
        sample_y = y[None] + offsets[..., 1]
        grid = torch.stack((
            2.0 * sample_x / max(width - 1, 1) - 1.0,
            2.0 * sample_y / max(height - 1, 1) - 1.0,
        ), dim=-1)
        warped = F.grid_sample(x.float(), grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        result = F.conv2d(warped, self.weight.float(), stride=1,
                          padding=self.kernel_size // 2, groups=channels)
        return result.to(dtype=x.dtype)

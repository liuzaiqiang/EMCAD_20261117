"""Optional directional boundary enhancement for encoder skip features.

The four fixed three-tap derivative kernels measure horizontal, vertical and
diagonal contrast. Learned per-channel orientation weights combine their
magnitudes; a zero-initialized residual gain starts from the original skip.
No segmentation labels, logits or test-time information enter this module.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryDirectionConv(nn.Module):
    """Enhance a skip feature with lightweight directional edge responses."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        kernels = torch.tensor([
            [[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]],
            [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
            [[-1., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
            [[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]],
        ]).unsqueeze(1)
        # Each group receives its own four orientations.
        self.register_buffer("direction_kernels", kernels.repeat(channels, 1, 1, 1))
        self.orientation_logits = nn.Parameter(torch.zeros(1, channels, 4, 1, 1))
        self.projection = nn.Conv2d(channels, channels, 1, groups=channels)
        self.gain = nn.Parameter(torch.zeros(1))
        self.channels = channels

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Expected NCHW skip with {} channels".format(self.channels))
        # Keep the derivative magnitudes in FP32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            responses = F.conv2d(x.float(), self.direction_kernels.float(),
                                 padding=1, groups=self.channels)
            responses = responses.abs().reshape(x.shape[0], self.channels, 4,
                                                x.shape[2], x.shape[3])
            weights = self.orientation_logits.float().softmax(dim=2)
            contrast = (responses * weights).sum(dim=2)
            gate = torch.tanh(self.projection(contrast))
            enhanced = x.float() + self.gain * gate * x.float()
        return enhanced.to(dtype=x.dtype)

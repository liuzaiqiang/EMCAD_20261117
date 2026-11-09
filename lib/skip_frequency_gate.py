"""Decoder-conditioned local frequency selection for encoder skip features.

The fixed blur and its residual are local low/high frequency proxies, not an
FFT decomposition. This experimental block adds no labels or auxiliary loss.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SkipFrequencyGate(nn.Module):
    """Independently adjust low and high skip bands using decoder guidance."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        if self.channels < 1:
            raise ValueError("channels must be positive")
        self.semantic = nn.Conv2d(
            self.channels, 2 * self.channels, kernel_size=1,
            groups=self.channels, bias=True,
        )
        self.frequency_gain = nn.Parameter(torch.zeros(self.channels, 2))
        self.alpha = nn.Parameter(torch.zeros(()))
        kernel = torch.tensor(
            [[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]],
            dtype=torch.float32,
        ) / 16.
        self.register_buffer("low_pass_kernel", kernel.view(1, 1, 3, 3))

    def forward(self, skip, decoder):
        if skip.shape != decoder.shape:
            raise ValueError("skip and decoder must have identical NCHW shapes")
        kernel = self.low_pass_kernel.to(dtype=skip.dtype)
        kernel = kernel.expand(self.channels, 1, 3, 3)
        low = F.conv2d(
            F.pad(skip, (1, 1, 1, 1), mode="replicate"),
            kernel, groups=self.channels,
        )
        high = skip - low
        # Each channel has separate low/high spatial gates. Normalized band
        # magnitudes help the gate adapt without depending on absolute scale.
        bands = torch.stack((low, high), dim=2)
        magnitudes = bands.float().abs()
        proportions = magnitudes / magnitudes.sum(dim=2, keepdim=True).clamp_min(1e-6)
        scores = self.semantic(decoder).reshape_as(bands).float()
        scores = scores + self.frequency_gain[None, :, :, None, None] * proportions
        correction = (torch.tanh(scores) * bands.float()).sum(dim=2)
        return skip + (torch.tanh(self.alpha) * correction).to(dtype=skip.dtype)

"""Decoder-conditioned local frequency selection for encoder skip features.

The fixed blur and its residual are local low/high frequency proxies, not an
FFT decomposition. This experimental block adds no labels or auxiliary loss.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SkipFrequencyGate(nn.Module):
    """Independently adjust low and high skip bands using decoder guidance."""

    _version = 2

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
        self.high_context_gain = nn.Parameter(torch.zeros(()))
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
        low_correction, high_correction = (scores.tanh() * bands.float()).unbind(dim=2)
        # A high-frequency peak with little neighborhood support may be a
        # texture spike. Adjust its existing correction, preserving its sign.
        high_fraction = proportions[:, :, 1].mean(dim=1, keepdim=True)
        neighborhood = F.avg_pool2d(
            high_fraction, 3, stride=1, padding=1, count_include_pad=False)
        support = neighborhood - high_fraction
        gain = 0.1 * self.high_context_gain.tanh()
        correction = low_correction + high_correction * (1.0 + gain * support)
        return skip + (torch.tanh(self.alpha) * correction).to(dtype=skip.dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "high_context_gain"
        if local_metadata.get("version", 1) < 2 and key not in state_dict:
            state_dict[key] = torch.zeros_like(self.high_context_gain)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

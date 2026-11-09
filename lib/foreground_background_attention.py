"""Latent foreground/background context attention for EMCAD features.

The two regions are inferred from features, without ground-truth masks or
additional losses. Their meaning must be checked empirically, not assumed.
"""

import torch
from torch import nn


class ForegroundBackgroundDecoupledAttention(nn.Module):
    """Use separate soft-region prototypes to calibrate a fused decoder map."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be positive")
        self.prior = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.foreground = nn.Conv2d(channels, channels, kernel_size=1,
                                    groups=channels, bias=True)
        self.background = nn.Conv2d(channels, channels, kernel_size=1,
                                    groups=channels, bias=True)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        if feature.ndim != 4:
            raise ValueError("feature must have NCHW shape")
        prior = torch.sigmoid(self.prior(feature).float())
        values = feature.float()
        foreground_context = (values * prior).sum(dim=(-2, -1), keepdim=True)
        foreground_context = foreground_context / prior.sum(
            dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        complement = 1.0 - prior
        background_context = (values * complement).sum(dim=(-2, -1), keepdim=True)
        background_context = background_context / complement.sum(
            dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        foreground_gate = 2.0 * torch.sigmoid(
            self.foreground(foreground_context.to(feature.dtype)).float()) - 1.0
        background_gate = 2.0 * torch.sigmoid(
            self.background(background_context.to(feature.dtype)).float()) - 1.0
        selection = prior * foreground_gate + complement * background_gate
        correction = (torch.tanh(self.alpha) * values * selection).to(feature.dtype)
        return feature + correction

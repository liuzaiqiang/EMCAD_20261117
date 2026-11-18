"""EMCAD adaptation of the Edge-Guided Feature Branch in CD-PolypNet.

Reference: Yue et al., MICCAI 2025, Section 2.3, Equations (7)-(10).
The original branch consumes SAM features. This version consumes EMCAD's d1
feature map and the original image; it does not reproduce the full SAM model.
"""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _MultiScaleChannelAttention(nn.Module):
    """Local and global attention used by the two stages of iAFF."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(1, channels // 4)
        # No BN on the pooled 1x1 path: medical validation often uses batch=1.
        self.local = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.local(x) + self.global_path(x))


class _IterativeAttentionFusion(nn.Module):
    """Two-stage iAFF from Dai et al., WACV 2021, Equation (5)."""

    def __init__(self, channels):
        super().__init__()
        self.first = _MultiScaleChannelAttention(channels)
        self.second = _MultiScaleChannelAttention(channels)

    def forward(self, edge, semantic):
        weight = self.first(edge + semantic)
        initial = weight * edge + (1.0 - weight) * semantic
        weight = self.second(initial)
        return weight * edge + (1.0 - weight) * semantic


class EdgeGuidedFeatureBranch(nn.Module):
    """Canny prior + fixed Gaussian high-pass + learned gate + iAFF."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        coords = torch.arange(7, dtype=torch.float32) - 3.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * 3.0 ** 2))
        kernel = kernel / kernel.sum()
        self.lowpass = nn.Conv2d(channels, channels, 7, padding=3,
                                 groups=channels, bias=False)
        with torch.no_grad():
            self.lowpass.weight.copy_(kernel.reshape(1, 1, 7, 7).repeat(channels, 1, 1, 1))
        self.lowpass.weight.requires_grad_(False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels + 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels + 1, 1, 1), nn.Sigmoid(),
        )
        self.semantic = nn.Conv2d(channels, channels, 3, padding=1)
        self.fusion = _IterativeAttentionFusion(channels)

    @staticmethod
    def _canny_prior(image, size, dtype):
        # Canny is a fixed, non-differentiable image operation, not a trainable
        # layer. Image intensities vary across CT/MRI/endoscopy datasets, so
        # convert each input to uint8 by its own intensity range first.
        with torch.no_grad():
            images = image.detach().float().cpu().numpy()
            edges = []
            for sample in images:
                gray = sample.mean(axis=0)
                low, high = float(gray.min()), float(gray.max())
                if high > low:
                    gray = (gray - low) * (255.0 / (high - low))
                else:
                    gray = np.zeros_like(gray)
                edge = cv2.Canny(np.ascontiguousarray(gray.astype(np.uint8)), 10, 100)
                edges.append(edge)
            prior = torch.from_numpy(np.stack(edges)[:, None]).to(
                device=image.device, dtype=dtype).div_(255.0)
        return F.interpolate(prior, size=size, mode="nearest")

    def forward(self, features, image):
        if features.ndim != 4 or image.ndim != 4 or features.shape[0] != image.shape[0]:
            raise ValueError("EFB expects matching BCHW image and feature batches")
        prior = self._canny_prior(image, features.shape[-2:], features.dtype)
        high_frequency = features - self.lowpass(features)
        attention = self.gate(torch.cat((high_frequency, prior), dim=1))
        edge = (1.0 + attention) * high_frequency
        semantic = self.semantic(features)
        return self.fusion(edge, semantic)

"""Coarse-context guided local-detail refinement for the final EMCAD feature.

This is an experimental, class-agnostic residual branch. It does not alter
labels, losses, or the four segmentation heads.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ContextDetailRefinement(nn.Module):
    """Use d2 context to gate a bounded high-frequency correction of d1."""

    def __init__(self, fine_channels, coarse_channels):
        super().__init__()
        self.context = nn.Conv2d(coarse_channels, fine_channels, 1, bias=False)
        self.detail_dw = nn.Conv2d(
            fine_channels, fine_channels, 3, padding=0,
            groups=fine_channels, bias=False,
        )
        self.detail_pw = nn.Conv2d(fine_channels, fine_channels, 1, bias=False)
        self.gate = nn.Conv2d(2, 1, 1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, fine, coarse):
        context = F.interpolate(
            self.context(coarse), size=fine.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        # Local high-pass evidence; replicate padding avoids an artificial
        # image-frame edge. The gate combines detail strength and context.
        padded = F.pad(fine, (1, 1, 1, 1), mode="replicate")
        detail = fine - F.avg_pool2d(padded, kernel_size=3, stride=1)
        evidence = torch.cat(
            (torch.tanh(detail).abs().mean(1, keepdim=True),
             (torch.tanh(fine) * torch.tanh(context)).mean(1, keepdim=True)), dim=1,
        )
        gate = torch.sigmoid(self.gate(evidence))
        correction = torch.tanh(self.detail_pw(self.detail_dw(
            F.pad(detail, (1, 1, 1, 1), mode="replicate"))))
        return fine + torch.tanh(self.alpha) * gate * correction

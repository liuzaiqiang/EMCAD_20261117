"""Lightweight axial context correction on EMCAD's final decoder feature."""

from torch import nn
from torch.nn import functional as F


class AxialContextRefinement(nn.Module):
    """Capture long horizontal/vertical structure at 1/4 resolution.

    The zero-initialized projection keeps the initial p1 logits identical to EMCAD.
    This branch uses decoder context only; the raw-image detail branch stays separate.
    """

    def __init__(self, decoder_channels, num_classes, kernel_size=7):
        super().__init__()
        if min(decoder_channels, num_classes) < 1 or kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("invalid channels or axial kernel size")
        padding = kernel_size // 2
        self.horizontal = nn.Conv2d(decoder_channels, decoder_channels, (1, kernel_size),
                                    padding=(0, padding), groups=decoder_channels, bias=False)
        self.vertical = nn.Conv2d(decoder_channels, decoder_channels, (kernel_size, 1),
                                  padding=(padding, 0), groups=decoder_channels, bias=False)
        self.norm = nn.BatchNorm2d(decoder_channels)
        self.activation = nn.ReLU(inplace=True)
        self.to_logits = nn.Conv2d(decoder_channels, num_classes, 1)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, d1, output_size):
        if d1.ndim != 4:
            raise ValueError("d1 must be [B,C,H/4,W/4]")
        context = self.horizontal(d1) + self.vertical(d1)
        correction = self.to_logits(self.activation(self.norm(context)))
        return F.interpolate(correction, size=output_size, mode="bilinear", align_corners=False)

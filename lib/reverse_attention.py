"""PraNet-style reverse attention adapted to an EMCAD d2-to-d1 transition.

Reference: Fan et al., PraNet, MICCAI 2020, Sec. 2.2 and the authors'
PraNet_Res2Net.py reverse-attention branch 2. This is a single-stage adapter,
not the complete PraNet with its Res2Net, PPD, and three RA stages.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _BasicConv2d(nn.Module):
    """The author's BasicConv2d: convolution then batch normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ReverseAttentionRefinement(nn.Module):
    """Use a d2 coarse logit map to reverse-gate d1 and predict a logit residual."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive")
        self.num_classes = num_classes
        # Same 1x1 -> 3x3 -> 3x3 -> 3x3, width-64 RA2 sequence as PraNet.
        # Only the input channel count and output class count are adapted.
        self.ra_conv1 = _BasicConv2d(in_channels, 64, 1)
        self.ra_conv2 = _BasicConv2d(64, 64, 3, padding=1)
        self.ra_conv3 = _BasicConv2d(64, 64, 3, padding=1)
        self.ra_conv4 = _BasicConv2d(64, num_classes, 3, padding=1)

    def forward(self, d1, coarse_logits):
        if d1.ndim != 4 or coarse_logits.ndim != 4:
            raise ValueError("reverse attention expects BCHW feature and logits")
        if d1.shape[0] != coarse_logits.shape[0] or coarse_logits.shape[1] != self.num_classes:
            raise ValueError("reverse attention batch or class count mismatch")
        coarse = F.interpolate(coarse_logits, size=d1.shape[-2:], mode="bilinear")
        if self.num_classes == 1:
            reverse = 1.0 - torch.sigmoid(coarse)
        else:
            # PraNet is binary. In exclusive multi-class segmentation, class 0
            # is background; its probability is the complement of foreground.
            reverse = torch.softmax(coarse, dim=1)[:, :1]
        x = d1 * reverse
        x = self.ra_conv1(x)
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        residual = self.ra_conv4(x)
        return coarse + residual

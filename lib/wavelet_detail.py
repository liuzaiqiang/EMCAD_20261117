"""Directional Haar detail compensation for EMCAD decoder features.

Haar analysis/synthesis follows an orthonormal, non-overlapping 2x2 convention.
This is a proposed feature correction, not a reproduction of MWCNN or a claim
that high-frequency feature coefficients necessarily represent true boundaries.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def haar_analysis(x):
    """Return LL, horizontal-edge, vertical-edge, diagonal bands and original HW.

    Horizontal means high-pass along rows (y), low-pass along columns (x).
    Vertical means low-pass along rows, high-pass along columns. Odd dimensions
    are replicate-padded on bottom/right only, and removed after synthesis.
    """
    height, width = x.shape[-2:]
    x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    horizontal = (a + b - c - d) * 0.5
    vertical = (a - b + c - d) * 0.5
    diagonal = (a - b - c + d) * 0.5
    return ll, torch.stack((horizontal, vertical, diagonal), dim=2), (height, width)


def haar_synthesis(ll, details, output_size):
    """Invert :func:`haar_analysis`; LL=0 reconstructs only high-frequency detail."""
    horizontal, vertical, diagonal = details.unbind(dim=2)
    a = (ll + horizontal + vertical + diagonal) * 0.5
    b = (ll + horizontal - vertical - diagonal) * 0.5
    c = (ll - horizontal + vertical - diagonal) * 0.5
    d = (ll - horizontal - vertical + diagonal) * 0.5
    # Interleave columns, then rows. No interpolation of decimated coefficients.
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    result = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return result[..., :output_size[0], :output_size[1]]


class DirectionalWaveletDetailCompensation(nn.Module):
    """Shrink/gate raw skip details and add them to the aligned EUCB output.

    Thresholds are relative to per-sample, per-channel, per-direction mean
    absolute coefficients. This is an adaptive feature scale, not an estimate
    of physical image noise. Independent sigmoid gates allow all bands to close.
    A zero-initialized, tanh-bounded residual scale preserves the original
    feature at initialization while receiving a gradient on the first step.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        hidden = max(8, self.channels // 8)
        self.log_threshold = nn.Parameter(
            torch.full((1, self.channels, 3, 1, 1), math.log(math.expm1(0.1))))
        self.context_gate = nn.Sequential(
            nn.Conv2d(2 * self.channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )
        # No bias: a zero detail signal must produce an exactly zero correction.
        self.detail_projection = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, upsampled, raw_skip):
        if upsampled.shape != raw_skip.shape:
            raise ValueError("Wavelet compensation needs aligned EUCB/skip shapes: {} vs {}".format(
                tuple(upsampled.shape), tuple(raw_skip.shape)))
        if raw_skip.shape[1] != self.channels:
            raise ValueError("Unexpected wavelet skip channel count")

        ll, details, size = haar_analysis(raw_skip)
        # Detaching the scale prevents optimizing the shrinkage reference itself.
        magnitude = details.detach().abs().mean(dim=(-2, -1), keepdim=True)
        threshold = F.softplus(self.log_threshold) * magnitude
        retained = details.sign() * F.relu(details.abs() - threshold)

        # LL/2 equals the 2x2 average under the orthonormal Haar convention.
        padded_up = F.pad(upsampled, (0, size[1] % 2, 0, size[0] % 2), mode="replicate")
        up_context = F.avg_pool2d(padded_up, kernel_size=2, stride=2)
        context = torch.cat((up_context, ll * 0.5), dim=1)
        gates = torch.sigmoid(self.context_gate(context)).unsqueeze(1)
        correction = haar_synthesis(torch.zeros_like(ll), retained * gates, size)
        correction = self.detail_projection(correction)
        return upsampled + torch.tanh(self.residual_scale) * correction

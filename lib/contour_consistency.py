"""Single-pass multi-scale feature-contour feedback (not a boundary loss).

Reused from local branch contentAware_and_dcdlkyzx, commit c4049ec.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleContourConsistency(nn.Module):
    """Compare d3/d2/d1 on d2's grid, then feed disagreements back.

    Contours are class-agnostic feature gradients, not predicted/GT boundaries.
    Fixed channel reduction prevents a learned contour projector collapsing to
    zero. Zero residual strengths give an identity start; they remain trainable.
    """

    def __init__(self, channels, hidden=8):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Expected d3/d2/d1 channels")
        self.feedback = nn.ModuleList([
            nn.Sequential(nn.Conv2d(4, hidden, 1, bias=False), nn.SiLU(),
                          nn.Conv2d(hidden, c, 1, bias=False))
            for c in channels
        ])
        self.strength = nn.Parameter(torch.zeros(3))
        sobel = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                              [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]) / 8
        self.register_buffer("sobel", sobel.unsqueeze(1))

    def contour(self, feature, size):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return self._contour_float(feature, size)

    def _contour_float(self, feature, size):
        # Compute in float32 even under AMP; normalize channel amplitudes before
        # reducing, so channels with large activations do not dominate edges.
        x = feature.float()
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False,
                              antialias=True)
        x = x / x.square().mean((-2, -1), keepdim=True).add(1e-6).sqrt()
        b, c, h, w = x.shape
        gradient = F.conv2d(F.pad(x.reshape(b*c, 1, h, w), (1, 1, 1, 1),
                                 mode="replicate"), self.sobel.float())
        magnitude = (gradient.square().sum(1, keepdim=True) + 1e-8).sqrt() - 1e-4
        edge = magnitude.reshape(b, c, h, w).mean(1, keepdim=True)
        return edge / (edge + edge.mean((-2, -1), keepdim=True) + 1e-6)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("Expected [d3, d2, d1]")
        edges = [self.contour(x, features[1].shape[-2:]) for x in features]
        outputs = []
        for i, (x, branch) in enumerate(zip(features, self.feedback)):
            own = edges[i]
            peer = (edges[(i+1) % 3] + edges[(i+2) % 3]) * 0.5
            delta = peer - own
            comparison = torch.cat((own, peer, delta, delta.abs()), dim=1)
            comparison = F.interpolate(comparison, size=x.shape[-2:],
                                       mode="bilinear", align_corners=False)
            modulation = torch.tanh(branch(comparison.to(x.dtype)))
            outputs.append(x + torch.tanh(self.strength[i]) * x * modulation)
        return outputs

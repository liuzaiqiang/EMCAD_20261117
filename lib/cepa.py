"""EMCAD skip refinement using directional support and two-stream edge agreement.

The CEPA switch retains its historical CLI name. This implementation borrows
the connectivity/edge idea, but is not a reproduction of Wu et al.'s CEPA.
The authors' complete equations and code have not been independently checked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPA(nn.Module):
    """Refine an LGAG-selected skip only at supported two-stream boundaries."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError('CEPA channels must be positive')
        hidden = max(8, channels // 4)
        self.skip_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.decoder_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2 + 1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer('sobel_x', torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3),
            persistent=False)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3),
            persistent=False)

    @staticmethod
    def _directional_support(features):
        """One-sided continuity across each opposite-neighbor pair."""
        z = F.normalize(features.float(), p=2, dim=1, eps=1e-6)
        height, width = z.shape[-2:]
        padded = F.pad(z, (1, 1, 1, 1), mode='replicate')
        scores = []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            def similarity(sign):
                neighbor = padded[:, :, 1 + sign * dy:1 + sign * dy + height,
                                  1 + sign * dx:1 + sign * dx + width]
                return ((z * neighbor).sum(dim=1, keepdim=True) + 1.0) * 0.5
            plus, minus = similarity(1), similarity(-1)
            scores.append((plus - minus).abs() * torch.maximum(plus, minus))
        return torch.stack(scores, dim=0).amax(dim=0)

    def _gradient(self, features):
        gray = features.float().mean(dim=1, keepdim=True)
        gray = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        return (F.conv2d(gray, self.sobel_x.float()),
                F.conv2d(gray, self.sobel_y.float()))

    def _edge_agreement(self, skip, decoder):
        sx, sy = self._gradient(skip)
        dx, dy = self._gradient(decoder)
        sm = torch.sqrt(sx.square() + sy.square() + 1e-6)
        dm = torch.sqrt(dx.square() + dy.square() + 1e-6)
        sn = sm / sm.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        dn = dm / dm.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        alignment = (sx * dx + sy * dy).abs() / (sm * dm + 1e-6)
        return torch.minimum(sn, dn) * alignment.clamp(max=1.0)

    def forward(self, skip, decoder):
        if skip.shape != decoder.shape:
            raise ValueError('CEPA skip and decoder must have the same [B,C,H,W] shape')
        skip_features = self.skip_projection(skip)
        decoder_features = self.decoder_projection(decoder)
        support = self._directional_support(skip_features)
        agreement = self._edge_agreement(skip, decoder)
        boundary = (support * agreement).to(skip_features.dtype)
        gate = torch.sigmoid(self.fuse(torch.cat(
            (skip_features, decoder_features, boundary), dim=1)))
        return skip + torch.tanh(self.residual_scale) * skip * gate * boundary

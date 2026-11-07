"""Optional boundary-guided residual refinement of EMCAD's d1 feature."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBoundaryRefinement(nn.Module):
    """Refine the highest-resolution decoder feature before its segmentation head.

    The Sobel response is a feature-gradient proxy, not a ground-truth boundary.
    A zero-initialized pointwise projection makes the module an exact identity at
    construction; later training can learn bounded, edge-modulated corrections.
    """

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError('channels must be positive')
        self.channels = channels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        self.edge_gate = nn.Conv2d(1, 1, kernel_size=1)
        self.residual_dw = nn.Conv2d(channels, channels, kernel_size=3,
                                     padding=1, groups=channels, bias=False)
        self.activation = nn.SiLU(inplace=True)
        self.residual_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.residual_pw.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected a BCHW d1 feature with {} channels'.format(self.channels))
        pad_mode = 'reflect' if min(x.shape[-2:]) > 1 else 'replicate'
        padded = F.pad(x, (1, 1, 1, 1), mode=pad_mode)
        kernel_x = self.sobel_x.to(dtype=x.dtype).expand(self.channels, 1, 3, 3)
        kernel_y = self.sobel_y.to(dtype=x.dtype).expand(self.channels, 1, 3, 3)
        gradient_x = F.conv2d(padded, kernel_x, groups=self.channels)
        gradient_y = F.conv2d(padded, kernel_y, groups=self.channels)
        edge = (gradient_x.abs() + gradient_y.abs()).mean(dim=1, keepdim=True)
        edge = edge / (edge + edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
        gate = edge * torch.sigmoid(self.edge_gate(edge))
        residual = self.residual_pw(self.activation(self.residual_dw(x)))
        return x + torch.tanh(self.scale) * gate * residual

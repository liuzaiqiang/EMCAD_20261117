"""Optional structure/texture decomposition of an EMCAD gated skip.

Low-pass and residual high-pass features are conditioned separately on the
decoder feature. This is an experimental combination of established signal
decomposition and gating ideas, not a claim of a new underlying operator.
"""

import torch
from torch import nn
from torch.nn import functional as F


class StructureTextureSkipFusion(nn.Module):
    """Refine a gated encoder skip before it is added to the decoder stream."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.structure_filter = nn.Conv2d(channels, channels, 5, padding=2,
                                          groups=channels, bias=False)
        self.texture_filter = nn.Conv2d(channels, channels, 3, padding=1,
                                        groups=channels, bias=False)
        self.structure_gate = nn.Conv2d(channels, channels, 1, groups=channels)
        self.texture_gate = nn.Conv2d(channels, channels, 1, groups=channels)
        self.structure_gain = nn.Parameter(torch.zeros(1))
        self.texture_gain = nn.Parameter(torch.zeros(1))

    def forward(self, decoder, gated_skip):
        if decoder.shape != gated_skip.shape or decoder.ndim != 4:
            raise ValueError("Decoder and gated skip must have matching NCHW shapes")
        if decoder.shape[1] != self.channels:
            raise ValueError("Expected {} channels".format(self.channels))
        # Keep subtraction accurate when the surrounding network uses AMP.
        with torch.autocast(device_type=decoder.device.type, enabled=False):
            guide = decoder.float()
            skip = gated_skip.float()
            structure = F.avg_pool2d(skip, 5, stride=1, padding=2,
                                     count_include_pad=False)
            texture = skip - structure
            structure_weight = torch.sigmoid(self.structure_gate(guide + structure))
            texture_weight = torch.sigmoid(self.texture_gate(guide + texture))
            structure_delta = structure_weight * self.structure_filter(structure)
            texture_delta = texture_weight * self.texture_filter(texture)
            refined = (skip + torch.tanh(self.structure_gain) * structure_delta
                       + torch.tanh(self.texture_gain) * texture_delta)
        return refined.to(dtype=gated_skip.dtype)

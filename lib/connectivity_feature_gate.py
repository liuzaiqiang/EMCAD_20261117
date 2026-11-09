"""Optional local-neighbor continuity hint for a gated encoder skip.

This module uses feature similarity between adjacent pixels to mix an already
LGAG-gated skip. It does not compute connected components or enforce topology.
The mechanism is an experimental, learned edge-aware local aggregation.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _four_neighbors(x):
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return (
        padded[:, :, :-2, 1:-1],
        padded[:, :, 2:, 1:-1],
        padded[:, :, 1:-1, :-2],
        padded[:, :, 1:-1, 2:],
    )


class ConnectivityFeatureGate(nn.Module):
    """Condition local skip propagation on decoder and skip similarity."""

    def __init__(self, channels, guide_channels=8):
        super().__init__()
        if channels < 1 or guide_channels < 1:
            raise ValueError("channels and guide_channels must be positive")
        self.channels = channels
        self.decoder_guide = nn.Conv2d(channels, guide_channels, 1, bias=False)
        self.skip_guide = nn.Conv2d(channels, guide_channels, 1, bias=False)
        self.confidence = nn.Conv2d(2, 1, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, decoder, gated_skip):
        if decoder.ndim != 4 or decoder.shape != gated_skip.shape:
            raise ValueError("Decoder and gated skip must have matching NCHW shapes")
        if decoder.shape[1] != self.channels:
            raise ValueError("Expected {} channels".format(self.channels))

        # FP32 affinity computation remains stable during mixed precision.
        with torch.autocast(device_type=decoder.device.type, enabled=False):
            decoder = decoder.float()
            skip = gated_skip.float()
            guide = torch.tanh(self.decoder_guide(decoder) + self.skip_guide(skip))
            temperature = self.log_temperature.exp().clamp(min=0.1, max=10.0)
            guide_neighbors = _four_neighbors(guide)
            scores = [torch.zeros_like(guide[:, :1])]
            scores.extend(-((guide - other).square().mean(dim=1, keepdim=True))
                          / temperature for other in guide_neighbors)
            affinity = torch.cat(scores, dim=1).softmax(dim=1)

            skip_neighbors = _four_neighbors(skip)
            context = affinity[:, :1] * skip
            for index, neighbor in enumerate(skip_neighbors, start=1):
                context = context + affinity[:, index:index + 1] * neighbor
            delta = context - skip
            evidence = torch.cat(((decoder - skip).abs().mean(dim=1, keepdim=True),
                                  delta.abs().mean(dim=1, keepdim=True)), dim=1)
            confidence = torch.sigmoid(self.confidence(evidence))
            refined = skip + torch.tanh(self.gain) * confidence * delta
        return refined.to(dtype=gated_skip.dtype)

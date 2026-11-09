"""Optional local-contrast scale selection on EMCAD decoder features.

This is a lightweight experimental module, not a proven performance gain.
Three- and five-pixel local contrasts are processed independently and mixed
with a per-location scale gate. A zero-initialized residual gain preserves the
incoming feature at initialization. No masks or predictions guide the gate.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SmallTargetScaleEnhancement(nn.Module):
    """Select short-range detail at each location while retaining the input."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.detail3 = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.detail5 = nn.Conv2d(channels, channels, 5, padding=2,
                                 groups=channels, bias=False)
        self.scale_gate = nn.Conv2d(2, 2, 1)
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Expected NCHW feature with {} channels".format(self.channels))
        # FP32 local contrast avoids overflow or cancellation under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature = x.float()
            contrast3 = feature - F.avg_pool2d(feature, 3, stride=1, padding=1,
                                               count_include_pad=False)
            contrast5 = feature - F.avg_pool2d(feature, 5, stride=1, padding=2,
                                               count_include_pad=False)
            evidence = torch.cat((contrast3.abs().mean(dim=1, keepdim=True),
                                  contrast5.abs().mean(dim=1, keepdim=True)), dim=1)
            scales = self.scale_gate(evidence).softmax(dim=1)
            detail = (scales[:, :1] * self.detail3(contrast3)
                      + scales[:, 1:] * self.detail5(contrast5))
            enhanced = feature + self.gain * detail
        return enhanced.to(dtype=x.dtype)

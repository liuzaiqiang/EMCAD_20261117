"""CaraNet channel-wise feature pyramid (CFP) adapted to EMCAD channels.

Reference: AngeLouCN/CaraNet/lib/context_module.py, CFPModule(nIn=32,d=8).
The author code uses dilation rates 1, 3, 5, 9 when d=8.
"""

import torch
from torch import nn


class _BNPReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.activation = nn.PReLU(channels)

    def forward(self, x):
        return self.activation(self.bn(x))


class _Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, dilation=1,
                 groups=1, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel,
                              padding=(kernel // 2) * dilation,
                              dilation=dilation, groups=groups, bias=False)
        self.activation = _BNPReLU(output_channels) if activate else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))


class CFPModule(nn.Module):
    """Four dilated channel pyramids with cumulative fusion and residual output."""

    def __init__(self, channels=32, dilation=8):
        super().__init__()
        if channels % 16:
            raise ValueError("CFP channels must be divisible by 16")
        self.pre = _BNPReLU(channels)
        self.reduce = _Conv(channels, channels // 4, 3)
        rates = (1, int(dilation / 4 + 1), int(dilation / 2 + 1), dilation + 1)
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(nn.Sequential(
                _Conv(channels // 4, channels // 16, 3, rate, channels // 16),
                _Conv(channels // 16, channels // 16, 3, rate, channels // 16),
                _Conv(channels // 16, channels // 8, 3, rate, channels // 16),
            ))
        self.post = _BNPReLU(channels)
        self.project = _Conv(channels, channels, 1, activate=False)

    def forward(self, x):
        reduced = self.reduce(self.pre(x))
        outputs = []
        for branch in self.branches:
            first = branch[0](reduced)
            second = branch[1](first)
            third = branch[2](second)
            outputs.append(torch.cat((first, second, third), dim=1))
        cumulative = []
        running = outputs[0]
        for output in outputs:
            if not cumulative:
                running = output
            else:
                running = running + output
            cumulative.append(running)
        return x + self.project(self.post(torch.cat(cumulative, dim=1)))


class CFPBridge(nn.Module):
    """Project EMCAD's encoder channels to CaraNet's 32 and back."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = _Conv(channels, 32, 3)
        self.cfp = CFPModule(32, dilation=8)
        self.expand = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, x):
        return x + self.expand(self.cfp(self.reduce(x)))

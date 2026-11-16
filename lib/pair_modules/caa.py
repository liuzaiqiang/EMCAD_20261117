import torch
from torch import nn
from functools import partial
from timm.models.helpers import named_apply
from lib.decoders import act_layer, _init_weights

class ContentAwareAntiAliasUpsample(nn.Module):
    """抗混叠基底与内容门控残差的可消融上采样。"""
    def __init__(self, channels, mode='caa', residual_scale=0.1):
        super(ContentAwareAntiAliasUpsample, self).__init__()
        valid_modes = {'aa_only', 'content_only', 'caa'}
        if mode not in valid_modes:
            raise ValueError('Unknown content-aware upsampling mode: {}'.format(mode))
        if float(residual_scale) < 0.0:
            raise ValueError('Caa residual scale must be non-negative')
        self.channels = int(channels)
        self.mode = mode
        self.residual_scale = (None if mode == 'aa_only' else
                               nn.Parameter(torch.tensor(float(residual_scale))))

        if self.mode == 'aa_only':
            self.content_gate = None
            self.residual_dw = None
        else:
            hidden = max(8, self.channels // 4)
            self.content_gate = nn.Sequential(
                nn.Conv2d(self.channels, hidden, kernel_size=1, bias=True),
                act_layer('relu', inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            )
            self.residual_dw = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, padding=1,
                groups=self.channels, bias=False)
        self.init_weights('normal')

        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32) / 16.0
        self.register_buffer('anti_alias_kernel', kernel.view(1, 1, 3, 3))

    def forward(self, x):
        if self.mode == 'content_only':
            base = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        else:
            kernel = self.anti_alias_kernel.to(dtype=x.dtype, device=x.device)
            kernel = kernel.expand(self.channels, 1, 3, 3)
            pad_mode = 'reflect' if min(x.shape[-2:]) > 1 else 'replicate'
            filtered = torch.nn.functional.pad(x, (1, 1, 1, 1), mode=pad_mode)
            filtered = torch.nn.functional.conv2d(filtered, kernel, groups=self.channels)
            base = torch.nn.functional.interpolate(
                filtered, scale_factor=2, mode='bilinear', align_corners=False)

        if self.mode == 'aa_only':
            return base

        gate = torch.sigmoid(self.content_gate(base))
        residual = self.residual_dw(base)
        scale = torch.clamp(self.residual_scale, min=0.0, max=1.0)
        return base + scale * gate * residual

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

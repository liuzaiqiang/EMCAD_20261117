"""UACA prediction refinement adapted from plemeri/UACANet (MIT license).

The three context weights, query/key/value relation and residual logit update
follow the author's ``lib/modules/attention_module.py``.  For mutually
exclusive multiclass segmentation, foreground means any non-background class.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, kernel_size, relu=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                        padding=kernel_size // 2, bias=False),
              nn.BatchNorm2d(out_channels)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UACA(nn.Module):
    """Author's three-region context attention, with an external EMCAD head."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.channels = channels
        self.conv_query = nn.Sequential(_conv(in_channels, channels, 3, True),
                                        _conv(channels, channels, 3, True))
        self.conv_key = nn.Sequential(_conv(in_channels, channels, 1, True),
                                      _conv(channels, channels, 1, True))
        self.conv_value = nn.Sequential(_conv(in_channels, channels, 1, True),
                                        _conv(channels, channels, 1, True))
        self.conv_out1 = _conv(channels, channels, 3, True)
        self.conv_out2 = _conv(in_channels + channels, channels, 3, True)
        self.conv_out3 = _conv(channels, channels, 3, True)

    def forward(self, feature, previous_logits, output_head):
        batch, _, height, width = feature.shape
        previous_logits = F.interpolate(previous_logits, size=(height, width),
                                        mode="bilinear", align_corners=False)
        if previous_logits.shape[1] == 1:
            foreground = torch.sigmoid(previous_logits)
        else:
            foreground = 1 - torch.softmax(previous_logits, dim=1)[:, :1]
        offset = foreground - 0.5
        probabilities = torch.cat((torch.clamp(offset, 0, 1),
                                   torch.clamp(-offset, 0, 1),
                                   0.5 - torch.abs(offset)), dim=1)
        vectors = torch.bmm(probabilities.reshape(batch, 3, height * width),
                            feature.reshape(batch, height * width, -1))
        vectors = vectors.permute(0, 2, 1).unsqueeze(-1)
        query = self.conv_query(feature).reshape(batch, self.channels, -1).permute(0, 2, 1)
        key = self.conv_key(vectors).reshape(batch, self.channels, 3)
        value = self.conv_value(vectors).reshape(batch, self.channels, 3).permute(0, 2, 1)
        relation = F.softmax(torch.bmm(query, key) * self.channels ** -0.5, dim=-1)
        context = torch.bmm(relation, value).permute(0, 2, 1)
        context = context.contiguous().reshape(batch, self.channels, height, width)
        context = self.conv_out1(context)
        refined = self.conv_out3(self.conv_out2(torch.cat((feature, context), dim=1)))
        return refined, output_head(refined) + previous_logits


class UACACascade(nn.Module):
    """Three author-style stages on EMCAD d3, d2 and d1 outputs."""

    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList(
            _conv(channels[i - 1], channels[i], 1, True) for i in range(1, 4)
        )
        self.stages = nn.ModuleList(UACA(2 * channels[i], channels[i]) for i in range(1, 4))

    def forward(self, features, heads):
        refined = features[0]
        logits = heads[0](refined)
        results = [logits]
        for i, stage in enumerate(self.stages, start=1):
            high = F.interpolate(refined, size=features[i].shape[-2:],
                                 mode="bilinear", align_corners=False)
            high = self.projections[i - 1](high)
            refined, logits = stage(torch.cat((features[i], high), dim=1), logits, heads[i])
            results.append(logits)
        return results

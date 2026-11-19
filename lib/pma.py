"""LB-UNet prediction-map auxiliary branches adapted to EMCAD's four stages.

RBP and PIF follow Xu & Tong, MICCAI 2024, equations (1)-(3).  The
lowest-resolution EMCAD stage has no paired encoder skip; the remaining three
stages use the paper's shallow-stage region/boundary weights.
"""

import torch
from torch import nn


class PredictionMapAuxiliary(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Deep to shallow: region at all four EMCAD stages, boundary at the
        # three stages with skip connections.
        self.region = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels)
        self.boundary = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels[1:])
        self.region_weights = (0.2, 0.3, 0.4, 0.5)
        self.boundary_weights = (0.1, 0.2, 0.3)

    def predict(self, feature, stage):
        region_logit = self.region[stage](feature)
        region = region_logit.sigmoid()
        if stage == 0:
            return feature + feature * region, region_logit, None
        boundary_logit = self.boundary[stage - 1](feature)
        boundary = boundary_logit.sigmoid()
        preserved = feature + feature * region + feature * boundary
        return preserved, region_logit, boundary_logit

    def fuse(self, preserved, skip, region_logit, boundary_logit, stage):
        # PIF: D_i + E_i + w1 E_i R_i + w2 E_i B_i.
        return (preserved + skip + self.region_weights[stage] * skip * region_logit.sigmoid()
                + self.boundary_weights[stage - 1] * skip * boundary_logit.sigmoid())

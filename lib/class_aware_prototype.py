"""Optional cross-scale, class-aware calibration of the finest decoder feature.

Coarse decoder logits provide soft class assignments for per-image prototypes.
The module never reads labels, dataset identity, or test-set statistics.
Binary segmentation uses an explicit background/foreground prototype pair.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ClassAwareCrossScalePrototype(nn.Module):
    """Pool d3/d2 class prototypes and use them to refine d1."""

    def __init__(self, channels3, channels2, channels1, num_classes,
                 embedding_dim=32):
        super().__init__()
        if min(channels3, channels2, channels1, num_classes, embedding_dim) < 1:
            raise ValueError("All channel and class counts must be positive")
        self.num_classes = num_classes
        self.prototype_count = 2 if num_classes == 1 else num_classes
        self.embed3 = nn.Conv2d(channels3, embedding_dim, 1, bias=False)
        self.embed2 = nn.Conv2d(channels2, embedding_dim, 1, bias=False)
        self.query1 = nn.Conv2d(channels1, embedding_dim, 1, bias=False)
        self.output1 = nn.Conv2d(embedding_dim, channels1, 1, bias=False)
        self.scale_logits = nn.Parameter(torch.zeros(self.prototype_count, 2))
        self.logit_scale = nn.Parameter(torch.tensor(2.302585093))  # log(10)
        self.gain = nn.Parameter(torch.zeros(1))

    def _assignments(self, logits):
        if logits.shape[1] != self.num_classes:
            raise ValueError("Logit channels do not match num_classes")
        if self.num_classes == 1:
            foreground = torch.sigmoid(logits.float())
            return torch.cat((1.0 - foreground, foreground), dim=1)
        return logits.float().softmax(dim=1)

    def _prototype(self, feature, logits, projection):
        assignment = self._assignments(logits).flatten(2)
        embedding = projection(feature.float()).flatten(2)
        mass = assignment.sum(dim=2).clamp_min(1e-6)
        prototype = torch.bmm(assignment, embedding.transpose(1, 2)) / mass.unsqueeze(2)
        # Confidence is the assignment probability averaged over its own mass.
        confidence = assignment.square().sum(dim=2) / mass
        return prototype, confidence.clamp_min(1e-6)

    def forward(self, d3, d2, d1, logits3, logits2):
        if d3.ndim != 4 or d2.ndim != 4 or d1.ndim != 4:
            raise ValueError("Decoder features must be NCHW tensors")
        if logits3.shape[:1] != d3.shape[:1] or logits3.shape[-2:] != d3.shape[-2:]:
            raise ValueError("d3 logits must match d3 batch and spatial shape")
        if logits2.shape[:1] != d2.shape[:1] or logits2.shape[-2:] != d2.shape[-2:]:
            raise ValueError("d2 logits must match d2 batch and spatial shape")
        if d1.shape[0] != d2.shape[0] or d1.shape[0] != d3.shape[0]:
            raise ValueError("Decoder stages must share the same batch size")

        with torch.autocast(device_type=d1.device.type, enabled=False):
            prototype3, confidence3 = self._prototype(d3, logits3, self.embed3)
            prototype2, confidence2 = self._prototype(d2, logits2, self.embed2)
            reliability = torch.stack((confidence3, confidence2), dim=2)
            scale_weights = (self.scale_logits.unsqueeze(0)
                             + reliability.log()).softmax(dim=2)
            prototypes = (scale_weights[:, :, :1] * prototype3
                          + scale_weights[:, :, 1:] * prototype2)

            query = F.normalize(self.query1(d1.float()), dim=1, eps=1e-6)
            keys = F.normalize(prototypes, dim=2, eps=1e-6)
            similarity = torch.einsum("bdhw,bkd->bkhw", query, keys)
            class_weights = (similarity * self.logit_scale.exp().clamp(max=20.0)).softmax(dim=1)
            context = torch.einsum("bkhw,bkd->bdhw", class_weights, prototypes)
            refined = d1.float() + torch.tanh(self.gain) * self.output1(context)
        return refined.to(dtype=d1.dtype)

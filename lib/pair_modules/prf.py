"""Pixel-wise reliability weighted fusion of the four EMCAD prediction heads."""

import torch
from torch import nn
from torch.nn import functional as F


class PixelReliabilityFusion(nn.Module):
    """Fuse logits with four learned per-pixel reliability scores.

    The original four segmentation heads remain intact. Reliability labels are
    computed only from training masks, never supplied to inference.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleList(nn.Conv2d(num_classes, 1, 1) for _ in range(4))

    def scores(self, outputs):
        if len(outputs) != 4:
            raise ValueError("Expected four EMCAD logits")
        if any(output.shape != outputs[0].shape for output in outputs):
            raise ValueError("EMCAD logits must share shape before fusion")
        return torch.cat([head(output) for head, output in zip(self.heads, outputs)], dim=1)

    def forward(self, outputs):
        scores = self.scores(outputs)
        weights = torch.softmax(scores, dim=1)
        fused = sum(weights[:, index:index + 1] * output
                    for index, output in enumerate(outputs))
        return fused

    def reliability_loss(self, outputs, target):
        """BCE against each head's detached pixel correctness during training."""
        with torch.no_grad():
            if self.num_classes == 1:
                truth = target.float()
                if truth.ndim == 3:
                    truth = truth.unsqueeze(1)
                correctness = torch.cat([
                    ((torch.sigmoid(output.detach()) >= 0.5) == (truth >= 0.5)).float()
                    for output in outputs
                ], dim=1)
            else:
                truth = target.long()
                if truth.ndim == 4 and truth.shape[1] == 1:
                    truth = truth[:, 0]
                correctness = torch.stack([
                    (output.detach().argmax(dim=1) == truth).float()
                    for output in outputs
                ], dim=1)
        return F.binary_cross_entropy_with_logits(self.scores(outputs).float(), correctness)


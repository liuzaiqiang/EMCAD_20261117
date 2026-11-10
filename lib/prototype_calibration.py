"""Prediction-guided, class-aware cross-scale prototype calibration.

The block uses only decoder features and predictions. It has no access to masks,
batch peers, or dataset statistics at inference time. Each image is independent.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CrossScalePrototypeCalibration(nn.Module):
    """Calibrate four native-resolution segmentation heads with other scales.

    For a one-logit binary head, internal classes are background and foreground.
    For a multi-logit head, they are the native softmax classes. Every scale
    supplies confidence-weighted feature centroids; each target scale receives
    the mean of the other three scales' valid class centroids. A zero-initialized
    gain makes enabling this block exactly neutral before optimization.
    """

    def __init__(self, channels, num_classes, embedding_dim=32):
        super().__init__()
        if len(channels) != 4 or num_classes < 1:
            raise ValueError("Expected four decoder scales and positive num_classes")
        self.num_classes = num_classes
        self.projections = nn.ModuleList(
            nn.Conv2d(channel, embedding_dim, 1, bias=False)
            for channel in channels
        )
        self.gain = nn.Parameter(torch.zeros(4))

    def forward(self, features, logits):
        if len(features) != 4 or len(logits) != 4:
            raise ValueError("Expected four features and four logits")
        embeddings = [F.normalize(proj(feature).float(), dim=1, eps=1e-6)
                      for proj, feature in zip(self.projections, features)]
        prototypes = []
        validities = []
        for embedding, logit in zip(embeddings, logits):
            if self.num_classes == 1:
                foreground = torch.sigmoid(logit.float())
                probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
            else:
                probabilities = torch.softmax(logit.float(), dim=1)
            # Stop gradients through the self-generated weights; the heads still
            # receive gradients through the additive calibrated logits.
            weights = probabilities.detach().flatten(2)
            mass = weights.sum(dim=-1)
            centroid = torch.bmm(weights, embedding.flatten(2).transpose(1, 2))
            centroid = F.normalize(centroid / mass.clamp_min(1e-6).unsqueeze(-1),
                                   dim=-1, eps=1e-6)
            prototypes.append(centroid)
            validities.append((mass > 1e-4).to(centroid.dtype))

        calibrated = []
        for index, (embedding, logit) in enumerate(zip(embeddings, logits)):
            source = sum(prototypes[j] * validities[j].unsqueeze(-1)
                         for j in range(4) if j != index)
            count = sum(validities[j] for j in range(4) if j != index)
            source = F.normalize(source / count.clamp_min(1).unsqueeze(-1),
                                 dim=-1, eps=1e-6)
            similarity = torch.einsum("bdh,bkd->bkh", embedding.flatten(2), source)
            similarity = similarity.reshape(logit.shape[0], -1, *logit.shape[-2:])
            if self.num_classes == 1:
                correction = similarity[:, 1:2] - similarity[:, 0:1]
            else:
                correction = similarity - similarity.mean(dim=1, keepdim=True)
            # Tanh bounds the trainable influence while leaving the baseline
            # heads and loss function intact.
            calibrated.append(logit + torch.tanh(self.gain[index]) * correction.to(logit.dtype))
        return calibrated

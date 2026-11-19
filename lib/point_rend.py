"""PointRend semantic point head adapted to EMCAD's 2D segmentation logits.

Based on Kirillov et al. (CVPR 2020) and the authors' Detectron2
projects/PointRend/point_rend/{point_features,point_head,semantic_seg}.py.
"""

import torch
from torch import nn
from torch.nn import functional as F


def point_sample(features, coords, mode="bilinear"):
    """Sample [B,C,H,W] at normalized [B,P,2] xy coordinates."""
    return F.grid_sample(features, 2.0 * coords.unsqueeze(2) - 1.0,
                         mode=mode, align_corners=False).squeeze(3)


def uncertainty(logits):
    """Larger values mean less confident predictions."""
    if logits.size(1) == 1:
        return -logits.abs()
    top_two = logits.topk(2, dim=1).values
    return top_two[:, 1:2] - top_two[:, 0:1]


def training_points(coarse_logits, num_points=196, oversample_ratio=3,
                    importance_ratio=0.75):
    """Official PointRend biased sampling: k=3, beta=0.75."""
    batch = coarse_logits.shape[0]
    count = int(num_points * oversample_ratio)
    coords = torch.rand(batch, count, 2, device=coarse_logits.device)
    with torch.no_grad():
        scores = uncertainty(point_sample(coarse_logits.detach(), coords))[:, 0]
        selected = int(num_points * importance_ratio)
        indices = scores.topk(selected, dim=1).indices
        important = coords.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        random = torch.rand(batch, num_points - selected, 2, device=coords.device)
    return torch.cat((important, random), dim=1)


def grid_points(logits, count=784):
    batch, _, height, width = logits.shape
    indices = uncertainty(logits).flatten(2)[:, 0].topk(min(count, height * width), dim=1).indices
    coords = torch.stack(((indices % width).float().add_(0.5) / width,
                          torch.div(indices, width, rounding_mode="floor").float().add_(0.5) / height), dim=-1)
    return indices, coords


class PointHead(nn.Module):
    """Author's three 1x1 Conv1d MLP layers with coarse features at each layer."""

    def __init__(self, fine_channels, classes, width=256):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = fine_channels + classes
        for _ in range(3):
            layer = nn.Conv1d(in_channels, width, 1)
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            in_channels = width + classes
        self.predictor = nn.Conv1d(in_channels, classes, 1)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, fine, coarse):
        features = torch.cat((fine, coarse), dim=1)
        for layer in self.layers:
            features = torch.cat((F.relu(layer(features)), coarse), dim=1)
        return self.predictor(features)


class PointRendRefiner(nn.Module):
    """Refine H/4 EMCAD logits to H via two adaptive subdivision steps."""

    def __init__(self, decoder_channels, classes):
        super().__init__()
        # Image resolution features supply detail missing from the H/4 decoder map.
        self.image_features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1),
                                            nn.ReLU(inplace=True))
        self.point_head = PointHead(decoder_channels + 32, classes)

    def _predict(self, coarse, decoder_features, image_features, coords):
        fine = torch.cat((point_sample(decoder_features, coords),
                          point_sample(image_features, coords)), dim=1)
        return self.point_head(fine, point_sample(coarse, coords))

    def train_points(self, coarse, decoder_features, image):
        coords = training_points(coarse)
        prediction = self._predict(coarse, decoder_features,
                                   self.image_features(image), coords)
        return prediction, coords

    def refine(self, coarse, decoder_features, image):
        image_features = self.image_features(image)
        refined = coarse
        for _ in range(2):
            refined = F.interpolate(refined, scale_factor=2, mode="bilinear",
                                    align_corners=False)
            indices, coords = grid_points(refined)
            point_logits = self._predict(coarse, decoder_features, image_features, coords)
            batch, classes, height, width = refined.shape
            refined = refined.reshape(batch, classes, height * width).scatter(
                2, indices.unsqueeze(1).expand(-1, classes, -1), point_logits
            ).reshape(batch, classes, height, width)
        return refined


def point_loss(prediction, coords, target):
    """Sample augmented mask at the same points; handle binary and multiclass."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    labels = point_sample(target.float(), coords, mode="nearest")[:, 0]
    if prediction.size(1) == 1:
        return F.binary_cross_entropy_with_logits(prediction[:, 0], labels)
    return F.cross_entropy(prediction, labels.long())

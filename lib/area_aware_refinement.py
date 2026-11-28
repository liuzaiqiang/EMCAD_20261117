"""Area-aware sparse correction for EMCAD segmentation logits."""

import torch
from torch import nn
from torch.nn import functional as F


def point_sample(features, coords, mode="bilinear", padding_mode="zeros"):
    """Sample [B,C,H,W] at normalized [B,P,2] xy coordinates."""
    return F.grid_sample(features, 2.0 * coords.unsqueeze(2) - 1.0,
                         mode=mode, padding_mode=padding_mode, align_corners=False).squeeze(3)


def _class_probabilities(logits):
    if logits.size(1) == 1:
        foreground = logits.sigmoid()
        return torch.cat((1.0 - foreground, foreground), dim=1)
    return logits.softmax(dim=1)


@torch.no_grad()
def foreground_rarity(fine, guide):
    """Estimate foreground occupancy from the native H/4 and H/8 predictions."""
    areas = []
    for logits in (fine, guide):
        probabilities = _class_probabilities(logits.float())
        top = probabilities.topk(2, dim=1).values
        confidence = top[:, :1] - top[:, 1:2]
        support = (probabilities == top[:, :1]) * confidence
        areas.append(support.mean(dim=(2, 3))[:, 1:])
    area = (areas[0] + areas[1]) * 0.5
    # A bounded weight and a visibility check keep absent classes from getting a bonus.
    rarity = (area > 0) * (0.05 / (0.05 + area))
    return F.pad(rarity, (1, 0))


@torch.no_grad()
def area_aware_indices(fine, guide, rarity, count):
    """Select from aligned [B,C,N] logits, reserving half for uncertainty alone."""
    probabilities = _class_probabilities(fine.float())
    top, pair = probabilities.topk(2, dim=1)
    ambiguity = 2.0 * top[:, 1] / top.sum(dim=1).clamp_min(1e-6)
    guide_pair = _class_probabilities(guide.float()).gather(1, pair)
    opposition = (guide_pair[:, 1] - guide_pair[:, 0]).clamp_min(0)
    pair_rarity = rarity.gather(1, pair[:, 0]).maximum(rarity.gather(1, pair[:, 1]))
    priority = ambiguity + pair_rarity * (0.5 * ambiguity + opposition)

    count = min(count, fine.size(-1))
    general = ambiguity.topk(count // 2, dim=1).indices
    priority.scatter_(1, general, float("-inf"))
    focused = priority.topk(count - general.size(1), dim=1).indices
    return torch.cat((general, focused), dim=1)


def stratified_points(batch, side, device):
    """Draw one point from each tile to cover the whole image during training."""
    axis = torch.arange(side, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    tiles = torch.stack((x, y), dim=-1).reshape(1, side * side, 2)
    offsets = torch.rand(batch, side * side, 2, device=device)
    return (tiles + offsets) / side


class AreaAwarePointRefiner(nn.Module):
    """Use adjacent EMCAD heads to prioritize points and correct fine logits."""

    def __init__(self, decoder_channels, classes):
        super().__init__()
        probability_channels = max(2, classes)
        # Center intensity and four center-to-neighbor differences for RGB/grayscale-adapted input.
        self.correction = nn.Sequential(
            nn.Conv1d(decoder_channels + 15 + classes + probability_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, classes, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def _predict(self, fine, guide, decoder_features, image, coords):
        batch, count, _ = coords.shape
        height, width = image.shape[-2:]
        offsets = coords.new_tensor([(0, 0), (-1 / width, 0), (1 / width, 0),
                                     (0, -1 / height), (0, 1 / height)])
        stencil = (coords.unsqueeze(2) + offsets).reshape(batch, count * 5, 2)
        intensities = point_sample(image, stencil, padding_mode="border")
        intensities = intensities.reshape(batch, 3, count, 5)
        center = intensities[..., :1]
        contrast = torch.cat((center, intensities[..., 1:] - center), dim=-1)
        contrast = contrast.transpose(2, 3).flatten(1, 2)

        anchor = point_sample(fine, coords, padding_mode="border")
        guide_points = point_sample(guide, coords, padding_mode="border")
        difference = _class_probabilities(guide_points) - _class_probabilities(anchor)
        detail = point_sample(decoder_features, coords, padding_mode="border")
        correction = self.correction(torch.cat((detail, contrast, anchor, difference), dim=1))
        # Keep a point correction local in logit space, especially for confident pixels.
        return anchor, 3.0 * correction.tanh()

    def train_points(self, fine, guide, decoder_features, image):
        batch = fine.size(0)
        with torch.no_grad():
            coords = stratified_points(batch, 24, fine.device)
            fine_points = point_sample(fine, coords, padding_mode="border")
            guide_points = point_sample(guide, coords, padding_mode="border")
            rarity = foreground_rarity(fine, guide)
            indices = area_aware_indices(fine_points, guide_points, rarity, 147)
            important = coords.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
            exploration = stratified_points(batch, 7, fine.device)
            coords = torch.cat((important, exploration), dim=1)
        anchor, correction = self._predict(fine, guide, decoder_features, image, coords)
        return anchor + correction, coords

    def refine(self, fine, guide, decoder_features, image):
        rarity = foreground_rarity(fine, guide)
        height, width = image.shape[-2:]
        size = (height, width)
        baseline = F.interpolate(fine, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            aligned_guide = F.interpolate(guide, size=size, mode="bilinear", align_corners=False)
            indices = area_aware_indices(baseline.flatten(2), aligned_guide.flatten(2), rarity, 1568)
            y = torch.div(indices, width, rounding_mode="floor")
            x = indices % width
            coords = torch.stack(((x + 0.5) / width, (y + 0.5) / height), dim=-1)

        _, corrections = self._predict(fine, guide, decoder_features, image, coords)
        flat = baseline.flatten(2)
        indices = indices.unsqueeze(1).expand_as(corrections)
        updated = flat.gather(2, indices) + corrections.to(flat.dtype)
        return flat.scatter(2, indices, updated).reshape_as(baseline)


def point_loss(prediction, coords, target):
    """Sample augmented mask at the same points; handle binary and multiclass."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    labels = point_sample(target.float(), coords, mode="nearest")[:, 0]
    if prediction.size(1) == 1:
        return F.binary_cross_entropy_with_logits(prediction[:, 0], labels)
    return F.cross_entropy(prediction, labels.long())

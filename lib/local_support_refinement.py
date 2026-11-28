"""Sparse refinement of locally small or narrow foreground structures in EMCAD."""

import torch
from torch import nn
from torch.nn import functional as F


def point_sample(features, coords, mode="bilinear"):
    grid = 2.0 * coords.unsqueeze(2) - 1.0
    return F.grid_sample(features, grid, mode=mode, padding_mode="border",
                         align_corners=False).squeeze(3)


@torch.no_grad()
def local_support_scores(logits):
    """Measure foreground transitions whose neighborhood has limited same-class support."""
    logits = logits.float()
    if logits.size(1) == 1:
        foreground = logits.sigmoid()
        probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
    else:
        probabilities = logits.softmax(dim=1)
        foreground = probabilities[:, 1:]

    top = probabilities.topk(2, dim=1).values
    uncertainty = 2.0 * top[:, 1:2] / top.sum(dim=1, keepdim=True).clamp_min(1e-6)
    wins = (foreground == top[:, :1]) & (top[:, :1] > top[:, 1:2])
    coverage = F.avg_pool2d(wins.float(), 5, stride=1, padding=2, count_include_pad=False)
    peak = F.max_pool2d(foreground, 3, stride=1, padding=1)
    trough = -F.max_pool2d(-foreground, 3, stride=1, padding=1)
    support = ((peak - trough) * peak * (1.0 - coverage).square()).amax(dim=1, keepdim=True)
    return support, uncertainty


@torch.no_grad()
def select_anchors(scores, count, exploration=0):
    """Rank the best anchor from each 2×2 native decoder tile."""
    scores, anchors = F.max_pool2d(scores, 2, stride=2, ceil_mode=True, return_indices=True)
    scores, anchors = scores.flatten(1), anchors.flatten(1)
    count = min(count, scores.size(1))
    exploration = min(exploration, count // 4)
    selected = scores.topk(count - exploration, dim=1).indices
    if exploration:
        random_scores = torch.rand_like(scores)
        random_scores.scatter_(1, selected, -1.0)
        selected = torch.cat((selected, random_scores.topk(exploration, dim=1).indices), dim=1)
    return anchors.gather(1, selected)


class LocalSupportRefiner(nn.Module):
    """Predict one bounded affine logit correction over a window at each anchor."""

    def __init__(self, decoder_channels, classes):
        super().__init__()
        self.correction = nn.Sequential(nn.Conv1d(decoder_channels + classes + 11, 96, 1),
                                        nn.ReLU(inplace=True), nn.Conv1d(96, 3 * classes, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

        axis = torch.arange(-2, 3)
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("offset_x", dx.flatten(), persistent=False)
        self.register_buffer("offset_y", dy.flatten(), persistent=False)
        basis = torch.stack((torch.ones_like(dx), dx / 2, dy / 2)).flatten(1)
        self.register_buffer("basis", basis, persistent=False)
        taper = torch.exp(-0.45 * (dx.square() + dy.square())).flatten()
        self.register_buffer("taper", taper, persistent=False)

    def _predict_windows(self, coarse, decoder, image, count, exploration=0):
        support, uncertainty = local_support_scores(coarse)
        anchors = select_anchors(2.0 * support + 0.25 * uncertainty, count, exploration)
        batch, classes, coarse_h, coarse_w = coarse.shape
        height, width = image.shape[-2:]
        x = ((anchors % coarse_w + 0.5) * width / coarse_w).long().clamp(2, width - 3)
        row = torch.div(anchors, coarse_w, rounding_mode="floor")
        y = ((row + 0.5) * height / coarse_h).long().clamp(2, height - 3)
        x = x.unsqueeze(-1) + self.offset_x
        y = y.unsqueeze(-1) + self.offset_y
        pixels = (y * width + x).flatten(1)
        coords = torch.stack(((x + 0.5) / width, (y + 0.5) / height), dim=-1).flatten(1, 2)

        image_indices = pixels.unsqueeze(1).expand(-1, 3, -1)
        patch = image.flatten(2).gather(2, image_indices).reshape(batch, 3, -1, 5, 5)
        mean = patch.mean(dim=(-2, -1))
        dx = patch[..., 3:].mean(dim=(-2, -1)) - patch[..., :2].mean(dim=(-2, -1))
        dy = patch[..., 3:, :].mean(dim=(-2, -1)) - patch[..., :2, :].mean(dim=(-2, -1))
        native = torch.cat((decoder, coarse, support, uncertainty), dim=1).flatten(2)
        center = native.gather(2, anchors.unsqueeze(1).expand(-1, native.size(1), -1))
        coefficients = self.correction(torch.cat((center, mean, dx, dy), dim=1))
        coefficients = coefficients.reshape(batch, classes, 3, -1).transpose(2, 3)
        residual = 3.0 * (coefficients @ self.basis.to(coefficients.dtype)).tanh()
        return residual, pixels, coords

    def train_points(self, coarse, decoder, image):
        residual, _, coords = self._predict_windows(coarse, decoder, image, 32, exploration=8)
        weighted = (residual * self.taper).flatten(2)
        return point_sample(coarse, coords) + weighted, coords

    def refine(self, coarse, decoder, image, baseline):
        residual, pixels, _ = self._predict_windows(coarse, decoder, image, 128)
        weighted = (residual.float() * self.taper).flatten(2)
        indices = pixels.unsqueeze(1).expand_as(weighted)
        flat = baseline.flatten(2)
        summed = weighted.new_zeros(flat.shape).scatter_add_(2, indices, weighted)
        weights = self.taper.repeat(residual.size(2)).expand(flat.size(0), -1)
        mass = weights.new_zeros(flat.size(0), flat.size(2)).scatter_add_(1, pixels, weights)
        correction = summed / mass.clamp_min_(1.0).unsqueeze(1)
        return baseline + correction.reshape_as(baseline).to(baseline.dtype)


def point_loss(prediction, coords, target):
    """Supervise window logits using the corresponding pixels in the augmented mask."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    labels = point_sample(target.float(), coords, mode="nearest")[:, 0]
    if prediction.size(1) == 1:
        return F.binary_cross_entropy_with_logits(prediction[:, 0], labels)
    return F.cross_entropy(prediction, labels.long())

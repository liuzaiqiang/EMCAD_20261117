"""Percentile-normalized signed-distance boundary supervision.

The public abstract of Li et al., "A Robust Boundary Loss for High Accuracy
Medical Image Segmentation" (IJCNN 2026, DOI: 10.1109/IJCNN69854.2026.11728433)
describes signed distance maps, percentile truncation, and normalization.
Its full formula and hyperparameters are not publicly available here. This is
an explicit implementation of that mechanism, combined with the signed
distance integral of Kervadec et al. (MIDL 2019). It is not claimed to be a
verbatim reproduction of the IJCNN paper.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def validate_robust_boundary_config(enabled, weight, percentile):
    if enabled not in (0, 1):
        raise ValueError("use_robust_boundary_loss must be 0 or 1")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("robust_boundary_weight must be finite and positive")
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("robust_boundary_percentile must be in (0, 100]")


def normalized_signed_distance(mask, percentile=95.0):
    """Return [-1, 1] distances; negative inside, positive outside.

    Interior and exterior distances receive independent percentile scales so
    a small foreground does not let the larger background set both scales.
    Empty or full masks have no two-sided boundary and return None.
    """
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not mask.any() or mask.all():
        return None
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    inside_scale = np.percentile(inside[mask], percentile)
    outside_scale = np.percentile(outside[~mask], percentile)
    signed = np.clip(outside / outside_scale, 0, 1)
    signed[mask] = -np.clip(inside[mask] / inside_scale, 0, 1)
    return signed.astype(np.float32, copy=False)


def robust_boundary_loss(logits, target, percentile=95.0):
    """Mean signed-distance error over present (image, foreground class) units.

    logits: [B, 1 or C, H, W]; target: [B, H, W]. Binary target is 0/1;
    multiclass target uses integer class IDs with background 0. The term
    (p-y)*distance differs from the boundary integral p*distance only by a
    target-dependent constant and is zero for a perfect prediction.
    """
    if logits.ndim != 4 or target.ndim != 3 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    classes = logits.shape[1]
    if classes == 1:
        if torch.any((target != 0) & (target != 1)):
            raise ValueError("Binary target must contain only 0 and 1")
        probabilities = torch.sigmoid(logits.float())
        labels = target.unsqueeze(1).to(probabilities.dtype)
    else:
        if torch.any((target < 0) | (target >= classes)):
            raise ValueError("Target class is outside the logits channels")
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1:]
        labels = F.one_hot(target.long(), num_classes=classes).permute(0, 3, 1, 2)[:, 1:].to(probabilities.dtype)

    label_masks = labels.detach().to(device="cpu", dtype=torch.uint8).numpy()
    terms = []
    for batch_index in range(label_masks.shape[0]):
        for class_index in range(label_masks.shape[1]):
            signed = normalized_signed_distance(label_masks[batch_index, class_index], percentile)
            if signed is None:
                continue
            distance = torch.from_numpy(signed).to(device=logits.device)
            terms.append(((probabilities[batch_index, class_index] - labels[batch_index, class_index]) * distance).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()

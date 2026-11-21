"""Two-sided 2D HD-DT auxiliary loss for training segmentation heads.

Karimi and Salcudean, IEEE TMI 2020, Eq. (8):
https://doi.org/10.1109/TMI.2019.2930068

The binary formulation is applied independently to foreground classes. This
is a training surrogate for Hausdorff distance, not the validation HD95 metric.
Distances are measured in resized training-image pixels, not physical units.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F


def _boundary_distance(mask):
    """Unsigned Euclidean distance to a binary mask's boundary in pixels."""
    if not mask.any():
        # An absent class has no boundary. Use the image diagonal as a finite
        # penalty when the other mask contains a spurious foreground region.
        return np.full(mask.shape, np.hypot(*mask.shape), dtype=np.float32)
    if mask.all():
        return distance_transform_edt(mask).astype(np.float32)
    return (distance_transform_edt(mask) +
            distance_transform_edt(~mask)).astype(np.float32)


def hausdorff_dt_loss(logits, target):
    """Eq. (8) with alpha=2, averaged over foreground classes and 2D pixels.

    Binary: logits [B,1,H,W], mask [B,1,H,W] or [B,H,W].
    Multiclass: logits [B,C,H,W], integer labels [B,H,W].
    The hard masks used for distance transforms are detached; gradients flow
    through the squared soft-probability error only.
    """
    if logits.ndim != 4 or logits.shape[1] < 1:
        raise ValueError("logits must have BCHW shape")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("target shape does not match logits")

    logits = logits.float()  # stable under AMP; no change to model parameters
    if logits.shape[1] == 1:
        probability = logits.sigmoid()
        truth = (target > 0.5).unsqueeze(1).to(probability.dtype)
        predicted = (probability.detach() > 0.5)
    else:
        if target.is_floating_point():
            raise ValueError("multiclass target must use integer class indices")
        probability = logits.softmax(dim=1)[:, 1:]
        truth = F.one_hot(target.long(), logits.shape[1]).permute(0, 3, 1, 2)[:, 1:]
        truth = truth.to(probability.dtype)
        predicted = (logits.detach().argmax(dim=1).unsqueeze(1) ==
                     torch.arange(1, logits.shape[1], device=logits.device)[None, :, None, None])

    truth_masks = truth.detach().bool().cpu().numpy()
    predicted_masks = predicted.cpu().numpy()
    weights = np.empty(truth_masks.shape, dtype=np.float32)
    for batch in range(weights.shape[0]):
        for channel in range(weights.shape[1]):
            if not truth_masks[batch, channel].any() and not predicted_masks[batch, channel].any():
                weights[batch, channel] = 0
                continue
            d_truth = _boundary_distance(truth_masks[batch, channel])
            d_predicted = _boundary_distance(predicted_masks[batch, channel])
            weights[batch, channel] = d_truth ** 2 + d_predicted ** 2
    distance_weight = torch.from_numpy(weights).to(device=logits.device)
    return ((probability - truth).square() * distance_weight).mean()

"""Training-only size-aware hard-positive auxiliary loss.

This combines established focal weighting with per-image foreground size
weighting. It has no trainable parameters and is never part of inference.
"""

import torch
from torch.nn import functional as F

HARD_POSITIVE_WEIGHT = 0.05


def small_target_hard_positive_loss(logits, target, reference_fraction=0.10,
                                    max_size_weight=3.0, gamma=2.0):
    """Return a class-balanced loss on difficult foreground pixels.

    Multiclass logits are [B,C,H,W] and integer targets [B,H,W]. Binary logits
    are [B,1,H,W] and masks [B,1,H,W] or [B,H,W]. Missing foreground classes
    contribute zero; weights use only the current training labels.
    """
    if logits.ndim != 4:
        raise ValueError('logits must have BCHW shape')
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('target shape does not match logits')
    if not (0 < reference_fraction <= 1 and max_size_weight >= 1 and gamma >= 0):
        raise ValueError('invalid small-target loss hyperparameters')
    # Keep per-class probabilities and tiny area fractions stable under AMP.
    logits = logits.float()

    if logits.shape[1] == 1:
        positive = (target > 0.5).unsqueeze(1).to(dtype=logits.dtype)
        log_probability = F.logsigmoid(logits)
        foreground = positive
    else:
        if target.is_floating_point():
            raise ValueError('multiclass target must contain integer class indices')
        one_hot = F.one_hot(target.long(), num_classes=logits.shape[1])
        foreground = one_hot.permute(0, 3, 1, 2)[:, 1:].to(dtype=logits.dtype)
        log_probability = F.log_softmax(logits, dim=1)[:, 1:]

    counts = foreground.sum(dim=(2, 3))
    area_fraction = counts / (logits.shape[2] * logits.shape[3])
    size_weight = (reference_fraction / area_fraction.clamp_min(1.0 /
                    (logits.shape[2] * logits.shape[3]))).sqrt().clamp(1.0, max_size_weight)
    size_weight = size_weight.detach()
    difficulty = (1.0 - log_probability.exp()).detach().pow(gamma)
    per_class = (-log_probability * difficulty * foreground).sum(dim=(2, 3))
    per_class = per_class / counts.clamp_min(1.0)
    present = (counts > 0).to(dtype=per_class.dtype)
    return (per_class * size_weight * present).sum() / present.sum().clamp_min(1.0)

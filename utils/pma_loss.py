"""Auxiliary PMA training loss (LB-UNet, MICCAI 2024, Eq. 5-7).

The source paper uses binary lesion masks.  Here one region channel means the
union of foreground classes, and boundary targets include both exterior and
inter-class edges.  Labels are generated after augmentation to stay aligned.
"""

import torch
import torch.nn.functional as F


def boundary_target(mask):
    """Two-sided one-pixel contour of a binary or integer class map."""
    if mask.ndim == 4:
        mask = mask[:, 0]
    mask = mask.long()
    # Compare neighboring integer labels rather than converting to foreground:
    # adjacent Synapse/ACDC structures need their shared boundary supervised.
    edge = torch.zeros_like(mask, dtype=torch.bool)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]
    return edge.unsqueeze(1).float()


def _bce_dice(logits, target):
    # Source auxiliary loss: 0.5 * BCE + Dice, smooth=1.
    # Author implementation applies sigmoid at the prediction stage, then
    # bilinearly upsamples the probability map for BCE and Dice.
    probability = F.interpolate(logits.float().sigmoid(), size=target.shape[-2:],
                                mode='bilinear', align_corners=True)
    target = target.float()
    bce = F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), target)
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dims)
    dice = 1 - ((2 * intersection + 1) /
                (probability.sum(dims) + target.sum(dims) + 1)).mean()
    return 0.5 * bce + dice


def pma_auxiliary_loss(pma_maps, mask):
    regions, boundaries = pma_maps
    if mask.ndim == 4:
        mask = mask[:, 0]
    region_target = (mask > 0).unsqueeze(1).float()
    edge_target = boundary_target(mask)
    # Deep to shallow stage mapping: EMCAD has four stages versus LB-UNet's
    # five auxiliary stages; retain the source weights for matched stages.
    loss = sum(weight * _bce_dice(pred, region_target)
               for weight, pred in zip((0.2, 0.3, 0.4, 0.5), regions))
    loss += sum(weight * _bce_dice(pred, edge_target)
                for weight, pred in zip((0.1, 0.2, 0.3), boundaries))
    return loss

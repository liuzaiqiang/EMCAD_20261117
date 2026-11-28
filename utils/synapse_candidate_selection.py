"""Pure selection rules for Synapse's optional online top-K checkpoints."""

import math
from decimal import Decimal


def validate_topk_config(enabled, topk, threshold_percent):
    if isinstance(enabled, bool) or enabled not in (0, 1):
        raise ValueError("synapse_topk_enabled must be 0 or 1")
    if enabled == 0:
        return
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError("synapse_topk_size must be a positive integer")
    if not math.isfinite(threshold_percent) or not 0 <= threshold_percent <= 100:
        raise ValueError("synapse_topk_dice_threshold must be a finite percentage in [0, 100]")


def is_new_best_dice(dice_fraction, best_dice_fraction):
    """The disabled mode keeps the earlier best.pth on a Dice tie."""
    return dice_fraction > best_dice_fraction


def should_enter_topk(dice_fraction, board, topk, threshold_percent):
    """Apply the threshold, then compare with the current board's lowest Dice."""
    validate_topk_config(1, topk, threshold_percent)
    if not math.isfinite(dice_fraction):
        return False
    if Decimal(str(dice_fraction)) * 100 < Decimal(str(threshold_percent)):
        return False
    return len(board) < topk or dice_fraction > min(item["rank_dice"] for item in board)


def add_topk_entry(board, entry, topk):
    """Return a descending board and the entry displaced by this one, if any."""
    ranked = sorted([*board, entry], key=lambda item: (-item["rank_dice"], item["epoch"]))
    return ranked[:topk], ranked[topk:]


def candidate_checkpoint_name(epoch, dice_fraction, hd95):
    if epoch < 1 or not math.isfinite(dice_fraction) or not math.isfinite(hd95):
        raise ValueError("Invalid Synapse candidate checkpoint metrics")
    return "epoch_{:03d}_valdice_{:.4f}_valhd95_{:.4f}.pth".format(
        epoch, dice_fraction * 100, hd95)

"""Surface-distance supervision for 2D multi-class segmentation.

The training datasets in this repository are resized 2D slices.  Therefore the
distance maps produced here are measured in pixels after preprocessing, not in
millimetres.  The implementation intentionally uses NumPy only so that the
loss does not introduce a new SciPy runtime dependency.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt
except ImportError:
    _scipy_distance_transform_edt = None

# 创新点：提供无 SciPy 环境下的二维 8 邻域距离变换 fallback。时间：20261002
def _chamfer_distance(mask: np.ndarray) -> np.ndarray:
    """Approximate distance to the nearest False pixel with an 8-neighbour chamfer pass."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W]")

    height, width = mask.shape
    infinity = np.float32(1e6)
    distance = np.where(mask, infinity, 0.0).astype(np.float32, copy=False)
    diagonal = np.float32(sqrt(2.0))

    for row in range(height):
        for col in range(width):
            value = distance[row, col]
            if row > 0:
                value = min(value, distance[row - 1, col] + 1.0)
                if col > 0:
                    value = min(value, distance[row - 1, col - 1] + diagonal)
                if col + 1 < width:
                    value = min(value, distance[row - 1, col + 1] + diagonal)
            if col > 0:
                value = min(value, distance[row, col - 1] + 1.0)
            distance[row, col] = value

    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            value = distance[row, col]
            if row + 1 < height:
                value = min(value, distance[row + 1, col] + 1.0)
                if col > 0:
                    value = min(value, distance[row + 1, col - 1] + diagonal)
                if col + 1 < width:
                    value = min(value, distance[row + 1, col + 1] + diagonal)
            if col + 1 < width:
                value = min(value, distance[row, col + 1] + 1.0)
            distance[row, col] = value

    return distance


# 创新点：统一二维 signed distance map 的距离变换实现并优先使用 Euclidean 加速路径。时间：20261002
def _distance_transform(mask: np.ndarray) -> np.ndarray:
    if _scipy_distance_transform_edt is not None:
        return _scipy_distance_transform_edt(mask).astype(np.float32, copy=False)
    return _chamfer_distance(mask)


# 创新点：从二维标签生成归一化 signed distance map，距离单位固定为预处理后的像素。时间：20261002
def build_signed_distance_maps(
    target: torch.Tensor,
    num_classes: int,
    distance_clip: float,
    include_background: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-sample signed distance maps and a validity mask.

    ``target`` has shape ``[B, H, W]``.  The returned map has shape
    ``[B, C, H, W]`` and is clipped to ``[-1, 1]`` after dividing by
    ``distance_clip``.  A class is valid only when it has both a foreground
    pixel and a background pixel in that slice; empty classes contribute zero
    rather than a NaN or an arbitrary penalty.
    """
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must have shape [B, H, W] or [B, 1, H, W]")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if distance_clip <= 0:
        raise ValueError("distance_clip must be positive")

    target_cpu = target.detach().to(device="cpu", dtype=torch.long).numpy()
    batch, height, width = target_cpu.shape
    maps = np.zeros((batch, num_classes, height, width), dtype=np.float32)
    valid = np.zeros((batch, num_classes), dtype=np.bool_)
    first_class = 0 if include_background else 1

    for batch_index in range(batch):
        for class_index in range(first_class, num_classes):
            class_mask = target_cpu[batch_index] == class_index
            if not class_mask.any() or class_mask.all():
                continue
            # Inside values are negative; outside values are positive.
            distance_inside = _distance_transform(class_mask)
            distance_outside = _distance_transform(~class_mask)
            signed = distance_outside - distance_inside
            maps[batch_index, class_index] = np.clip(
                signed / float(distance_clip), -1.0, 1.0
            )
            valid[batch_index, class_index] = True

    return torch.from_numpy(maps), torch.from_numpy(valid)


# 创新点：在最终融合概率上加入标签 signed distance map 的表面距离联合监督。时间：20261002
def surface_distance_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    distance_clip: float = 20.0,
    include_background: bool = False,
) -> torch.Tensor:
    """Return a differentiable surface loss for final logits.

    ``logits`` is ``[B, C, H, W]`` and ``target`` is ``[B, H, W]``.  The
    signed map is detached from autograd, while the softmax probabilities keep
    their gradient.  The mean is taken over valid sample/class pairs only.
    """
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, C, H, W]")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != logits.shape[0:1] + logits.shape[2:]:
        raise ValueError(
            "target shape {} does not match logits spatial shape {}".format(
                tuple(target.shape), (logits.shape[0], logits.shape[2], logits.shape[3])
            )
        )

    probabilities = torch.softmax(logits.float(), dim=1)
    distance_maps, valid = build_signed_distance_maps(
        target,
        num_classes=logits.shape[1],
        distance_clip=distance_clip,
        include_background=include_background,
    )
    distance_maps = distance_maps.to(device=logits.device, dtype=probabilities.dtype)
    valid = valid.to(device=logits.device)
    valid_count = valid.sum()
    if int(valid_count.item()) == 0:
        return probabilities.sum() * 0.0

    per_pair = (probabilities * distance_maps).sum(dim=(2, 3))
    return per_pair.masked_select(valid).mean()

"""Filesystem-only naming for training runs; model behavior is unchanged."""

from datetime import datetime
from pathlib import Path
import re
from uuid import uuid4


def _part(value):
    if isinstance(value, (list, tuple)):
        value = "-".join(map(str, value))
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._") or "na"


def experiment_subpath(args, run_name=None):
    """Encode the settings that identify a model and its training protocol."""
    settings = (
        ("img", getattr(args, "img_size", "na")),
        ("b", getattr(args, "batch_size", "na")),
        ("ep", getattr(args, "max_epochs", "na")),
        ("sup", getattr(args, "supervision", "na")),
        ("lr", getattr(args, "base_lr", "na")),
        ("ef", getattr(args, "expansion_factor", "na")),
        ("k", getattr(args, "kernel_sizes", "na")),
        ("g", getattr(args, "lgag_ks", "na")),
        ("act", getattr(args, "activation_mscb", "na")),
        ("dw", int(not getattr(args, "no_dw_parallel", False))),
        ("ag", "concat" if getattr(args, "concatenation", False) else "add"),
        ("pre", int(not getattr(args, "no_pretrain", False))),
        ("gray", int(args.grayscale) if hasattr(args, "grayscale") else "na"),
        ("merge", int(bool(args.merge_instance_masks)) if hasattr(args, "merge_instance_masks") else "na"),
        ("ms", int(not args.no_multi_scale) if hasattr(args, "no_multi_scale") else "na"),
        ("s", getattr(args, "seed", "na")),
    )
    tail = run_name or "{}_{}".format(
        datetime.now().strftime("%Y%m%d%H%M%S"), uuid4().hex[:12]
    )
    detail = "_".join(f"{key}{_part(value)}" for key, value in settings)
    return Path("encoder_" + _part(getattr(args, "encoder", "na")), detail, _part(tail))


def make_experiment_dir(args, family, output_root="./model_pth"):
    root = Path(output_root).expanduser()
    path = root / family / experiment_subpath(args)
    path.mkdir(parents=True, exist_ok=False)
    return str(path.resolve())

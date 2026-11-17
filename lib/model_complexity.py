"""Record full-network complexity using the EMCAD authors' THOP convention."""

import json
import logging
from pathlib import Path

import torch


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Profile one image once; report THOP's raw result as the paper's FLOPs."""
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("Complexity input must be a single NCHW image")

    modules = list(target.modules())
    before = [(module, module.training, set(module._buffers),
               set(module._forward_hooks)) for module in modules]
    direct_params = sum(parameter.numel() for parameter in target.parameters())
    record = {
        "method": "EMCAD author convention: THOP profile, raw output / 1e9",
        "scope": "full network, one forward pass, batch size 1",
        "input_shape": shape,
        "parameters": direct_params,
        "parameters_million": direct_params / 1e6,
        "thop_flops": None,
        "thop_gflops": None,
        "thop_parameters": None,
        "thop_parameter_gap": None,
        "note": "THOP omits CCSF einsum and LRCI functional conv2d/bmm; this value is not complete operator FLOPs.",
    }
    try:
        from thop import profile

        device = next(target.parameters()).device
        target.eval()
        with torch.no_grad():
            # A local generator prevents profiling from advancing training RNG.
            generator = torch.Generator(device=device).manual_seed(0)
            sample = torch.randn(shape, device=device, generator=generator)
            flops, thop_params = profile(target, inputs=(sample,), verbose=False)
        record["thop_flops"] = int(round(flops))
        record["thop_gflops"] = flops / 1e9
        record["thop_parameters"] = int(round(thop_params))
        record["thop_parameter_gap"] = direct_params - record["thop_parameters"]
    except Exception as exc:
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
    finally:
        # THOP registers temporary counters on modules. Remove only additions.
        for module, training, buffers, hooks in before:
            for name in set(module._buffers) - buffers:
                if name in ("total_ops", "total_params"):
                    delattr(module, name)
            for key in set(module._forward_hooks) - hooks:
                del module._forward_hooks[key]
            module.training = training

    logging.info("%s_parameters=%d (%.6f M, direct count)",
                 prefix, direct_params, direct_params / 1e6)
    if record["thop_gflops"] is None:
        logging.warning("%s_THOP_GFlops_unavailable=%s", prefix, record["error"])
    else:
        logging.info("%s_THOP_GFlops=%.6f (raw THOP / 1e9; input=%s)",
                     prefix, record["thop_gflops"], shape)
        logging.info("%s_THOP_parameters=%d", prefix, record["thop_parameters"])
        logging.info("%s_THOP_parameter_gap=%d (direct minus THOP)",
                     prefix, record["thop_parameter_gap"])
        if set(getattr(target, "_pair_active", ())) & {"ccsf", "lrci"}:
            logging.warning("%s_THOP_coverage_incomplete: functional CCSF/LRCI "
                            "operations are not fully counted", prefix)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record

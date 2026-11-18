"""Record model complexity with the project's original EMCAD CalParams method."""

from contextlib import redirect_stdout
from io import StringIO
import logging

import torch

from utils.utils import CalParams


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Run the original THOP single-image forward estimate once before training.

    CalParams defines the FLOPs/Params convention used in the original EMCAD
    project. A direct registered-parameter count is included as an audit field.
    THOP does not count OpenCV Canny or all functional tensor operations.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    parameter_count = sum(parameter.numel() for parameter in target.parameters())
    records = [
        "{}_registered_parameters={}".format(prefix, parameter_count),
        "{}_complexity_input_shape={}".format(prefix, tuple(input_shape)),
        "complexity_method=EMCAD CalParams: THOP profile + clever_format",
        "complexity_scope=single forward; excludes loss, backward, OpenCV Canny and unprofiled tensor ops",
    ]
    logging.info("%s_registered_parameters=%d", prefix, parameter_count)
    logging.info("%s_complexity_input_shape=%s", prefix, tuple(input_shape))

    modules = tuple(target.modules())
    original_training = {module: module.training for module in modules}
    original_buffers = {module: set(module._buffers) for module in modules}
    try:
        target.eval()
        parameter = next(target.parameters())
        dummy_input = torch.zeros(tuple(input_shape), device=parameter.device,
                                  dtype=parameter.dtype)
        capture = StringIO()
        with torch.no_grad(), redirect_stdout(capture):
            CalParams(target, dummy_input)
        summary = capture.getvalue().split("[Statistics Information]")[-1].strip()
        summary = "[Statistics Information]\n" + summary
        print(summary)
        logging.info("%s", summary)
        records.append(summary)
    except Exception as exc:
        logging.warning("%s_complexity_unavailable=%s", prefix, exc)
        records.append("{}_complexity_unavailable={}".format(prefix, exc))
    finally:
        # THOP registers total_ops/total_params as buffers on submodules.
        # Remove only counters introduced by this call, so later checkpoints
        # contain exactly the model's own weights and buffers.
        for module in modules:
            for name in ("total_ops", "total_params"):
                if name not in original_buffers[module]:
                    module._buffers.pop(name, None)
            module.training = original_training[module]
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(records) + "\n")

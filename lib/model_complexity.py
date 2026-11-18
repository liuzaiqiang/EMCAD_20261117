"""Record model complexity with the original EMCAD CalParams function."""

from contextlib import redirect_stdout
from io import StringIO
import logging

import torch

from utils.utils import CalParams


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Capture the original THOP/clever_format single-forward output.

    The label FLOPs follows CalParams exactly. THOP does not count every
    functional operation; the independent registered-parameter count is an
    audit field and is not substituted for CalParams' Params output.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    modules = tuple(target.modules())
    original_training = {module: module.training for module in modules}
    original_buffers = {module: set(module._buffers) for module in modules}
    parameter = next(target.parameters())
    registered_parameters = sum(p.numel() for p in target.parameters())
    records = [
        "{}_registered_parameters={}".format(prefix, registered_parameters),
        "{}_complexity_input_shape={}".format(prefix, tuple(input_shape)),
        "complexity_method=EMCAD CalParams: THOP profile + clever_format",
        "complexity_scope=single forward; excludes loss, backward and unprofiled functional ops",
    ]
    logging.info("%s_registered_parameters=%d", prefix, registered_parameters)
    logging.info("%s_complexity_input_shape=%s", prefix, tuple(input_shape))
    try:
        target.eval()
        dummy_input = torch.zeros(tuple(input_shape), device=parameter.device,
                                  dtype=parameter.dtype)
        captured = StringIO()
        with torch.no_grad(), redirect_stdout(captured):
            CalParams(target, dummy_input)
        if "[Statistics Information]" not in captured.getvalue():
            raise RuntimeError("CalParams did not print its statistics summary")
        summary = "[Statistics Information]" + captured.getvalue().split(
            "[Statistics Information]")[-1].rstrip()
        print(summary)
        logging.info("%s", summary)
        records.append(summary)
    except Exception as exc:
        logging.warning("%s_complexity_unavailable=%s", prefix, exc)
        records.append("{}_complexity_unavailable={}".format(prefix, exc))
    finally:
        # THOP temporarily registers these as buffers. A later checkpoint
        # must contain the model's own state, not profiler counters.
        for module in modules:
            for name in ("total_ops", "total_params"):
                if name not in original_buffers[module]:
                    module._buffers.pop(name, None)
            module.training = original_training[module]
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(records) + "\n")

"""Record complexity with the project's cal_params_flops convention."""

import contextlib
import io
import logging

import torch

from utils.utils import cal_params_flops


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Profile the active model with its actual input shape and save the report.

    ``cal_params_flops`` owns the THOP call, printed units and direct parameter
    count.  This wrapper preserves model mode and random state, and persists
    the report so profiling does not alter the subsequent training sequence.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    was_training = target.training
    shape = tuple(input_shape)
    lines = []
    device = next(target.parameters()).device
    rng_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                   if device.type == "cuda" else [])

    class _Logger:
        def info(self, message):
            logging.info("%s", message)
            lines.append(str(message))

    try:
        target.eval()
        output = io.StringIO()
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad(), contextlib.redirect_stdout(output):
            cal_params_flops(target, shape, _Logger())
        printed = output.getvalue().strip()
        if printed:
            print(printed)
    except Exception as exc:
        message = "{} complexity unavailable: {}".format(prefix, exc)
        logging.warning("%s", message)
        printed = message
    finally:
        target.train(was_training)

    report = "{}_complexity_input_shape={}\n{}\n{}\n".format(
        prefix, shape, printed, "\n".join(lines)
    )
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(report)

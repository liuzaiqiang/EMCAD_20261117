"""Record model complexity with the project's original THOP counting convention."""

import contextlib
import io
import logging

import torch


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Use cal_params_flops and restore train/eval mode and random streams."""
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    shape = tuple(input_shape)
    was_training = target.training
    lines = []
    logger = logging.getLogger()

    class Capture:
        def info(self, message):
            lines.append(message)
            logger.info(message)

    try:
        from utils.utils import cal_params_flops

        device = next(target.parameters()).device
        cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            target.eval()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cal_params_flops(target, shape, Capture())
            for line in output.getvalue().splitlines():
                print(line)
                lines.append(line)
        logger.info("%s_complexity_input_shape=%s", prefix, shape)
        lines.append("{}_complexity_input_shape={}".format(prefix, shape))
    except Exception as exc:
        logger.warning("%s_flops_unavailable=%s", prefix, exc)
        lines.append("{}_flops_unavailable={}".format(prefix, exc))
        total = sum(parameter.numel() for parameter in target.parameters())
        message = "Total params: %.2fM" % (total / 1e6)
        print(message)
        lines.append(message)
    finally:
        target.train(was_training)
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")

"""Record the author's original CalParams(model, input_tensor) report."""

import contextlib
import io
import logging

import torch

from utils.utils import CalParams


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    was_training = target.training
    device = next(target.parameters()).device
    input_tensor = torch.zeros(tuple(input_shape), device=device)
    output = io.StringIO()
    try:
        target.eval()
        with torch.no_grad(), contextlib.redirect_stdout(output):
            CalParams(target, input_tensor)
        report = output.getvalue().strip()
    except Exception as exc:
        report = "[Statistics Information]\nFLOPs: unavailable\nParams: unavailable\nReason: {}".format(exc)
        logging.warning("%s complexity unavailable: %s", prefix, exc)
    finally:
        target.train(was_training)
    record = "{}_complexity_input_shape={}\n{}\n".format(prefix, tuple(input_shape), report)
    logging.info("%s", record.strip())
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(record)

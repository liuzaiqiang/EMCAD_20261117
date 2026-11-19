"""Record model complexity using EMCAD's original CalParams convention."""

import contextlib
import io
import logging

import torch


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Call utils.utils.CalParams(model, input_tensor) with the author's THOP units."""
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    was_training = target.training
    device = next(target.parameters()).device
    input_tensor = torch.zeros(tuple(input_shape), device=device)
    try:
        from utils.utils import CalParams
        target.eval()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()) as capture:
            CalParams(target, input_tensor)
        report = capture.getvalue().strip()
    except Exception as exc:
        report = "[Statistics Information]\nFLOPs: unavailable\nParams: unavailable\nReason: {}".format(exc)
        logging.warning("CalParams failed: %s", exc)
    finally:
        target.train(was_training)
    logging.info("%s_complexity_input_shape=%s", prefix, tuple(input_shape))
    logging.info("%s", report)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write("input_shape={}\n{}\n".format(tuple(input_shape), report))

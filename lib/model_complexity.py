"""Record model complexity using the THOP convention in cal_params_flops."""

import logging

import torch


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Use THOP's raw FLOPs and parameter count, as in utils.cal_params_flops.

    The author's helper assumes a three-channel CUDA input. The training entry
    points also use one-channel models, so keep their actual shape and device.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    shape = tuple(input_shape)
    total_params = sum(parameter.numel() for parameter in target.parameters())
    records = [
        "{}_parameters={}".format(prefix, total_params),
        "{}_complexity_input_shape={}".format(prefix, shape),
    ]
    logging.info("%s_parameters=%d", prefix, total_params)
    logging.info("%s_complexity_input_shape=%s", prefix, shape)

    was_training = target.training
    try:
        from thop import profile

        target.eval()
        device = next(target.parameters()).device
        sample = torch.randn(shape, device=device)
        with torch.no_grad():
            flops, thop_params = profile(target, inputs=(sample,), verbose=False)

        # Match cal_params_flops: report profile() directly, without 2 * FLOPs.
        logging.info("%s_flops=%.0f", prefix, flops)
        logging.info("%s_thop_parameters=%.0f", prefix, thop_params)
        logging.info("flops: %s, params: %s, Total params: : %.4f",
                     flops / 1e9, thop_params / 1e6, total_params / 1e6)
        records.extend([
            "{}_flops={:.0f}".format(prefix, flops),
            "{}_thop_parameters={:.0f}".format(prefix, thop_params),
            "{}_flops_g={:.6f}".format(prefix, flops / 1e9),
            "{}_thop_parameters_m={:.6f}".format(prefix, thop_params / 1e6),
            "{}_parameters_m={:.6f}".format(prefix, total_params / 1e6),
        ])
    except Exception as exc:
        logging.warning("%s_flops_unavailable=%s", prefix, exc)
        records.append("{}_flops_unavailable={}".format(prefix, exc))
    finally:
        if was_training:
            target.train()
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(records) + "\n")

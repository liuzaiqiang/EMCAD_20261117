"""EMCAD-author-compatible THOP complexity record for the complete network."""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch import nn

from utils.utils import cal_params_flops


class _DecoderWithPDS(nn.Module):
    """Decoder path used by pair_runtime.forward, excluding prediction heads."""

    def __init__(self, decoder, pds):
        super().__init__()
        self.decoder = decoder
        self.pds = pds

    def forward(self, deep, skips):
        dec = self.decoder
        d = dec.cab4(deep) * deep
        d = dec.sab(d) * d
        d = dec.mscb4(d)
        features = [d]
        semantic = self.pds.start(d)
        for index, (stage, skip) in enumerate(zip((3, 2, 1), skips)):
            d = getattr(dec, "eucb" + str(stage))(d)
            gated = getattr(dec, "lgag" + str(stage))(g=d, x=skip)
            semantic, gated = self.pds.filter_skip(semantic, gated, index)
            d = d + gated
            d = getattr(dec, "cab" + str(stage))(d) * d
            d = dec.sab(d) * d
            d = getattr(dec, "mscb" + str(stage))(d)
            features.append(d)
            if index < 2:
                semantic = self.pds.update(semantic, d, index)
        return features


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """Replicate the original EMCAD utility's profile and direct Params count.

    The authors label THOP's first raw return value ``flops`` and divide it by
    1e9, without multiplying by two. The paper's binary whole-model table uses
    a single 256x256 image. THOP calls this value MACs; both names are saved.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    shape = tuple(int(x) for x in input_shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("EMCAD complexity input must be NCHW, batch size 1")

    parameters = list(target.parameters())
    active = tuple(getattr(target, "_pair_active", ()))
    record = {
        "protocol": "SLDGroup/EMCAD utils/utils.py cal_params_flops",
        "scope": "whole model, eval forward, all four output heads",
        "input_shape_nchw": shape,
        "torch_version": torch.__version__,
        "thop_version": None,
        "author_labeled_flops_g": None,
        "thop_raw_ops": None,
        "thop_macs_g": None,
        "thop_parameters": None,
        "thop_parameters_m": None,
        "direct_parameters": sum(p.numel() for p in parameters),
        "direct_parameters_m": sum(p.numel() for p in parameters) / 1e6,
        "direct_trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "decoder_224": None,
        "note": "Original EMCAD code labels raw THOP result FLOPs; THOP documents it as MACs. No x2 conversion.",
    }
    was_training = target.training
    rng_devices = ([parameters[0].device.index if parameters[0].device.index is not None
                    else torch.cuda.current_device()]
                   if parameters[0].device.type == "cuda" else [])
    try:
        from thop import profile
        try:
            record["thop_version"] = version("thop")
        except PackageNotFoundError:
            record["thop_version"] = "unavailable"
        target.eval()
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            raw_ops, thop_params, direct_params = cal_params_flops(target, shape, logging)
            record.update(
                author_labeled_flops_g=raw_ops / 1e9,
                thop_raw_ops=raw_ops,
                thop_macs_g=raw_ops / 1e9,
                thop_parameters=thop_params,
                thop_parameters_m=thop_params / 1e6,
                direct_parameters=direct_params,
                direct_parameters_m=direct_params / 1e6,
            )
            # Paper Table 6 measures the decoder alone at 224x224, not the network.
            # PDS lives outside model.decoder, so include it explicitly when enabled.
            decoder = target.decoder
            if "pds" in active:
                decoder = _DecoderWithPDS(target.decoder, target.pair_modules["pds"])
            channels = [getattr(target, "out_head" + str(stage)).in_channels
                        for stage in (4, 3, 2, 1)]
            features = [torch.randn((1, channel, side, side),
                                    device=parameters[0].device, dtype=parameters[0].dtype)
                        for channel, side in zip(channels, (7, 14, 28, 56))]
            decoder_ops, decoder_thop_params = profile(
                decoder, inputs=(features[0], features[1:]), verbose=False)
        record["decoder_224"] = {
            "scope": "decoder including PDS if enabled; no backbone or prediction heads",
            "author_labeled_flops_g": decoder_ops / 1e9,
            "thop_raw_ops": decoder_ops,
            "thop_parameters_m": decoder_thop_params / 1e6,
            "direct_parameters_m": sum(p.numel() for p in decoder.parameters()) / 1e6,
        }
    except Exception as exc:
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
    finally:
        target.train(was_training)

    logging.info("%s_complexity=%s", prefix, json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["author_labeled_flops_g"] is None:
        print("[EMCAD complexity] THOP unavailable; Direct Params={:.6f}M; error={}".format(
            record["direct_parameters_m"], record.get("error", "unknown"),
        ))
    if record["decoder_224"] is not None:
        print("[EMCAD decoder 224] THOP raw FLOPs(label)/MACs={:.6f}G Direct Params={:.6f}M".format(
            record["decoder_224"]["author_labeled_flops_g"],
            record["decoder_224"]["direct_parameters_m"],
        ))
    if "error" in record:
        logging.warning("%s THOP profiling incomplete: %s", prefix, record["error"])
    if output_path is not None:
        Path(output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record

"""Audited convolution/matrix MACs, including functional Sobel and attention.

One MAC = one multiplication plus accumulation; FLOPs_2MAC = 2*MACs.
This is explicitly NOT an all-scalar-instructions count. Non-MAC operations
(interpolation, normalization, activations, reductions, etc.) are inventoried.
Unknown operators invalidate the report rather than silently counting zero.
"""
import copy
import json
import logging
from collections import Counter
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Explicitly excluded from the conv/matrix MAC convention, never implicit zero.
NON_MAC = set('add add_ sub mul div pow square sqrt rsqrt abs tanh sigmoid silu relu relu_ gelu clamp clamp_min mean sum amax max _softmax native_layer_norm native_batch_norm _native_batch_norm_legit_no_training adaptive_avg_pool2d _adaptive_avg_pool2d avg_pool2d adaptive_max_pool2d max_pool2d_with_indices upsample_nearest2d upsample_bilinear2d _upsample_bilinear2d_aa'.split())
MOVEMENT = set('view _unsafe_view reshape transpose permute t expand squeeze unsqueeze slice select split split_with_sizes unbind cat stack clone contiguous detach alias _to_copy copy_ lift_fresh empty empty_like empty_strided zeros zeros_like ones ones_like full arange scalar_tensor fill_ constant_pad_nd replication_pad2d reflection_pad2d resolve_conj resolve_neg'.split())

class ArithmeticAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.macs = Counter()
        self.calls = Counter()
        self.excluded = Counter()
        self.unknown = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name.split('::')[-1]
        self.calls[name] += 1
        if name in ('convolution', '_convolution'):
            x, w = args[:2]
            if args[6]:
                raise RuntimeError('Transposed convolution requires an explicit MAC formula')
            self.macs[name] += out.numel() * w[0].numel()
        elif name in ('mm', 'bmm', 'addmm', 'baddbmm'):
            a, b = args[1:3] if name in ('addmm', 'baddbmm') else args[:2]
            self.macs[name] += out.numel() * a.shape[-1]
        elif name in NON_MAC:
            self.excluded[name] += 1
        elif name not in MOVEMENT:
            self.unknown[name] += 1
        return out


def measure_model_complexity(model, input_shape):
    """Profile an isolated CPU copy in eval mode; never mutate the live model.

    Counts the actually executed full four-head inference forward. Returns exact
    parameter elements (including allocated but unused parameters), trainable
    parameter elements, per-operator MACs and an exhaustive operator inventory.
    Batch totals and per-image totals are both explicit.
    """
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    parameters = sum(p.numel() for p in target.parameters())
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    probe = copy.deepcopy(target).cpu().float().eval()
    shape = tuple(int(x) for x in input_shape)
    if len(shape) != 4 or min(shape) <= 0:
        raise ValueError('input_shape must be positive N,C,H,W')
    audit = ArithmeticAudit()
    with torch.no_grad(), audit:
        probe(torch.zeros(shape))
    if audit.unknown:
        raise RuntimeError('Unclassified operators; no publishable MAC total: ' + str(dict(audit.unknown)))
    macs = sum(audit.macs.values())
    return dict(parameters=parameters, trainable_parameters=trainable,
                input_shape=shape, active_modules=list(getattr(target, '_pair_active', ())),
                macs=macs, flops=2*macs, macs_per_image=macs/shape[0],
                flops_per_image=2*macs/shape[0],
                convention='conv_matrix_MACs; 1MAC=2FLOPs; excludes non-MAC scalar ops',
                scope='eval full four-head forward, no loss/backward/optimizer',
                macs_by_operator=dict(audit.macs), excluded_non_mac_calls=dict(audit.excluded),
                operator_calls=dict(audit.calls), unknown_operators={}, torch_version=torch.__version__)


def log_model_complexity(model, input_shape, prefix='model', output_path=None):
    # Fail closed: never emit a plausible incomplete number after an audit error.
    report = measure_model_complexity(model, input_shape)
    for key in ('parameters', 'trainable_parameters', 'macs', 'flops', 'convention'):
        logging.info('%s_%s=%s', prefix, key, report[key])
    logging.info('%s_complexity_audit=%s', prefix, json.dumps(report))
    if output_path is not None:
        path = Path(output_path)
        path.write_text('\n'.join(f'{prefix}_{k}={report[k]}' for k in ('parameters','trainable_parameters','macs','flops','convention'))+'\n', encoding='utf-8')
        Path(str(path)+'.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

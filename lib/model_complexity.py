"""Audited convolution/matrix MACs, including functional Sobel and attention.

One MAC = one multiplication plus accumulation; FLOPs_2MAC = 2*MACs.
This is explicitly NOT an all-scalar-instructions count. Non-MAC operations
(interpolation, normalization, activations, reductions, etc.) are inventoried.
Unknown operators invalidate the report rather than silently counting zero.
"""
import copy
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
import torch
try:
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:
    class TorchDispatchMode:
        def __enter__(self):
            raise RuntimeError("Complexity audit requires PyTorch with TorchDispatchMode (tested: 2.5.1)")
        def __exit__(self, *args):
            pass

# Explicitly excluded from the conv/matrix MAC convention, never implicit zero.
NON_MAC = set('add add_ sub mul div pow square sqrt rsqrt abs tanh sigmoid silu relu relu_ hardtanh hardtanh_ gelu clamp clamp_min mean sum amax max _softmax native_layer_norm native_batch_norm _native_batch_norm_legit_no_training adaptive_avg_pool2d _adaptive_avg_pool2d avg_pool2d adaptive_max_pool2d max_pool2d_with_indices upsample_nearest2d upsample_bilinear2d _upsample_bilinear2d_aa'.split())
MOVEMENT = set('view _unsafe_view reshape transpose permute t expand squeeze unsqueeze slice select split split_with_sizes unbind cat stack clone contiguous detach alias _to_copy copy_ lift_fresh empty empty_like empty_strided zeros zeros_like ones ones_like full arange scalar_tensor fill_ constant_pad_nd replication_pad2d reflection_pad2d resolve_conj resolve_neg'.split())

def scalar_ops(name, args, out):
    """Supplemental algorithmic estimate, NOT hardware instruction FLOPs.

    Unary transcendental/comparison = 1 op. Bilinear = 4 multiplies+3 adds.
    AA bilinear uses clipped separable triangle support; coefficient generation
    and coordinate arithmetic are excluded. Fused norm/activation costs are
    declared estimates, never included in the comparable conv/matmul MAC total.
    """
    y = out[0] if isinstance(out, (tuple,list)) else out
    n = y.numel()
    x = args[0]
    if name in ('mean','sum','amax','max'):
        return max(x.numel()-n,0) + (n if name=='mean' else 0)
    if name == '_softmax':
        return 5*x.numel()  # stable max/subtract/exp/sum/div convention
    if name == 'native_layer_norm':
        return 8*x.numel()  # two-pass variance, affine, scalar roots estimated
    if 'batch_norm' in name:
        return 4*x.numel()  # unfused inference normalization plus affine
    if name in ('hardtanh', 'hardtanh_'): return 2*n  # lower and upper clamp comparisons
    if name == 'gelu': return 8*n  # erf formulation, erf counts as one
    if name in ('sigmoid','silu'): return (4 if name=='sigmoid' else 5)*n
    if name == 'upsample_nearest2d': return 0
    if name == 'upsample_bilinear2d': return 7*n
    if name == '_upsample_bilinear2d_aa':
        ih,iw=x.shape[-2:];oh,ow=y.shape[-2:]
        def taps(src,dst):
            scale=src/dst; support=max(scale,1.)
            return sum(max(0,min(src,int((i+.5)*scale+support+.5))-
                           max(0,int((i+.5)*scale-support+.5))) for i in range(dst))
        # Horizontal pass then vertical pass, matching separable filtering.
        hcost=ih*(2*taps(iw,ow)-ow) if iw!=ow else 0
        vcost=ow*(2*taps(ih,oh)-oh) if ih!=oh else 0
        return x.shape[0]*x.shape[1]*(hcost+vcost)
    if name in ('adaptive_avg_pool2d','_adaptive_avg_pool2d','adaptive_max_pool2d'):
        ih,iw=x.shape[-2:];oh,ow=y.shape[-2:]
        samples=sum(math.ceil((i+1)*ih/oh)-math.floor(i*ih/oh) for i in range(oh))*sum(math.ceil((i+1)*iw/ow)-math.floor(i*iw/ow) for i in range(ow))
        return x.shape[0]*x.shape[1]*samples - (n if name=='adaptive_max_pool2d' else 0)
    if name in ('avg_pool2d','max_pool2d_with_indices'):
        k=args[1]; area=math.prod(k) if isinstance(k,(tuple,list)) else k*k
        return n*(area if name=='avg_pool2d' else area-1)
    return n


class ArithmeticAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.macs = Counter()
        self.calls = Counter()
        self.excluded = Counter()
        self.unknown = Counter()
        self.scalar_estimates = Counter()

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
            if args[2] is not None:
                self.scalar_estimates['convolution_bias_add'] += out.numel()
        elif name in ('mm', 'bmm', 'addmm', 'baddbmm'):
            a, b = args[1:3] if name in ('addmm', 'baddbmm') else args[:2]
            self.macs[name] += out.numel() * a.shape[-1]
            if name in ('addmm','baddbmm'):
                self.scalar_estimates['matrix_bias_add'] += out.numel()
        elif name in NON_MAC:
            self.excluded[name] += 1
            self.scalar_estimates[name] += scalar_ops(name, args, out)
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
    # Memoize CPU copies before deepcopy: never duplicate parameter storage on GPU.
    memo = {}
    for tensor in list(target.parameters()) + list(target.buffers()):
        if id(tensor) not in memo:
            value = tensor.detach().to(device='cpu', copy=True)
            if isinstance(tensor, torch.nn.Parameter):
                value = torch.nn.Parameter(value, requires_grad=tensor.requires_grad)
            memo[id(tensor)] = value
    probe = copy.deepcopy(target, memo).float().eval()
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
                status='ok', macs=macs, flops=2*macs, macs_per_image=macs/shape[0],
                supplemental_scalar_ops_estimate=sum(audit.scalar_estimates.values()),
                supplemental_scalar_ops_by_operator=dict(audit.scalar_estimates),
                expanded_ops_estimate=2*macs+sum(audit.scalar_estimates.values()),
                flops_per_image=2*macs/shape[0],
                convention='conv_matrix_MACs; 1MAC=2FLOPs; excludes non-MAC scalar ops',
                scope='eval full four-head forward, no loss/backward/optimizer',
                macs_by_operator=dict(audit.macs), excluded_non_mac_calls=dict(audit.excluded),
                operator_calls=dict(audit.calls), unknown_operators={}, torch_version=torch.__version__)


def log_model_complexity(model, input_shape, prefix='model', output_path=None):
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    try:
        report = measure_model_complexity(model, input_shape)
    except Exception as exc:
        report = dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                      parameters=sum(p.numel() for p in target.parameters()),
                      input_shape=list(input_shape), macs=None, flops=None)
        logging.exception('%s_complexity_failed; do not use this run for complexity reporting', prefix)
        if os.getenv('COMPLEXITY_STRICT', '0') == '1':
            raise
    logging.info('%s_complexity_audit=%s', prefix, json.dumps(report))
    for key in ('parameters', 'trainable_parameters', 'macs', 'flops', 'convention'):
        if key in report:
            logging.info('%s_%s=%s', prefix, key, report[key])
    if output_path is not None:
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(f'{prefix}_{k}={v}' for k,v in report.items()
                                      if not isinstance(v, dict))+'\n', encoding='utf-8')
            Path(str(path)+'.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        except OSError:
            logging.exception('%s_complexity_report_write_failed', prefix)
    return report

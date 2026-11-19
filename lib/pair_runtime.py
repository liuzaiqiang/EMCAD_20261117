"""Pair integration, keeping 00 on the original baseline path.

Module algorithms are copied from the commits in docs/pair_sources.json.
This file defines only composition, switch validation and dataset adapters.
"""
import os
import json
import torch
from torch import nn
from torch.nn import functional as F
from lib.pair_config import FULL_RESOLUTION_DETAIL, SURFACE_DISTANCE_SUPERVISION


def settings():
    values = tuple(os.environ.get(k, '0') for k in (
        'FULL_RESOLUTION_DETAIL_ENABLED', 'SURFACE_DISTANCE_SUPERVISION_ENABLED'))
    if any(v not in ('0', '1') for v in values):
        raise ValueError('FULL_RESOLUTION_DETAIL_ENABLED and '
                         'SURFACE_DISTANCE_SUPERVISION_ENABLED must be 0 or 1')
    return tuple(m for m, v in zip((FULL_RESOLUTION_DETAIL, SURFACE_DISTANCE_SUPERVISION), values)
                 if v == '1')


def initialize(model, channels):
    active = settings()
    model._pair_active = active
    if not active:
        return
    modules = nn.ModuleDict()
    k = model.out_head1.out_channels
    if 'caa' in active:
        from lib.pair_modules.caa import ContentAwareAntiAliasUpsample
        # Preserve EUCB depthwise/BN/activation/projection; replace upsampler only.
        for stage, c in zip((3, 2, 1), channels[:-1]):
            getattr(model.decoder, 'eucb' + str(stage)).up_dwc[0] = ContentAwareAntiAliasUpsample(c)
    if 'pds' in active:
        from lib.pair_modules.pds import ProgressiveDeepSemantic
        modules['pds'] = ProgressiveDeepSemantic(channels)
    if 'lrci' in active:
        from lib.pair_modules.lrci import LowRankCovarianceInteraction
        modules['lrci'] = nn.ModuleList([LowRankCovarianceInteraction(c) for c in channels[:2]])
    if 'mcc' in active:
        from lib.pair_modules.mcc import MultiScaleContourConsistency
        modules['mcc'] = MultiScaleContourConsistency(channels[1:])
    if 'ccsf' in active:
        from lib.pair_modules.ccsf import ClassConditionalScaleFusion
        modules['ccsf'] = ClassConditionalScaleFusion(channels[1:], k)
    if 'frd' in active:
        from lib.pair_modules.frd import FullResolutionDetailBranch
        modules['frd'] = FullResolutionDetailBranch(k)
    if 'sgr' in active:
        from lib.pair_modules.sgr import SemanticGatedResidual
        modules['sgr'] = SemanticGatedResidual(channels[-1], k)
    if 'prf' in active:
        from lib.pair_modules.prf import PixelReliabilityFusion
        modules['prf'] = PixelReliabilityFusion(k)
    model.pair_modules = modules
    print('PAIR_CONFIG ' + json.dumps({'pair': [FULL_RESOLUTION_DETAIL, SURFACE_DISTANCE_SUPERVISION],
          'full_resolution_detail_enabled': 'frd' in active,
          'surface_distance_supervision_enabled': 'sdg' in active,
          'active': active,
          'aat_weight': float(os.getenv('AAT_WEIGHT', '1.0')),
          'bfc_weight': float(os.getenv('BFC_WEIGHT', '0.05')),
          'sdg_weight': float(os.getenv('SDG_WEIGHT', '0.001'))}))


def forward(model, image, mode='test', return_pair_aux=False):
    active, modules, dec = model._pair_active, model.pair_modules, model.decoder
    x = model.conv(image) if image.shape[1] == 1 else image
    x1, x2, x3, x4 = model.backbone(x)
    d = dec.cab4(x4) * x4
    d = dec.sab(d) * d
    d = dec.mscb4(d)
    if 'lrci' in active:
        d = modules['lrci'][0](d)
    features = [d]
    if 'pds' in active:
        semantic = modules['pds'].start(d)
    consistency = []
    for i, (stage, skip) in enumerate(zip((3, 2, 1), (x3, x2, x1))):
        d = getattr(dec, 'eucb' + str(stage))(d)
        if 'bfc' in active and return_pair_aux:
            from lib.pair_modules.bfc import bidirectional_feature_consistency
            # Source's BFC-only mode: no learned skip alignment is enabled.
            consistency.append(bidirectional_feature_consistency(d, skip, skip, d))
        s = getattr(dec, 'lgag' + str(stage))(g=d, x=skip)
        if 'pds' in active:
            semantic, s = modules['pds'].filter_skip(semantic, s, i)
        d = d + s
        d = getattr(dec, 'cab' + str(stage))(d) * d
        d = dec.sab(d) * d
        d = getattr(dec, 'mscb' + str(stage))(d)
        if 'lrci' in active and i == 0:
            d = modules['lrci'][1](d)
        features.append(d)
        if 'pds' in active and i < 2:
            semantic = modules['pds'].update(semantic, d, i)
    if 'mcc' in active:
        features[1:] = modules['mcc'](features[1:])
    logits = [getattr(model, 'out_head' + str(s))(f) for s, f in zip((4, 3, 2, 1), features)]
    if 'ccsf' in active:
        logits[-1] = modules['ccsf'](features[1:], logits[1:])
    outputs = [F.interpolate(p, scale_factor=s, mode='bilinear') for p, s in zip(logits, (32, 16, 8, 4))]
    if 'frd' in active:
        outputs[-1] = outputs[-1] + modules['frd'](image)
    if 'sgr' in active:
        outputs[-1] = outputs[-1] + modules['sgr'](image, features[-1])
    aux = {}
    if consistency:
        aux['bfc'] = torch.stack(consistency).mean().reshape(1)
    if 'prf' in active:
        scores = modules['prf'].scores(outputs)
        weights = scores.softmax(dim=1)
        fused = sum(weights[:, i:i+1] * p for i, p in enumerate(outputs))
        if return_pair_aux:
            aux.update(scores=scores, fused=fused)
        else:
            # All existing dataset inference paths consume the last output.
            outputs = outputs[:-1] + [fused]
    return (outputs, aux) if return_pair_aux else outputs


def train_forward(model, images, mode='train'):
    raw = model.module if hasattr(model, 'module') else model
    if not getattr(raw, '_pair_active', ()):
        return model(images, mode=mode), {}
    return model(images, mode=mode, return_pair_aux=True)


def add_loss(loss, model, outputs, aux, target, segmentation_loss):
    raw = model.module if hasattr(model, 'module') else model
    active = getattr(raw, '_pair_active', ())
    if not active:
        return loss
    truth = target[:, 0] if target.ndim == 4 else target
    if 'prf' in active:
        with torch.no_grad():
            if outputs[0].shape[1] == 1:
                correct = torch.cat([((p.detach().sigmoid() >= .5) == (truth[:, None] >= .5)).float() for p in outputs], 1)
            else:
                correct = torch.stack([(p.detach().argmax(1) == truth.long()).float() for p in outputs], 1)
        loss = loss + segmentation_loss(aux['fused'], target) + F.binary_cross_entropy_with_logits(aux['scores'].float(), correct)
    if 'aat' in active:
        from lib.pair_modules.aat import area_adaptive_tversky_loss
        # Source applies AAT to p1, even when PRF additionally supervises fused logits.
        loss = loss + float(os.getenv('AAT_WEIGHT', '1.0')) * area_adaptive_tversky_loss(outputs[-1], truth)
    if 'bfc' in active:
        loss = loss + float(os.getenv('BFC_WEIGHT', '0.05')) * aux['bfc'].mean()
    if 'sdg' in active:
        from lib.pair_modules.sdg import surface_distance_loss
        final = aux.get('fused', outputs[-1])
        # Exact binary adapter: softmax([0,z])_foreground == sigmoid(z).
        if final.shape[1] == 1:
            final = torch.cat((torch.zeros_like(final), final), dim=1)
        loss = loss + float(os.getenv('SDG_WEIGHT', '0.001')) * surface_distance_loss(final, truth)
    return loss

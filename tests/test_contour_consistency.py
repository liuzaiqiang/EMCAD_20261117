"""Ablation wiring and feature-level checks for the reused contour module."""

import pytest
import torch

from lib.contour_consistency import MultiScaleContourConsistency
from lib.networks import EMCADNet


@pytest.mark.parametrize("fusion,contour", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_four_ablation_architectures_have_independent_parameters(fusion, contour):
    model = EMCADNet(encoder="pvt_v2_b0", pretrain=False, num_classes=1,
                     use_pixel_reliability_fusion=fusion,
                     use_contour_consistency=contour)
    keys = model.state_dict()
    assert any("pixel_reliability_fusion" in key for key in keys) == bool(fusion)
    assert any("contour_consistency" in key for key in keys) == bool(contour)
    assert (model.decoder.contour_consistency is not None) == bool(contour)


def test_contour_identity_start_and_gradient_flow_on_three_scales():
    module = MultiScaleContourConsistency([16, 8, 4])
    features = [torch.randn(2, 16, 8, 8, requires_grad=True),
                torch.randn(2, 8, 16, 16, requires_grad=True),
                torch.randn(2, 4, 32, 32, requires_grad=True)]
    outputs = module(features)
    assert all(torch.equal(a, b) for a, b in zip(features, outputs))
    sum(x.square().mean() for x in outputs).backward()
    assert torch.isfinite(module.strength.grad).all()
    assert module.strength.grad.abs().sum() > 0
    assert all(x.grad is not None for x in features)

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.strength.fill_(0.2)
    sum(x.square().mean() for x in module(features)).backward()
    assert all(branch[-1].weight.grad is not None for branch in module.feedback)


def test_contour_detects_step_and_handles_constant_features():
    module = MultiScaleContourConsistency([16, 8, 4])
    flat = torch.ones(1, 8, 17, 19)
    step = flat.clone()
    step[:, :, :, 10:] = 2
    assert module.contour(flat, (17, 19)).max() < 1e-4
    assert module.contour(step, (17, 19)).max() > 0.1

"""Run with pytest in the project environment containing PyTorch and timm."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lib.decoders import EMCAD
from lib.small_target_scale_enhancement import SmallTargetScaleEnhancement


def test_odd_shape_identity_at_initialization_and_gradients():
    module = SmallTargetScaleEnhancement(16)
    x = torch.randn(2, 16, 15, 17, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, x)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.gain.grad is not None and torch.isfinite(module.gain.grad).all()


def test_cpu_autocast():
    module = SmallTargetScaleEnhancement(8)
    x = torch.randn(1, 8, 15, 17, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x)
        y.mean().backward()
    assert y.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("caa_mode", ["off", "caa"])
@pytest.mark.parametrize("small_target", [False, True])
def test_decoder_four_ablations_and_strict_checkpoint(caa_mode, small_target):
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2,
                  caa_mode=caa_mode, small_target_scale_enhancement=small_target)
    model = EMCAD(**kwargs)
    clone = EMCAD(**kwargs)
    clone.load_state_dict(model.state_dict(), strict=True)
    if not small_target:
        assert not any("small_target" in key for key in model.state_dict())
    x = torch.randn(2, 64, 4, 4, requires_grad=True)
    skips = [torch.randn(2, 32, 8, 8),
             torch.randn(2, 32, 16, 16),
             torch.randn(2, 32, 32, 32)]
    outputs = model(x, skips)
    assert [out.shape[-2:] for out in outputs] == [(4, 4), (8, 8), (16, 16), (32, 32)]
    sum(out.square().mean() for out in outputs).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(param.grad is None or torch.isfinite(param.grad).all()
               for param in model.parameters())


def test_switch_mismatch_rejected_by_strict_loading():
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2, caa_mode="off")
    baseline = EMCAD(**kwargs, small_target_scale_enhancement=False)
    enhanced = EMCAD(**kwargs, small_target_scale_enhancement=True)
    with pytest.raises(RuntimeError):
        enhanced.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(enhanced.state_dict(), strict=True)

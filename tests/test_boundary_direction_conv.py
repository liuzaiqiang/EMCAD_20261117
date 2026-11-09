"""Run with pytest on an environment that has PyTorch and timm installed."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lib.boundary_direction_conv import BoundaryDirectionConv
from lib.decoders import EMCAD


def test_directional_skip_gradient_and_odd_shape():
    module = BoundaryDirectionConv(16)
    x = torch.randn(2, 16, 15, 17, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, x)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.gain.grad is not None and torch.isfinite(module.gain.grad).all()


def test_directional_skip_cpu_autocast():
    module = BoundaryDirectionConv(8)
    x = torch.randn(1, 8, 15, 17, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x)
        y.mean().backward()
    assert y.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("caa_mode", ["off", "caa"])
@pytest.mark.parametrize("boundary", [False, True])
def test_decoder_ablation_strict_checkpoint_and_backward(caa_mode, boundary):
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2,
                  caa_mode=caa_mode, boundary_direction_conv=boundary)
    model = EMCAD(**kwargs)
    clone = EMCAD(**kwargs)
    clone.load_state_dict(model.state_dict(), strict=True)
    if not boundary:
        assert not any("boundary_skip" in name for name in model.state_dict())
    x = torch.randn(2, 64, 4, 4, requires_grad=True)
    skips = [torch.randn(2, 32, 8, 8),
             torch.randn(2, 32, 16, 16),
             torch.randn(2, 32, 32, 32)]
    outputs = model(x, skips)
    assert [item.shape[-2:] for item in outputs] == [(4, 4), (8, 8), (16, 16), (32, 32)]
    sum(item.square().mean() for item in outputs).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(param.grad is None or torch.isfinite(param.grad).all()
               for param in model.parameters())


def test_architecture_mismatch_is_not_silent():
    base = EMCAD(channels=[64, 32, 32, 32], expansion_factor=2,
                 caa_mode="off", boundary_direction_conv=False)
    enhanced = EMCAD(channels=[64, 32, 32, 32], expansion_factor=2,
                     caa_mode="off", boundary_direction_conv=True)
    with pytest.raises(RuntimeError):
        enhanced.load_state_dict(base.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        base.load_state_dict(enhanced.state_dict(), strict=True)

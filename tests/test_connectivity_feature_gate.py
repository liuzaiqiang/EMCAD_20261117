"""Run with pytest in an environment containing PyTorch and timm."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lib.connectivity_feature_gate import ConnectivityFeatureGate
from lib.decoders import EMCAD


def test_odd_shape_identity_and_trainable_affinity():
    gate = ConnectivityFeatureGate(16)
    decoder = torch.randn(2, 16, 15, 17, requires_grad=True)
    skip = torch.randn(2, 16, 15, 17, requires_grad=True)
    output = gate(decoder, skip)
    assert output.shape == skip.shape
    torch.testing.assert_close(output, skip)
    output.square().mean().backward()
    assert gate.gain.grad is not None and torch.isfinite(gate.gain.grad).all()
    with torch.no_grad():
        gate.gain.fill_(0.1)
    gate.zero_grad(set_to_none=True)
    decoder.grad = None
    skip.grad = None
    gate(decoder, skip).square().mean().backward()
    assert decoder.grad is not None and torch.isfinite(decoder.grad).all()
    assert skip.grad is not None and torch.isfinite(skip.grad).all()
    assert gate.decoder_guide.weight.grad is not None
    assert gate.skip_guide.weight.grad is not None


def test_cpu_autocast():
    gate = ConnectivityFeatureGate(8)
    decoder = torch.randn(1, 8, 15, 17)
    skip = torch.randn(1, 8, 15, 17, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = gate(decoder, skip)
        output.mean().backward()
    assert output.shape == skip.shape
    assert skip.grad is not None and torch.isfinite(skip.grad).all()


@pytest.mark.parametrize("caa_mode", ["off", "caa"])
@pytest.mark.parametrize("connectivity", [False, True])
def test_four_ablations_forward_backward_and_strict_load(caa_mode, connectivity):
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2,
                  caa_mode=caa_mode, connectivity_feature_gate=connectivity)
    model = EMCAD(**kwargs)
    clone = EMCAD(**kwargs)
    clone.load_state_dict(model.state_dict(), strict=True)
    if not connectivity:
        assert not any("connectivity_skip" in key for key in model.state_dict())
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


def test_strict_load_rejects_wrong_switch():
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2, caa_mode="off")
    baseline = EMCAD(**kwargs, connectivity_feature_gate=False)
    enhanced = EMCAD(**kwargs, connectivity_feature_gate=True)
    with pytest.raises(RuntimeError):
        enhanced.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(enhanced.state_dict(), strict=True)

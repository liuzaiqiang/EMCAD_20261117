"""Run with pytest in a project environment with PyTorch and timm."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lib.decoders import EMCAD
from lib.structure_texture_skip import StructureTextureSkipFusion


def test_skip_fusion_odd_shape_identity_and_gradients():
    fusion = StructureTextureSkipFusion(16)
    decoder = torch.randn(2, 16, 15, 17, requires_grad=True)
    skip = torch.randn(2, 16, 15, 17, requires_grad=True)
    output = fusion(decoder, skip)
    assert output.shape == skip.shape
    torch.testing.assert_close(output, skip)
    output.square().mean().backward()
    assert decoder.grad is None or torch.isfinite(decoder.grad).all()
    assert skip.grad is not None and torch.isfinite(skip.grad).all()
    assert fusion.structure_gain.grad is not None
    assert fusion.texture_gain.grad is not None
    fusion.zero_grad(set_to_none=True)
    decoder.grad = None
    skip.grad = None
    with torch.no_grad():
        fusion.structure_gain.fill_(0.1)
        fusion.texture_gain.fill_(0.1)
    fusion(decoder, skip).square().mean().backward()
    assert decoder.grad is not None and torch.isfinite(decoder.grad).all()
    assert fusion.structure_filter.weight.grad is not None
    assert fusion.texture_filter.weight.grad is not None


def test_cpu_autocast():
    fusion = StructureTextureSkipFusion(8)
    decoder = torch.randn(1, 8, 15, 17)
    skip = torch.randn(1, 8, 15, 17, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = fusion(decoder, skip)
        output.mean().backward()
    assert output.shape == skip.shape
    assert skip.grad is not None and torch.isfinite(skip.grad).all()


@pytest.mark.parametrize("caa_mode", ["off", "caa"])
@pytest.mark.parametrize("skip_fusion", [False, True])
def test_four_ablations_forward_backward_and_checkpoint(caa_mode, skip_fusion):
    kwargs = dict(channels=[64, 32, 32, 32], expansion_factor=2,
                  caa_mode=caa_mode, structure_texture_skip=skip_fusion)
    model = EMCAD(**kwargs)
    copy = EMCAD(**kwargs)
    copy.load_state_dict(model.state_dict(), strict=True)
    if not skip_fusion:
        assert not any("skip_fusion" in key for key in model.state_dict())
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
    baseline = EMCAD(**kwargs, structure_texture_skip=False)
    enhanced = EMCAD(**kwargs, structure_texture_skip=True)
    with pytest.raises(RuntimeError):
        enhanced.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(enhanced.state_dict(), strict=True)

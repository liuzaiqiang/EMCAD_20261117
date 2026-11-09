"""Run with pytest in the project environment with PyTorch and timm."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lib.class_aware_prototype import ClassAwareCrossScalePrototype
from lib.networks import EMCADNet


@pytest.mark.parametrize("num_classes", [1, 4, 9])
def test_prototypes_shape_and_gradients(num_classes):
    module = ClassAwareCrossScalePrototype(16, 8, 4, num_classes, embedding_dim=8)
    d3 = torch.randn(2, 16, 5, 7, requires_grad=True)
    d2 = torch.randn(2, 8, 9, 13, requires_grad=True)
    d1 = torch.randn(2, 4, 17, 25, requires_grad=True)
    logits3 = torch.randn(2, num_classes, 5, 7, requires_grad=True)
    logits2 = torch.randn(2, num_classes, 9, 13, requires_grad=True)
    output = module(d3, d2, d1, logits3, logits2)
    assert output.shape == d1.shape
    torch.testing.assert_close(output, d1)
    output.square().mean().backward()
    assert module.gain.grad is not None and torch.isfinite(module.gain.grad).all()

    with torch.no_grad():
        module.gain.fill_(0.1)
    module.zero_grad(set_to_none=True)
    module(d3, d2, d1, logits3, logits2).square().mean().backward()
    assert module.embed3.weight.grad is not None
    assert module.embed2.weight.grad is not None
    assert module.query1.weight.grad is not None
    assert torch.isfinite(module.scale_logits.grad).all()


def test_cpu_autocast():
    module = ClassAwareCrossScalePrototype(16, 8, 4, 1, embedding_dim=8)
    features = (torch.randn(1, 16, 5, 7), torch.randn(1, 8, 9, 13),
                torch.randn(1, 4, 17, 25, requires_grad=True))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(*features, torch.randn(1, 1, 5, 7),
                        torch.randn(1, 1, 9, 13))
        output.mean().backward()
    assert output.shape == features[2].shape
    assert features[2].grad is not None and torch.isfinite(features[2].grad).all()


@pytest.mark.parametrize("num_classes", [1, 4])
def test_nearly_absent_class_is_finite(num_classes):
    module = ClassAwareCrossScalePrototype(16, 8, 4, num_classes, embedding_dim=8)
    with torch.no_grad():
        module.gain.fill_(0.1)
    logits3 = torch.full((1, num_classes, 5, 7), -100.0)
    logits2 = torch.full((1, num_classes, 9, 13), -100.0)
    if num_classes > 1:
        logits3[:, 0] = 100.0
        logits2[:, 0] = 100.0
    output = module(torch.randn(1, 16, 5, 7), torch.randn(1, 8, 9, 13),
                    torch.randn(1, 4, 17, 25), logits3, logits2)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("caa_mode", ["off", "caa"])
@pytest.mark.parametrize("prototype", [False, True])
def test_four_ablations_full_model(caa_mode, prototype):
    model = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False,
                     caa_mode=caa_mode, class_aware_prototype=prototype)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    outputs = model(x)
    assert len(outputs) == 4
    assert all(out.shape == (2, 1, 64, 64) for out in outputs)
    sum(out.square().mean() for out in outputs).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(param.grad is None or torch.isfinite(param.grad).all()
               for param in model.parameters())
    if not prototype:
        assert not any("prototype_calibration" in key for key in model.state_dict())


def test_strict_checkpoint_rejects_wrong_architecture():
    kwargs = dict(num_classes=1, encoder="pvt_v2_b0", pretrain=False, caa_mode="off")
    baseline = EMCADNet(**kwargs, class_aware_prototype=False)
    enhanced = EMCADNet(**kwargs, class_aware_prototype=True)
    with pytest.raises(RuntimeError):
        enhanced.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(enhanced.state_dict(), strict=True)

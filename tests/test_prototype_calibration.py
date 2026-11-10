"""Focused checks for the optional cross-scale calibration block."""

import torch

from lib.prototype_calibration import CrossScalePrototypeCalibration
from lib.networks import EMCADNet


def _inputs(channels, classes):
    sizes = (4, 8, 16, 32)
    features = [torch.randn(2, channel, size, size, requires_grad=True)
                for channel, size in zip(channels, sizes)]
    logits = [torch.randn(2, classes, size, size, requires_grad=True)
              for size in sizes]
    return features, logits


def test_binary_and_multiclass_start_as_exact_identity_and_receive_gradients():
    channels = (16, 12, 8, 4)
    for classes in (1, 4, 9):
        module = CrossScalePrototypeCalibration(channels, classes)
        features, logits = _inputs(channels, classes)
        output = module(features, logits)
        assert all(torch.equal(actual, original)
                   for actual, original in zip(output, logits))
        sum(item.square().mean() for item in output).backward()
        assert torch.isfinite(module.gain.grad).all()
        assert module.gain.grad.abs().sum() > 0


def test_cross_scale_signal_changes_logits_and_is_per_image():
    module = CrossScalePrototypeCalibration((8, 8, 8, 8), 1)
    with torch.no_grad():
        module.gain.fill_(0.5)
    features, logits = _inputs((8, 8, 8, 8), 1)
    original = module(features, logits)
    changed = [item.clone() for item in features]
    changed[0] = changed[0].clone()
    changed[0][0] = -changed[0][0]
    updated = module(changed, logits)
    assert not torch.equal(original[1][0], updated[1][0])
    assert torch.equal(original[1][1], updated[1][1])


def test_both_switches_off_keep_original_decoder_structure():
    model = EMCADNet(encoder="pvt_v2_b0", pretrain=False, caa_mode="off",
                     use_cross_scale_prototypes=False)
    assert model.prototype_calibration is None
    assert isinstance(model.decoder.eucb3.up_dwc[0], torch.nn.Upsample)
    assert isinstance(model.decoder.eucb2.up_dwc[0], torch.nn.Upsample)
    assert isinstance(model.decoder.eucb1.up_dwc[0], torch.nn.Upsample)
    assert not any(key.startswith("prototype_calibration.")
                   for key in model.state_dict())

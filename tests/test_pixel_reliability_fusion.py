import torch

from lib.networks import EMCADNet
from lib.pixel_reliability_fusion import PixelReliabilityFusion


def test_off_is_original_emcad_readout_and_has_no_fusion_parameters():
    model = EMCADNet(encoder="pvt_v2_b0", pretrain=False,
                     use_pixel_reliability_fusion=False)
    outputs = [torch.randn(1, 4, 8, 8) for _ in range(4)]
    assert model.fuse_outputs(outputs) is outputs[-1]
    assert isinstance(model.decoder.eucb3.up_dwc[0], torch.nn.Upsample)
    assert not any("pixel_reliability_fusion" in key for key in model.state_dict())


def test_fusion_weights_are_pixelwise_and_receive_gradients():
    module = PixelReliabilityFusion(4)
    outputs = [torch.randn(2, 4, 8, 8, requires_grad=True) for _ in range(4)]
    fused = module(outputs)
    assert fused.shape == outputs[0].shape
    weights = torch.softmax(module.scores(outputs), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    target = torch.randint(0, 4, (2, 8, 8))
    (fused.square().mean() + module.reliability_loss(outputs, target)).backward()
    assert all(head.weight.grad is not None for head in module.heads)


def test_binary_reliability_uses_foreground_threshold():
    module = PixelReliabilityFusion(1)
    outputs = [torch.full((1, 1, 2, 2), value) for value in (5.0, -5.0, 5.0, -5.0)]
    target = torch.ones(1, 1, 2, 2)
    expected = torch.tensor([[[[1.0] * 2] * 2,
                              [[0.0] * 2] * 2,
                              [[1.0] * 2] * 2,
                              [[0.0] * 2] * 2]])
    actual = torch.cat([((torch.sigmoid(output) >= 0.5) == (target >= 0.5)).float()
                        for output in outputs], dim=1)
    assert torch.equal(actual, expected)
    assert torch.isfinite(module.reliability_loss(outputs, target))

"""Runtime checks for the skip frequency gate and four CAA combinations."""

import unittest

try:
    import torch
    from lib.skip_frequency_gate import SkipFrequencyGate
except ImportError:
    torch = None

if torch is not None:
    try:
        from lib.networks import EMCADNet
    except ImportError:
        EMCADNet = None
else:
    EMCADNet = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SkipFrequencyGateTests(unittest.TestCase):
    def test_identity_shape_and_first_gradient(self):
        module = SkipFrequencyGate(17)
        skip = torch.randn(2, 17, 15, 13, requires_grad=True)
        decoder = torch.randn(2, 17, 15, 13, requires_grad=True)
        output = module(skip, decoder)
        self.assertEqual(output.shape, skip.shape)
        self.assertTrue(torch.equal(output, skip))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(module.alpha.grad).all())

    def test_active_band_selection_backward_and_strict_load(self):
        module = SkipFrequencyGate(8)
        module.alpha.data.fill_(0.2)
        skip = torch.randn(2, 8, 11, 13, requires_grad=True)
        decoder = torch.randn(2, 8, 11, 13, requires_grad=True)
        output = module(skip, decoder)
        self.assertFalse(torch.equal(output, skip))
        output.square().mean().backward()
        for tensor in (skip, decoder, *module.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        restored = SkipFrequencyGate(8)
        restored.load_state_dict(module.state_dict(), strict=True)
        self.assertTrue(torch.allclose(restored(skip, decoder), output))

    def test_cpu_autocast(self):
        module = SkipFrequencyGate(8)
        module.alpha.data.fill_(0.2)
        skip = torch.randn(1, 8, 9, 11)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(skip, skip)
        self.assertEqual(output.shape, skip.shape)
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(EMCADNet is None, "EMCADNet dependencies are unavailable")
    def test_four_ablation_combinations_and_baseline_initialization(self):
        for caa in ("off", "caa"):
            num_classes, input_channels = ((1, 1) if caa == "off" else (3, 3))
            torch.manual_seed(12345)
            baseline = EMCADNet(
                num_classes=num_classes, encoder="pvt_v2_b0", pretrain=False,
                caa_mode=caa, use_skip_frequency_gate=0,
            ).eval()
            torch.manual_seed(12345)
            enabled = EMCADNet(
                num_classes=num_classes, encoder="pvt_v2_b0", pretrain=False,
                caa_mode=caa, use_skip_frequency_gate=1,
            ).eval()
            old = baseline.state_dict()
            new = enabled.state_dict()
            for key, value in old.items():
                self.assertTrue(torch.equal(value, new[key]), key)
            self.assertFalse(any("skip_frequency_gates" in key for key in old))
            self.assertTrue(any("skip_frequency_gates" in key for key in new))
            image = torch.randn(1, input_channels, 64, 64, requires_grad=True)
            old_outputs = baseline(image)
            new_outputs = enabled(image)
            self.assertEqual(len(new_outputs), 4)
            for old_output, new_output in zip(old_outputs, new_outputs):
                self.assertEqual(new_output.shape, (1, num_classes, 64, 64))
                self.assertTrue(torch.equal(old_output, new_output))
            new_outputs[-1].square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            enabled.load_state_dict(new, strict=True)


if __name__ == "__main__":
    unittest.main()

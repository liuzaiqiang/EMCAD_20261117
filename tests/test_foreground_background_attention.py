"""Run on a PyTorch environment to check the attention and four ablations."""

import unittest

try:
    import torch
    from lib.foreground_background_attention import ForegroundBackgroundDecoupledAttention
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
class ForegroundBackgroundAttentionTests(unittest.TestCase):
    def test_identity_odd_size_and_gradient(self):
        module = ForegroundBackgroundDecoupledAttention(17)
        feature = torch.randn(2, 17, 15, 13, requires_grad=True)
        output = module(feature)
        self.assertEqual(output.shape, feature.shape)
        self.assertTrue(torch.equal(output, feature))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(module.alpha.grad).all())

    def test_active_path_and_checkpoint(self):
        module = ForegroundBackgroundDecoupledAttention(8)
        module.alpha.data.fill_(0.2)
        feature = torch.randn(2, 8, 11, 13, requires_grad=True)
        output = module(feature)
        self.assertFalse(torch.equal(output, feature))
        output.square().mean().backward()
        for tensor in (feature, *module.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        restored = ForegroundBackgroundDecoupledAttention(8)
        restored.load_state_dict(module.state_dict(), strict=True)
        self.assertTrue(torch.allclose(restored(feature), output))

    def test_cpu_autocast(self):
        module = ForegroundBackgroundDecoupledAttention(8)
        module.alpha.data.fill_(0.2)
        feature = torch.randn(1, 8, 9, 11)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(feature)
        self.assertEqual(output.shape, feature.shape)
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(EMCADNet is None, "EMCADNet dependencies are unavailable")
    def test_four_ablation_combinations_and_seeded_baseline(self):
        for caa in ("off", "caa"):
            classes, channels = ((1, 1) if caa == "off" else (3, 3))
            torch.manual_seed(12345)
            baseline = EMCADNet(
                num_classes=classes, encoder="pvt_v2_b0", pretrain=False,
                caa_mode=caa, use_foreground_background_attention=0,
            ).eval()
            torch.manual_seed(12345)
            enabled = EMCADNet(
                num_classes=classes, encoder="pvt_v2_b0", pretrain=False,
                caa_mode=caa, use_foreground_background_attention=1,
            ).eval()
            old, new = baseline.state_dict(), enabled.state_dict()
            for key, value in old.items():
                self.assertTrue(torch.equal(value, new[key]), key)
            self.assertFalse(any("foreground_background_attention" in key for key in old))
            self.assertTrue(any("foreground_background_attention" in key for key in new))
            image = torch.randn(1, channels, 64, 64, requires_grad=True)
            old_outputs, new_outputs = baseline(image), enabled(image)
            self.assertEqual(len(new_outputs), 4)
            for left, right in zip(old_outputs, new_outputs):
                self.assertEqual(right.shape, (1, classes, 64, 64))
                self.assertTrue(torch.equal(left, right))
            new_outputs[-1].square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            enabled.load_state_dict(new, strict=True)


if __name__ == "__main__":
    unittest.main()

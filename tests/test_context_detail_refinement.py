"""Run on the training environment with: python -m unittest tests.test_context_detail_refinement"""

import unittest

try:
    import torch
    from lib.context_detail_refinement import ContextDetailRefinement
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
class ContextDetailRefinementTests(unittest.TestCase):
    def test_identity_at_initialization_and_odd_shapes(self):
        for channels in ((64, 128), (32, 64)):
            module = ContextDetailRefinement(*channels)
            fine = torch.randn(2, channels[0], 55, 53, requires_grad=True)
            coarse = torch.randn(2, channels[1], 28, 27, requires_grad=True)
            output = module(fine, coarse)
            self.assertEqual(output.shape, fine.shape)
            self.assertTrue(torch.equal(output, fine))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(module.alpha.grad).all())

    def test_active_branch_has_finite_gradients_and_strict_state_dict(self):
        module = ContextDetailRefinement(16, 32)
        module.alpha.data.fill_(0.1)
        fine = torch.randn(2, 16, 17, 19, requires_grad=True)
        coarse = torch.randn(2, 32, 9, 10, requires_grad=True)
        output = module(fine, coarse)
        output.square().mean().backward()
        for tensor in (fine, coarse, *module.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        restored = ContextDetailRefinement(16, 32)
        restored.load_state_dict(module.state_dict(), strict=True)
        self.assertTrue(torch.allclose(restored(fine, coarse), output))

    def test_cpu_autocast(self):
        module = ContextDetailRefinement(8, 16)
        module.alpha.data.fill_(0.1)
        fine = torch.randn(1, 8, 15, 13)
        coarse = torch.randn(1, 16, 8, 7)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(fine, coarse)
        self.assertEqual(output.shape, fine.shape)
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(EMCADNet is None, "EMCADNet dependencies are unavailable")
    def test_four_ablation_models_and_checkpoint_keys(self):
        # Small PVTv2-B0 inputs exercise the actual EMCAD routing and all heads.
        for caa in ("off", "caa"):
            for refinement in (0, 1):
                with self.subTest(caa=caa, refinement=refinement):
                    model = EMCADNet(
                        num_classes=3, encoder="pvt_v2_b0", pretrain=False,
                        caa_mode=caa, use_context_detail_refinement=refinement,
                    ).eval()
                    height, width = (65, 67) if refinement else (64, 64)
                    image = torch.randn(1, 3, height, width, requires_grad=True)
                    outputs = model(image)
                    self.assertEqual(len(outputs), 4)
                    self.assertTrue(all(out.shape == (1, 3, height, width) for out in outputs))
                    outputs[-1].square().mean().backward()
                    self.assertTrue(torch.isfinite(image.grad).all())
                    keys = model.state_dict()
                    self.assertEqual(
                        any(key.startswith("context_detail_refinement.") for key in keys),
                        bool(refinement),
                    )
                    restored = EMCADNet(
                        num_classes=3, encoder="pvt_v2_b0", pretrain=False,
                        caa_mode=caa, use_context_detail_refinement=refinement,
                    )
                    restored.load_state_dict(keys, strict=True)

    @unittest.skipIf(EMCADNet is None, "EMCADNet dependencies are unavailable")
    def test_enabling_branch_preserves_original_initialization(self):
        torch.manual_seed(314159)
        baseline = EMCADNet(encoder="pvt_v2_b0", pretrain=False,
                            use_context_detail_refinement=0)
        torch.manual_seed(314159)
        enhanced = EMCADNet(encoder="pvt_v2_b0", pretrain=False,
                            use_context_detail_refinement=1)
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, enhanced.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()

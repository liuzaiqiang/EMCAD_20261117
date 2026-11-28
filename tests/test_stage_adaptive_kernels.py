"""Regression checks for the optional MSDC scale selector."""

import unittest

import torch
from torch.nn import functional as F

from lib.decoders import MSCB, GradientAlignedScaleFusion, PhaseConservativeUpsample
from lib.networks import EMCADNet


class StageAdaptiveKernelTests(unittest.TestCase):
    def test_disabled_path_has_original_state_and_same_initial_output(self):
        torch.manual_seed(21)
        baseline = MSCB(16, 16, 1, stage_adaptive_kernels=0).eval()
        torch.manual_seed(21)
        adaptive = MSCB(16, 16, 1, stage_adaptive_kernels=1).eval()

        baseline_state = baseline.state_dict()
        adaptive_state = adaptive.state_dict()
        self.assertFalse(any('gradient_scale_fusion' in key for key in baseline_state))
        self.assertTrue(set(baseline_state).issubset(adaptive_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, adaptive_state[key]), key)

        inputs = torch.randn(2, 16, 12, 12)
        with torch.no_grad():
            expected = baseline(inputs)
            actual = adaptive(inputs)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

        with torch.no_grad():
            adaptive.gradient_scale_fusion.scale_gains.fill_(1.0)
            changed = adaptive(inputs)
        self.assertFalse(torch.allclose(expected, changed))
        adaptive.train()
        adaptive.zero_grad(set_to_none=True)
        adaptive(inputs).square().mean().backward()
        self.assertIsNotNone(adaptive.gradient_scale_fusion.scale_gains.grad)

    def test_full_model_switches_are_independent(self):
        for phase_upsampling in (0, 1):
            for stage_adaptive_kernels in (0, 1):
                with self.subTest(phase_upsampling=phase_upsampling, stage_adaptive_kernels=stage_adaptive_kernels):
                    model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                                     phase_upsampling=phase_upsampling,
                                     stage_adaptive_kernels=stage_adaptive_kernels).eval()
                    modules = list(model.modules())
                    self.assertEqual(sum(isinstance(m, GradientAlignedScaleFusion) for m in modules),
                                     4 * stage_adaptive_kernels)
                    self.assertEqual(sum(isinstance(m, PhaseConservativeUpsample) for m in modules),
                                     3 * phase_upsampling)
                    with torch.no_grad():
                        outputs = model(torch.randn(1, 3, 64, 64))
                    self.assertEqual(len(outputs), 4)
                    self.assertTrue(all(output.shape == (1, 1, 64, 64) for output in outputs))

    def test_full_model_original_weights_keep_same_seed(self):
        torch.manual_seed(2222)
        baseline = EMCADNet(num_classes=9, encoder='pvt_v2_b0', pretrain=False,
                            phase_upsampling=0, stage_adaptive_kernels=0)
        torch.manual_seed(2222)
        adaptive = EMCADNet(num_classes=9, encoder='pvt_v2_b0', pretrain=False,
                            phase_upsampling=1, stage_adaptive_kernels=1)
        baseline_state = baseline.state_dict()
        adaptive_state = adaptive.state_dict()
        self.assertTrue(set(baseline_state).issubset(adaptive_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, adaptive_state[key]), key)

    def test_scale_fusion_rejects_concatenation(self):
        with self.assertRaisesRegex(ValueError, 'additive'):
            MSCB(16, 16, 1, add=False, stage_adaptive_kernels=1)

    def test_phase_guidance_changes_only_detail_supported_by_the_skip(self):
        module = PhaseConservativeUpsample(1, residual_scale=0.5)
        with torch.no_grad():
            module.guide.weight.fill_(1)
        decoder = torch.arange(4.0).view(1, 1, 1, 4).expand(1, 1, 4, 4)
        parallel = torch.tensor([-1., 1.] * 4).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        baseline = F.interpolate(decoder, scale_factor=2, mode='bilinear', align_corners=False)
        output = module(decoder, parallel)
        torch.testing.assert_close(F.avg_pool2d(output, 2), F.avg_pool2d(baseline, 2))
        torch.testing.assert_close(module(decoder, parallel.transpose(-1, -2)), baseline)
        self.assertFalse(torch.equal(output, baseline))
        output.square().mean().backward()
        self.assertGreater(module.guide.weight.grad.abs().sum().item(), 0)

        step = (decoder >= 2).float()
        sharp = module(step, parallel)
        reference = F.interpolate(step, scale_factor=2, mode='bilinear', align_corners=False)
        torch.testing.assert_close(F.avg_pool2d(sharp, 2), F.avg_pool2d(reference, 2))
        self.assertTrue(((sharp >= 0) & (sharp <= 1)).all())

    def test_gradient_fusion_uses_local_agreement_and_preserves_initial_sum(self):
        module = GradientAlignedScaleFusion([1, 3], parallel=True)
        edge = torch.tensor([0., 0., 1., 1., 1.]).view(1, 1, 1, 5).expand(1, 1, 5, 5)
        branches = [torch.zeros_like(edge), edge]
        torch.testing.assert_close(module(edge, branches), edge, rtol=0, atol=0)
        module(edge, branches).square().mean().backward()
        self.assertGreater(module.scale_gains.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.scale_gains.fill_(1)
        torch.testing.assert_close(module(edge.transpose(-1, -2), branches), edge)
        output = module(edge, branches)
        self.assertFalse(torch.equal(output, edge))
        torch.testing.assert_close(output[..., 0], edge[..., 0])
        torch.testing.assert_close(output[..., -1], edge[..., -1])


if __name__ == '__main__':
    unittest.main()

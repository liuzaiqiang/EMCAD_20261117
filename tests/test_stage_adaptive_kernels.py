"""Regression checks for the optional MSDC scale selector."""

import unittest

import torch

from lib.decoders import MSCB, StageAdaptiveKernelCompetition, ContentAwareAntiAliasUpsample
from lib.networks import EMCADNet


class StageAdaptiveKernelTests(unittest.TestCase):
    def test_disabled_path_has_original_state_and_same_initial_output(self):
        torch.manual_seed(21)
        baseline = MSCB(16, 16, 1, stage_adaptive_kernels=0).eval()
        torch.manual_seed(21)
        adaptive = MSCB(16, 16, 1, stage_adaptive_kernels=1).eval()

        baseline_state = baseline.state_dict()
        adaptive_state = adaptive.state_dict()
        self.assertFalse(any('kernel_competition' in key for key in baseline_state))
        self.assertTrue(set(baseline_state).issubset(adaptive_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, adaptive_state[key]), key)

        inputs = torch.randn(2, 16, 12, 12)
        with torch.no_grad():
            expected = baseline(inputs)
            actual = adaptive(inputs)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

        with torch.no_grad():
            adaptive.kernel_competition.selector[-1].bias.copy_(torch.tensor([2., 0., -2.]))
            changed = adaptive(inputs)
        self.assertFalse(torch.allclose(expected, changed))
        adaptive.train()
        adaptive.zero_grad(set_to_none=True)
        adaptive(inputs).square().mean().backward()
        self.assertIsNotNone(adaptive.kernel_competition.selector[-1].weight.grad)

    def test_full_model_switches_are_independent(self):
        for caa_mode in ('off', 'caa'):
            for stage_adaptive_kernels in (0, 1):
                with self.subTest(caa_mode=caa_mode, stage_adaptive_kernels=stage_adaptive_kernels):
                    model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                                     caa_mode=caa_mode,
                                     stage_adaptive_kernels=stage_adaptive_kernels).eval()
                    modules = list(model.modules())
                    self.assertEqual(any(isinstance(m, ContentAwareAntiAliasUpsample)
                                         for m in modules), caa_mode == 'caa')
                    self.assertEqual(any(isinstance(m, StageAdaptiveKernelCompetition)
                                         for m in modules), bool(stage_adaptive_kernels))
                    with torch.no_grad():
                        outputs = model(torch.randn(1, 3, 64, 64))
                    self.assertEqual(len(outputs), 4)
                    self.assertTrue(all(output.shape == (1, 1, 64, 64) for output in outputs))

    def test_full_model_original_weights_keep_same_seed(self):
        torch.manual_seed(2222)
        baseline = EMCADNet(num_classes=9, encoder='pvt_v2_b0', pretrain=False,
                            caa_mode='off', stage_adaptive_kernels=0)
        torch.manual_seed(2222)
        adaptive = EMCADNet(num_classes=9, encoder='pvt_v2_b0', pretrain=False,
                            caa_mode='off', stage_adaptive_kernels=1)
        baseline_state = baseline.state_dict()
        adaptive_state = adaptive.state_dict()
        self.assertTrue(set(baseline_state).issubset(adaptive_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, adaptive_state[key]), key)

    def test_competition_rejects_concatenation(self):
        with self.assertRaisesRegex(ValueError, 'additive'):
            MSCB(16, 16, 1, add=False, stage_adaptive_kernels=1)


if __name__ == '__main__':
    unittest.main()

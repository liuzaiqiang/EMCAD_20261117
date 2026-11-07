"""Regression checks for optional post-decoder d1 boundary refinement."""

import unittest

import torch

from lib.boundary_refinement import ResidualBoundaryRefinement
from lib.decoders import ContentAwareAntiAliasUpsample
from lib.networks import EMCADNet


class ResidualBoundaryRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_identity_initialization_and_trainable_correction(self):
        module = ResidualBoundaryRefinement(16)
        x = torch.randn(2, 16, 12, 12)
        self.assertTrue(torch.equal(module(x), x))
        module(x).square().mean().backward()
        self.assertIsNotNone(module.residual_pw.weight.grad)
        self.assertGreater(module.residual_pw.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.residual_pw.weight.fill_(0.02)
            changed = module(x)
        self.assertFalse(torch.allclose(changed, x))

    def test_all_switch_combinations_and_baseline_weights(self):
        torch.manual_seed(2222)
        baseline = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                            caa_mode='off', residual_boundary_refinement=0).eval()
        torch.manual_seed(2222)
        refined = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                           caa_mode='off', residual_boundary_refinement=1).eval()
        base_state = baseline.state_dict()
        refined_state = refined.state_dict()
        self.assertFalse(any('boundary_refinement' in key for key in base_state))
        self.assertTrue(set(base_state).issubset(refined_state))
        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, refined_state[key]), key)

        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            base_outputs = baseline(image)
            refined_outputs = refined(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(base_outputs, refined_outputs)))

        with torch.no_grad():
            refined.boundary_refinement.residual_pw.weight.fill_(0.02)
            changed_outputs = refined(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(base_outputs[:3], changed_outputs[:3])))
        self.assertFalse(torch.allclose(base_outputs[3], changed_outputs[3]))

        for caa_mode in ('off', 'caa'):
            for switch in (0, 1):
                with self.subTest(caa_mode=caa_mode, refinement=switch):
                    model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                                     caa_mode=caa_mode, residual_boundary_refinement=switch).eval()
                    modules = list(model.modules())
                    self.assertEqual(any(isinstance(m, ContentAwareAntiAliasUpsample)
                                         for m in modules), caa_mode == 'caa')
                    self.assertEqual(any(isinstance(m, ResidualBoundaryRefinement)
                                         for m in modules), bool(switch))
                    with torch.no_grad():
                        outputs = model(image)
                    self.assertEqual(len(outputs), 4)
                    self.assertTrue(all(o.shape == (1, 1, 64, 64) for o in outputs))

    def test_invalid_switch_rejected(self):
        with self.assertRaisesRegex(ValueError, '0 or 1'):
            EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                     residual_boundary_refinement=2)


if __name__ == '__main__':
    unittest.main()

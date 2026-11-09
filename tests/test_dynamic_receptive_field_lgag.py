"""Small architecture checks; run on a server with PyTorch and timm installed."""

import unittest

import torch

from lib.decoders import EMCAD, LGAG, DynamicReceptiveFieldLGAG


class DynamicReceptiveFieldLGAGTest(unittest.TestCase):
    def test_initial_gate_and_odd_spatial_size(self):
        torch.manual_seed(13)
        dynamic = DynamicReceptiveFieldLGAG(16, 16, 8, groups=8)
        fixed = LGAG(16, 16, 8, groups=8)
        fixed.load_state_dict({key: value for key, value in dynamic.state_dict().items()
                               if key in fixed.state_dict()}, strict=True)
        dynamic.eval()
        fixed.eval()
        g = torch.randn(2, 16, 13, 17, requires_grad=True)
        x = torch.randn(2, 16, 13, 17, requires_grad=True)
        actual = dynamic(g, x)
        torch.testing.assert_close(actual, fixed(g, x), rtol=0, atol=0)
        self.assertEqual(actual.shape, x.shape)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(g.grad).all().item())
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(dynamic.rf_scale.grad).all().item())
        with self.assertRaises(RuntimeError):
            fixed.load_state_dict(dynamic.state_dict(), strict=True)

    def test_four_ablation_combinations(self):
        for caa_mode in ('off', 'caa'):
            for enabled in (0, 1):
                with self.subTest(caa_mode=caa_mode, enabled=enabled):
                    decoder = EMCAD(channels=[64, 32, 16, 8], expansion_factor=2,
                                    caa_mode=caa_mode, dynamic_receptive_field_lgag=enabled)
                    self.assertIsInstance(decoder.lgag1,
                                          DynamicReceptiveFieldLGAG if enabled else LGAG)
                    x = torch.randn(2, 64, 3, 5, requires_grad=True)
                    skips = [torch.randn(2, 32, 6, 10), torch.randn(2, 16, 12, 20),
                             torch.randn(2, 8, 24, 40)]
                    outputs = decoder(x, skips)
                    self.assertEqual([o.shape[-2:] for o in outputs],
                                     [(3, 5), (6, 10), (12, 20), (24, 40)])
                    sum(o.square().mean() for o in outputs).backward()
                    self.assertTrue(torch.isfinite(x.grad).all().item())
                    self.assertTrue(all(torch.isfinite(p.grad).all().item()
                                        for p in decoder.parameters() if p.grad is not None))
                    clone = EMCAD(channels=[64, 32, 16, 8], expansion_factor=2,
                                  caa_mode=caa_mode, dynamic_receptive_field_lgag=enabled)
                    clone.load_state_dict(decoder.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()

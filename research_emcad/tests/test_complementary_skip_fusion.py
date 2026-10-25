import unittest

import torch

from research_emcad.complementary_skip_fusion import ComplementarySkipFusion


class ComplementarySkipFusionTests(unittest.TestCase):
    def test_initial_identity_and_trainable_correction(self):
        module = ComplementarySkipFusion(8)
        decoder = torch.randn(2, 8, 9, 11, requires_grad=True)
        skip = torch.randn_like(decoder, requires_grad=True)
        initial = module(decoder, skip)
        self.assertTrue(torch.equal(initial, decoder + skip))
        initial.square().mean().backward()
        self.assertTrue(torch.count_nonzero(module.shared_gain.grad).item() > 0)
        self.assertTrue(torch.count_nonzero(module.novel_gain.grad).item() > 0)
        with torch.no_grad():
            module.shared_gain.fill_(0.2)
            module.novel_gain.fill_(0.2)
        changed = module(decoder.detach(), skip.detach())
        self.assertTrue(torch.isfinite(changed).all().item())
        self.assertFalse(torch.equal(changed, decoder.detach() + skip.detach()))

    def test_shape_validation(self):
        module = ComplementarySkipFusion(4)
        with self.assertRaises(ValueError):
            module(torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8, 8))


if __name__ == '__main__':
    unittest.main()

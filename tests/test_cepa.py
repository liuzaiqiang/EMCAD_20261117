"""CEPA architecture smoke tests for binary and multiclass EMCAD paths."""

import unittest

import torch

from lib.cepa import CEPA


class CEPATests(unittest.TestCase):
    def test_identity_at_initialization_and_gradients(self):
        for channels, side in ((64, 56), (128, 28), (320, 14)):
            module = CEPA(channels)
            skip = torch.randn(2, channels, side, side, requires_grad=True)
            decoder = torch.randn_like(skip)
            output = module(skip, decoder)
            self.assertTrue(torch.equal(output, skip))
            output.mean().backward()
            self.assertTrue(torch.isfinite(skip.grad).all())
            self.assertTrue(torch.isfinite(module.residual_scale.grad))

    def test_shapes_after_residual_learns(self):
        module = CEPA(64)
        with torch.no_grad():
            module.residual_scale.fill_(0.2)
        for side in (56, 64, 88):
            skip = torch.randn(1, 64, side, side, requires_grad=True)
            output = module(skip, torch.randn_like(skip))
            self.assertEqual(output.shape, skip.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(skip.grad).all())
        with self.assertRaises(ValueError):
            module(torch.randn(1, 64, 56, 56), torch.randn(1, 64, 28, 28))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp_and_checkpoint(self):
        module = CEPA(64).cuda()
        with torch.no_grad():
            module.residual_scale.fill_(0.2)
        skip = torch.randn(1, 64, 56, 56, device="cuda", requires_grad=True)
        decoder = torch.randn_like(skip)
        with torch.cuda.amp.autocast():
            result = module(skip, decoder)
            result.mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(skip.grad).all())
        restored = CEPA(64).cuda()
        restored.load_state_dict(module.state_dict(), strict=True)
        restored.eval()
        module.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(restored(skip, decoder), module(skip, decoder)))


if __name__ == "__main__":
    unittest.main()

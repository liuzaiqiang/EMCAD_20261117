import unittest

import numpy as np
import torch

from utils.robust_boundary_loss import (
    normalized_signed_distance,
    robust_boundary_loss,
    validate_robust_boundary_config,
)


class RobustBoundaryLossTests(unittest.TestCase):
    def test_distance_sign_and_clipping(self):
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[8:16, 8:16] = 1
        signed = normalized_signed_distance(mask, percentile=50)
        self.assertLess(signed[12, 12], 0)
        self.assertGreater(signed[0, 0], 0)
        self.assertLessEqual(float(np.abs(signed).max()), 1.0)
        self.assertIsNone(normalized_signed_distance(np.zeros((4, 4))))
        self.assertIsNone(normalized_signed_distance(np.ones((4, 4))))

    def test_binary_correct_prediction_has_smaller_loss(self):
        target = torch.zeros((1, 16, 16))
        target[:, 5:11, 5:11] = 1
        correct = ((target * 2 - 1) * 8).unsqueeze(1).requires_grad_()
        inverted = (-correct.detach()).requires_grad_()
        good_loss = robust_boundary_loss(correct, target)
        bad_loss = robust_boundary_loss(inverted, target)
        self.assertLess(good_loss.item(), bad_loss.item())
        good_loss.backward()
        self.assertTrue(torch.isfinite(correct.grad).all())

    def test_multiclass_missing_classes_and_gradient(self):
        target = torch.zeros((2, 16, 16), dtype=torch.long)
        target[0, 3:8, 3:8] = 1
        target[1, 9:14, 9:14] = 2
        logits = torch.randn((2, 9, 16, 16), requires_grad=True)
        loss = robust_boundary_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_background_and_config_validation(self):
        logits = torch.randn((2, 1, 8, 8), requires_grad=True)
        loss = robust_boundary_loss(logits, torch.zeros((2, 8, 8)))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for enabled, weight, percentile in ((2, 0.1, 95), (1, 0, 95), (1, 0.1, 0)):
            with self.assertRaises(ValueError):
                validate_robust_boundary_config(enabled, weight, percentile)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward_binary_and_multiclass(self):
        for channels in (1, 4):
            logits = torch.randn((2, channels, 32, 32), device="cuda", requires_grad=True)
            target = torch.zeros((2, 32, 32), device="cuda", dtype=torch.long)
            target[:, 9:21, 8:22] = 1
            if channels > 1:
                target[1, 13:18, 13:18] = 2
            loss = robust_boundary_loss(logits, target)
            loss.backward()
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()

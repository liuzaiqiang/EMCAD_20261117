"""Checks for training-only size-aware hard-positive supervision."""

import unittest

import torch

from utils.small_target_loss import small_target_hard_positive_loss
from utils.acdc_utils import DiceLoss, supervised_loss
from utils.polyp_utils import supervised_structure_loss


class SmallTargetLossTests(unittest.TestCase):
    def test_binary_small_and_hard_positives_receive_more_loss(self):
        small = torch.zeros(1, 1, 10, 10)
        small[..., 0, 0] = 1
        large = torch.zeros(1, 1, 10, 10)
        large[..., :3, :] = 1
        same_prediction = torch.zeros(1, 1, 10, 10)
        small_loss = small_target_hard_positive_loss(same_prediction, small)
        large_loss = small_target_hard_positive_loss(same_prediction, large)
        self.assertGreater(small_loss.item(), large_loss.item())

        hard = torch.full((1, 1, 10, 10), -2.0, requires_grad=True)
        easy = torch.full((1, 1, 10, 10), 2.0)
        hard_loss = small_target_hard_positive_loss(hard, small)
        easy_loss = small_target_hard_positive_loss(easy, small)
        self.assertGreater(hard_loss.item(), easy_loss.item())
        hard_loss.backward()
        self.assertIsNotNone(hard.grad)
        self.assertLess(hard.grad[0, 0, 0, 0].item(), 0)
        self.assertEqual(hard.grad[0, 0, 0, 1].item(), 0)

    def test_multiclass_missing_classes_and_empty_binary(self):
        logits = torch.zeros(1, 4, 8, 8, requires_grad=True)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[0, 0, 0] = 2
        value = small_target_hard_positive_loss(logits, target)
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(value.item(), 0)
        value.backward()
        self.assertIsNotNone(logits.grad)

        empty = torch.zeros(1, 1, 8, 8)
        binary_logits = torch.randn(1, 1, 8, 8, requires_grad=True)
        zero = small_target_hard_positive_loss(binary_logits, empty)
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertTrue(torch.equal(binary_logits.grad, torch.zeros_like(binary_logits)))

    def test_multiclass_and_binary_training_paths_keep_off_loss(self):
        multi_outputs = [torch.randn(1, 4, 16, 16, requires_grad=True) for _ in range(4)]
        multi_target = torch.randint(0, 4, (1, 16, 16))
        ce = torch.nn.CrossEntropyLoss()
        dice = DiceLoss(4)
        baseline = supervised_loss(multi_outputs, multi_target, 'deep_supervision', ce, dice)
        explicit_off = supervised_loss(multi_outputs, multi_target, 'deep_supervision',
                                       ce, dice, hard_positive_loss=None)
        active = supervised_loss(multi_outputs, multi_target, 'deep_supervision',
                                 ce, dice, hard_positive_loss=small_target_hard_positive_loss)
        self.assertTrue(torch.equal(baseline, explicit_off))
        self.assertGreater(active.item(), baseline.item())

        binary_outputs = [torch.randn(1, 1, 16, 16, requires_grad=True) for _ in range(4)]
        binary_target = (torch.rand(1, 1, 16, 16) > 0.8).float()
        baseline_binary = supervised_structure_loss(binary_outputs, binary_target, 'paper')
        explicit_off_binary = supervised_structure_loss(binary_outputs, binary_target,
                                                        'paper', hard_positive_loss=None)
        active_binary = supervised_structure_loss(binary_outputs, binary_target,
                                                  'paper', small_target_hard_positive_loss)
        self.assertTrue(torch.equal(baseline_binary, explicit_off_binary))
        self.assertGreater(active_binary.item(), baseline_binary.item())


if __name__ == '__main__':
    unittest.main()

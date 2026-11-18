"""Forward, gradient, switch and profiler checks for the RA adapter."""

import contextlib
from io import StringIO
import tempfile
import unittest

import torch

from lib.model_complexity import log_model_complexity
from lib.networks import EMCADNet
from lib.reverse_attention import ReverseAttentionRefinement


class ReverseAttentionTests(unittest.TestCase):
    def test_binary_and_multiclass_residual_shapes(self):
        torch.set_num_threads(2)
        for classes in (1, 4, 9):
            module = ReverseAttentionRefinement(32, classes)
            fine = torch.randn(1, 32, 56, 56, requires_grad=True)
            coarse = torch.randn(1, classes, 28, 28, requires_grad=True)
            output = module(fine, coarse)
            self.assertEqual(tuple(output.shape), (1, classes, 56, 56))
            output.mean().backward()
            self.assertIsNotNone(fine.grad)
            self.assertIsNotNone(coarse.grad)

    def test_switch_and_calparams_state(self):
        torch.set_num_threads(2)
        with contextlib.redirect_stdout(StringIO()):
            off = EMCADNet(num_classes=9, encoder="pvt_v2_b0",
                           pretrain=False, ra_enabled=0)
            on = EMCADNet(num_classes=9, encoder="pvt_v2_b0",
                          pretrain=False, ra_enabled=1)
        self.assertFalse(any(key.startswith("ra_d1.") for key in off.state_dict()))
        self.assertTrue(any(key.startswith("ra_d1.") for key in on.state_dict()))
        output = on(torch.rand(1, 1, 224, 224))
        self.assertEqual([tuple(item.shape) for item in output],
                         [(1, 9, 224, 224)] * 4)
        keys_before = set(on.state_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir + "/model_complexity.txt"
            with contextlib.redirect_stdout(StringIO()):
                log_model_complexity(on, (1, 1, 224, 224), output_path=path)
            with open(path, encoding="utf-8") as stream:
                report = stream.read()
        self.assertIn("[Statistics Information]", report)
        self.assertIn("FLOPs:", report)
        self.assertIn("Params:", report)
        self.assertEqual(keys_before, set(on.state_dict()))
        self.assertTrue(on.training)


if __name__ == "__main__":
    unittest.main()

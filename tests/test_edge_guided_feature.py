"""Smoke checks for the EMCAD EFB integration and complexity wrapper."""

import contextlib
from io import StringIO
import tempfile
import unittest

import torch

from lib.model_complexity import log_model_complexity
from lib.networks import EMCADNet


class EdgeGuidedFeatureTests(unittest.TestCase):
    def test_switch_and_backward_with_grayscale_input(self):
        torch.set_num_threads(2)
        with contextlib.redirect_stdout(StringIO()):
            off = EMCADNet(num_classes=9, encoder="pvt_v2_b0",
                           pretrain=False, efb_enabled=0)
            on = EMCADNet(num_classes=9, encoder="pvt_v2_b0",
                          pretrain=False, efb_enabled=1)
        self.assertFalse(any(key.startswith("efb.") for key in off.state_dict()))
        self.assertTrue(any(key.startswith("efb.") for key in on.state_dict()))
        output = on(torch.rand(1, 1, 224, 224))
        self.assertEqual([tuple(item.shape) for item in output],
                         [(1, 9, 224, 224)] * 4)
        output[-1].mean().backward()
        self.assertIsNotNone(on.efb.gate[0].weight.grad)

    def test_original_calparams_output_does_not_pollute_checkpoint(self):
        torch.set_num_threads(2)
        with contextlib.redirect_stdout(StringIO()):
            model = EMCADNet(num_classes=1, encoder="pvt_v2_b0",
                             pretrain=False, efb_enabled=1)
        keys_before = set(model.state_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir + "/model_complexity.txt"
            with contextlib.redirect_stdout(StringIO()):
                log_model_complexity(model, (1, 3, 224, 224), output_path=path)
            with open(path, encoding="utf-8") as stream:
                report = stream.read()
        self.assertIn("[Statistics Information]", report)
        self.assertIn("FLOPs:", report)
        self.assertIn("Params:", report)
        self.assertEqual(keys_before, set(model.state_dict()))
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

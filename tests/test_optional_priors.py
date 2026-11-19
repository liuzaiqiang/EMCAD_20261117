"""Runtime checks for isolated shape prior branches and multiclass training."""

import unittest

import torch

from lib.networks import EMCADNet


class OptionalPriorTests(unittest.TestCase):
    def test_all_switch_combinations(self):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        baseline = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False)
        baseline_keys = tuple(baseline.state_dict())
        self.assertFalse(any(".spm." in k for k in baseline_keys))
        for spm in (False, True):
            with self.subTest(spm=spm):
                model = EMCADNet(num_classes=1, encoder="pvt_v2_b0",
                                 pretrain=False, spm_enabled=spm).to(device).eval()
                if not spm:
                    self.assertEqual(baseline_keys, tuple(model.state_dict()))
                image = torch.randn(1, 3, 64, 64, device=device)
                outputs = model(image, mode="train")
                loss = outputs[-1].square().mean()
                self.assertEqual(len(outputs), 4)
                self.assertEqual(tuple(outputs[-1].shape), (1, 1, 64, 64))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_multiclass_spm(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EMCADNet(num_classes=4, encoder="pvt_v2_b0", pretrain=False,
                         spm_enabled=True).to(device).eval()
        outputs = model(torch.randn(1, 1, 64, 64, device=device), mode="train")
        loss = outputs[-1].square().mean()
        loss.backward()
        self.assertEqual(tuple(outputs[-1].shape), (1, 4, 64, 64))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

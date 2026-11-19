"""Check CFP isolation and 2D forward/backward for all task class counts."""

import unittest

import torch

from lib.cfp_context import CFPModule
from lib.networks import EMCADNet


class CFPContextTests(unittest.TestCase):
    def test_disabled_state_matches_base(self):
        base = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False)
        off = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False,
                       cfp_enabled=False)
        self.assertEqual(tuple(base.state_dict()), tuple(off.state_dict()))
        self.assertFalse(any(".cfp" in name for name in off.state_dict()))
        off.load_state_dict(base.state_dict(), strict=True)
        base.eval()
        off.eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = base(image, mode="test")
            actual = off(image, mode="test")
        self.assertTrue(all(torch.equal(left, right)
                            for left, right in zip(expected, actual)))

    def test_author_dilations_and_shapes(self):
        module = CFPModule(32, dilation=8)
        self.assertEqual([branch[0].conv.dilation for branch in module.branches],
                         [(1, 1), (3, 3), (5, 5), (9, 9)])
        self.assertEqual(tuple(module(torch.randn(2, 32, 7, 7)).shape), (2, 32, 7, 7))

    def test_binary_acdc_synapse(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for classes, channels in ((1, 3), (4, 1), (9, 1)):
            with self.subTest(classes=classes):
                model = EMCADNet(num_classes=classes, encoder="pvt_v2_b0",
                                 pretrain=False, cfp_enabled=True).to(device).eval()
                outputs = model(torch.randn(1, channels, 64, 64, device=device), mode="train")
                self.assertEqual(tuple(outputs[-1].shape), (1, classes, 64, 64))
                loss = outputs[-1].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

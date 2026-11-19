"""Checks for optional PointRend architecture, point supervision and inference."""

import unittest

import torch

from lib.networks import EMCADNet
from lib.point_rend import point_loss, training_points


class PointRendTests(unittest.TestCase):
    def test_switch_isolation(self):
        base = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False)
        off = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False,
                       pointrend_enabled=False)
        self.assertEqual(tuple(base.state_dict()), tuple(off.state_dict()))
        self.assertFalse(any("point_rend" in key for key in off.state_dict()))

    def test_train_and_inference_binary_multiclass(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for classes, channels in ((1, 3), (4, 1), (9, 1)):
            with self.subTest(classes=classes):
                model = EMCADNet(num_classes=classes, encoder="pvt_v2_b0",
                                 pretrain=False, pointrend_enabled=True).to(device).eval()
                image = torch.randn(1, channels, 64, 64, device=device)
                outputs, point_logits, coords = model(image, mode="train_point")
                target = torch.randint(0, 2 if classes == 1 else classes,
                                       (1, 64, 64), device=device)
                loss = outputs[-1].square().mean() + point_loss(point_logits, coords, target)
                loss.backward()
                self.assertEqual(tuple(point_logits.shape), (1, classes, 196))
                self.assertTrue(torch.isfinite(loss))
                with torch.no_grad():
                    refined = model(image, mode="test")
                self.assertEqual(tuple(refined[-1].shape), (1, classes, 64, 64))

    def test_sampling_policy(self):
        coords = training_points(torch.randn(2, 4, 16, 16))
        self.assertEqual(tuple(coords.shape), (2, 196, 2))
        self.assertTrue(((coords >= 0) & (coords <= 1)).all())


if __name__ == "__main__":
    unittest.main()

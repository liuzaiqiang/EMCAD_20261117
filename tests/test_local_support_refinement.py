"""Checks for sparse local-support refinement and its supervision."""

import unittest

import torch

from lib.networks import EMCADNet
from lib.local_support_refinement import local_support_scores, point_loss, select_anchors


class LocalSupportRefinementTests(unittest.TestCase):
    def test_switch_isolation(self):
        base = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False)
        off = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False,
                       local_support_refinement=False)
        self.assertEqual(tuple(base.state_dict()), tuple(off.state_dict()))
        self.assertFalse(any("local_support_refiner" in key for key in off.state_dict()))

    def test_train_and_inference_binary_multiclass(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for classes, channels in ((1, 3), (1, 1), (4, 1), (9, 1)):
            with self.subTest(classes=classes, channels=channels):
                model = EMCADNet(num_classes=classes, encoder="pvt_v2_b0",
                                 pretrain=False, local_support_refinement=True).to(device).eval()
                image = torch.randn(2, channels, 64, 96, device=device)
                target = torch.randint(0, max(2, classes), (2, 64, 96), device=device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    _, point_logits, coords = model(image, mode="train_point")
                    loss = point_loss(point_logits, coords, target)
                loss.backward()
                self.assertEqual(tuple(point_logits.shape), (2, classes, 32 * 25))
                self.assertEqual(tuple(coords.shape), (2, 32 * 25, 2))
                self.assertTrue(torch.isfinite(loss))
                projection = model.local_support_refiner.correction[-1]
                self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    baseline = model(image, mode="train")
                    initial = model(image, mode="test")
                    torch.testing.assert_close(initial[-1], baseline[-1])
                    projection.bias[0].fill_(10)
                    refined = model(image, mode="test")
                    delta = refined[-1] - baseline[-1]
                    self.assertEqual(tuple(refined[-1].shape), (2, classes, 64, 96))
                    self.assertTrue(torch.isfinite(delta).all())
                    self.assertTrue(0 < delta.abs().max().item() <= 3.00001)
                    changed = delta.ne(0).any(dim=1).sum(dim=(1, 2))
                    self.assertTrue(((changed > 128) & (changed <= 128 * 25)).all())
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline[:-1], refined[:-1])))

    def test_support_prioritizes_a_thin_structure_over_an_equally_confident_edge(self):
        logits = torch.full((1, 1, 16, 24), -3.0)
        logits[:, :, 3:13, 4] = 3.0
        logits[:, :, 2:14, 12:22] = 3.0
        score, uncertainty = local_support_scores(logits)
        self.assertEqual(uncertainty[0, 0, 8, 4], uncertainty[0, 0, 8, 12])
        self.assertGreater(score[0, 0, 8, 4], score[0, 0, 8, 12])
        self.assertEqual(score[0, 0, 8, 8], 0.0)

    def test_anchors_use_distinct_tiles_including_odd_edges(self):
        scores = torch.rand(2, 1, 9, 13)
        anchors = select_anchors(scores, 128, exploration=8)
        self.assertEqual(tuple(anchors.shape), (2, 35))
        y = torch.div(anchors, 13, rounding_mode="floor")
        x = anchors % 13
        tiles = torch.div(y, 2, rounding_mode="floor") * 7 + torch.div(x, 2, rounding_mode="floor")
        expected = torch.arange(35).expand_as(tiles)
        self.assertTrue(torch.equal(tiles.sort(dim=1).values, expected))


if __name__ == "__main__":
    unittest.main()

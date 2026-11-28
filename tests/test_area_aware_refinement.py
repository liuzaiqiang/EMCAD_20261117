"""Checks for optional area-aware refinement, point supervision and inference."""

import unittest

import torch

from lib.networks import EMCADNet
from lib.area_aware_refinement import area_aware_indices, foreground_rarity, point_loss, stratified_points


class AreaAwareRefinementTests(unittest.TestCase):
    def test_switch_isolation(self):
        base = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False)
        off = EMCADNet(num_classes=1, encoder="pvt_v2_b0", pretrain=False,
                       area_aware_refinement=0)
        self.assertEqual(tuple(base.state_dict()), tuple(off.state_dict()))
        self.assertFalse(any("area_aware_refiner" in key for key in off.state_dict()))

    def test_area_aware_training_and_inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for classes, channels in ((1, 3), (1, 1), (4, 1), (9, 1)):
            with self.subTest(classes=classes, channels=channels):
                model = EMCADNet(num_classes=classes, encoder="pvt_v2_b0", pretrain=False,
                                 area_aware_refinement=1).to(device).eval()
                image = torch.randn(2, channels, 64, 96, device=device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs, points, coords = model(image, mode="train_point")
                    target = torch.randint(0, max(2, classes), (2, 64, 96), device=device)
                    loss = point_loss(points, coords, target)
                loss.backward()
                self.assertEqual(tuple(points.shape), (2, classes, 196))
                self.assertEqual(tuple(coords.shape), (2, 196, 2))
                self.assertTrue(torch.isfinite(loss))
                final_layer = model.area_aware_refiner.correction[-1]
                self.assertGreater(final_layer.weight.grad.abs().sum().item(), 0)

                with torch.no_grad():
                    base = model(image, mode="train")
                    initial = model(image, mode="test")
                    torch.testing.assert_close(initial[-1], base[-1])
                    self.assertEqual(tuple(initial[-1].shape), (2, classes, 64, 96))
                    final_layer.bias[-1].fill_(0.1)
                    refined = model(image, mode="test")
                    self.assertTrue(torch.isfinite(refined[-1]).all())
                    changed = refined[-1].ne(initial[-1]).any(dim=1).sum(dim=(1, 2))
                    self.assertEqual(changed.tolist(), [1568, 1568])
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(base[:-1], refined[:-1])))
                    self.assertEqual(tuple(outputs[-1].shape), tuple(refined[-1].shape))

    def test_area_aware_selection_keeps_uncertain_points_and_checks_cross_scale_evidence(self):
        fine = torch.tensor([[[0.0, -0.7, -0.7, 5.0]]])
        guide = torch.tensor([[[0.0, -4.0, 4.0, 5.0]]])
        rarity = foreground_rarity(fine.unsqueeze(2), guide.unsqueeze(2))
        self.assertEqual(rarity[0, 0].item(), 0.0)
        self.assertGreater(rarity[0, 1].item(), 0.0)
        self.assertEqual(area_aware_indices(fine, guide, rarity, 2).tolist(), [[0, 2]])
        small_foreground = torch.tensor([[[[3.0, -3.0, -3.0, -3.0]]]])
        large_foreground = torch.full_like(small_foreground, 3.0)
        absent = torch.full_like(small_foreground, -3.0)
        small_rarity = foreground_rarity(small_foreground, small_foreground)
        large_rarity = foreground_rarity(large_foreground, large_foreground)
        self.assertGreater(small_rarity[0, 1].item(), large_rarity[0, 1].item())
        self.assertEqual(foreground_rarity(absent, absent)[0, 1].item(), 0.0)

    def test_stratified_exploration_covers_every_tile(self):
        coords = stratified_points(2, 7, torch.device("cpu"))
        self.assertEqual(tuple(coords.shape), (2, 49, 2))
        self.assertTrue(((coords >= 0) & (coords < 1)).all())
        tiles = (coords * 7).long()
        self.assertTrue(torch.equal(tiles[..., 0] + 7 * tiles[..., 1],
                                    torch.arange(49).expand(2, -1)))


if __name__ == "__main__":
    unittest.main()

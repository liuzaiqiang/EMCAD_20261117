"""Mechanism and ablation smoke checks; these do not establish clinical gains."""

import unittest

import torch
import torch.nn.functional as F

from lib.decoders import ContentAwareAntiAliasUpsample, EMCAD
from lib.cepa import CEPA
from utils.area_adaptive_tversky import area_adaptive_tversky_loss


class ResearchComponentsTests(unittest.TestCase):
    def test_upsample_support_and_gradient(self):
        module = ContentAwareAntiAliasUpsample(8)
        x = torch.randn(1, 8, 8, 8, requires_grad=True)
        support = torch.rand(1, 1, 8, 8, requires_grad=True)
        output = module(x, support)
        self.assertEqual(output.shape, (1, 8, 16, 16))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(support.grad).all())
        self.assertIsNotNone(module.content_gate[0].weight.grad)
        with self.assertRaises(ValueError):
            module(x, None)

    def test_directional_boundary_filters_isolated_noise(self):
        # One-sided continuity is stronger than an isolated dissimilar center.
        planar = torch.ones(1, 2, 7, 7)
        planar[:, 0, :, 4:] = -1
        isolated = torch.ones_like(planar)
        isolated[:, 0, 3, 3] = -1
        edge = CEPA._directional_support(planar)
        noise = CEPA._directional_support(isolated)
        self.assertGreater(edge[0, 0, 3, 3].item(), noise[0, 0, 3, 3].item())

    def test_area_loss_binary_multiclass_empty_and_batch_one(self):
        binary_logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
        small = torch.zeros(1, 16, 16, dtype=torch.long)
        small[:, 3:5, 3:5] = 1
        large = torch.zeros_like(small)
        large[:, 2:14, 2:14] = 1
        small_loss = area_adaptive_tversky_loss(binary_logits, small)
        large_loss = area_adaptive_tversky_loss(binary_logits, large)
        self.assertGreater(small_loss.item(), large_loss.item())
        small_loss.backward()
        self.assertTrue(torch.isfinite(binary_logits.grad).all())
        empty = torch.zeros_like(small)
        empty_loss = area_adaptive_tversky_loss(binary_logits, empty)
        self.assertEqual(empty_loss.item(), 0.0)
        binary_logits.grad = None
        empty_loss.backward()
        self.assertTrue(torch.equal(binary_logits.grad, torch.zeros_like(binary_logits.grad)))
        multiclass = torch.randn(1, 4, 16, 16, requires_grad=True)
        multi_loss = area_adaptive_tversky_loss(multiclass, small)
        multi_loss.backward()
        self.assertTrue(torch.isfinite(multiclass.grad).all())

    def test_all_decoder_switch_combinations_and_reload(self):
        channels = [64, 32, 16, 8]
        x = torch.randn(1, 64, 4, 4)
        skips = [torch.randn(1, 32, 8, 8), torch.randn(1, 16, 16, 16),
                 torch.randn(1, 8, 32, 32)]
        for caa in ('off', 'caa'):
            for cepa in (0, 1):
                model = EMCAD(channels=channels, caa_mode=caa, cepa_enabled=cepa).eval()
                with torch.no_grad():
                    expected = model(x, skips)[-1]
                restored = EMCAD(channels=channels, caa_mode=caa, cepa_enabled=cepa).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    actual = restored(x, skips)[-1]
                self.assertTrue(torch.equal(expected, actual))
                if caa == 'off' and not cepa:
                    self.assertFalse(any('cepa' in k or 'anti_alias' in k or 'content_gate' in k
                                         for k in model.state_dict()))

    def test_eight_train_switch_combinations(self):
        channels = [32, 16, 8, 4]
        images = torch.randn(1, 32, 4, 4)
        skips = [torch.randn(1, 16, 8, 8), torch.randn(1, 8, 16, 16),
                 torch.randn(1, 4, 32, 32)]
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[:, 10:14, 10:14] = 1
        for caa in ('off', 'caa'):
            for cepa in (0, 1):
                for area in (0, 1):
                    model = EMCAD(channels=channels, caa_mode=caa, cepa_enabled=cepa)
                    head = torch.nn.Conv2d(4, 1, 1)
                    logits = head(model(images, skips)[-1])
                    loss = F.binary_cross_entropy_with_logits(logits[:, 0], target.float())
                    if area:
                        loss = loss + area_adaptive_tversky_loss(logits, target)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(model.eucb1.pwc[0].weight.grad).all())


if __name__ == '__main__':
    unittest.main()

"""End-to-end EMCAD/CAA/CEPA synthetic GPU smoke tests."""

import unittest

import torch
import torch.nn.functional as F

from lib.networks import EMCADNet


class CEPAModelTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_training_and_strict_reload_for_task_shapes(self):
        for classes, side, input_channels, encoder in (
            (9, 224, 1, "pvt_v2_b2"),
            (4, 256, 1, "pvt_v2_b0"),
            (1, 352, 3, "pvt_v2_b0"),
        ):
            model = EMCADNet(num_classes=classes, encoder=encoder,
                             pretrain=False, caa_mode="caa", cepa_enabled=1).cuda()
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            images = torch.randn(1, input_channels, side, side, device="cuda")
            target = torch.zeros(1, side, side, device="cuda", dtype=torch.long)
            target[:, side // 4:side // 2, side // 4:side // 2] = 1
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images, mode="train")
                self.assertEqual(len(outputs), 4)
                self.assertTrue(all(output.shape == (1, classes, side, side) for output in outputs))
                if classes == 1:
                    loss = F.binary_cross_entropy_with_logits(outputs[-1][:, 0], target.float())
                else:
                    loss = F.cross_entropy(outputs[-1], target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(loss))
            model.eval()
            with torch.no_grad():
                expected = model(images, mode="test")[-1]
            restored = EMCADNet(num_classes=classes, encoder=encoder,
                                pretrain=False, caa_mode="caa", cepa_enabled=1).cuda()
            restored.load_state_dict(model.state_dict(), strict=True)
            restored.eval()
            with torch.no_grad():
                actual = restored(images, mode="test")[-1]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))
            del model, restored, outputs, expected, actual, optimizer
            torch.cuda.empty_cache()

    def test_off_has_no_cepa_parameters(self):
        model = EMCADNet(num_classes=1, encoder="pvt_v2_b0",
                         pretrain=False, caa_mode="off", cepa_enabled=0)
        self.assertFalse(any("cepa" in name for name in model.state_dict()))

    def test_on_starts_from_identical_baseline_forward(self):
        baseline = EMCADNet(num_classes=1, encoder="pvt_v2_b0",
                            pretrain=False, caa_mode="caa", cepa_enabled=0).eval()
        adapted = EMCADNet(num_classes=1, encoder="pvt_v2_b0",
                           pretrain=False, caa_mode="caa", cepa_enabled=1).eval()
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all("cepa" in name for name in missing))
        self.assertFalse(unexpected)
        image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            before = baseline(image)[-1]
            after = adapted(image)[-1]
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()

"""Run with: python -m unittest discover -s tests -p test_wavelet_detail.py -v.

These are numerical/structural regression checks, not segmentation experiments.
Historical comparisons require this repository's existing local Git objects.
"""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

from lib.decoders import ContentAwareAntiAliasUpsample, EMCAD
from lib.networks import EMCADNet
from lib.wavelet_detail import (
    DirectionalWaveletDetailCompensation,
    haar_analysis,
    haar_synthesis,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REF = "f70e6b0c06b4f5b6a4408b43b0c88be05608ac5b"
PRE_EDIT_REF = "768974840d2c4fd705162def6e3ad7d98c6ed5a1"


def historical_model_class(ref):
    def source(path):
        return subprocess.check_output([
            "git", "-c", "safe.directory=" + ROOT.as_posix(),
            "show", ref + ":" + path,
        ], cwd=str(ROOT)).decode("utf-8-sig")

    decoder = types.ModuleType("_wavelet_reference_decoder")
    exec(compile(source("lib/decoders.py"), ref + ":lib/decoders.py", "exec"), decoder.__dict__)
    network = types.ModuleType("_wavelet_reference_network")
    code = source("lib/networks.py").replace(
        "from lib.decoders import EMCAD", "from _wavelet_reference_decoder import EMCAD")
    with patch.dict(sys.modules, {"_wavelet_reference_decoder": decoder}):
        exec(compile(code, ref + ":lib/networks.py", "exec"), network.__dict__)
    return network.EMCADNet


class WaveletMathTests(unittest.TestCase):
    def test_haar_roundtrip_and_energy(self):
        for height, width in [(14, 14), (7, 9), (1, 1), (1, 5), (5, 1)]:
            x = torch.randn(2, 4, height, width, dtype=torch.float64)
            ll, bands, size = haar_analysis(x)
            torch.testing.assert_close(haar_synthesis(ll, bands, size), x, atol=1e-14, rtol=1e-14)
            if height % 2 == width % 2 == 0:
                torch.testing.assert_close(x.square().sum(), ll.square().sum() + bands.square().sum())

    def test_directions_and_pixel_alignment(self):
        for index, values in enumerate([
            [[1., 1.], [-1., -1.]],
            [[1., -1.], [1., -1.]],
            [[1., -1.], [-1., 1.]],
        ]):
            x = torch.tensor(values).reshape(1, 1, 2, 2)
            ll, bands, size = haar_analysis(x)
            self.assertEqual(ll.item(), 0.)
            expected = torch.zeros_like(bands)
            expected[:, :, index] = 2.
            self.assertTrue(torch.equal(bands, expected))
            self.assertTrue(torch.equal(haar_synthesis(torch.zeros_like(ll), bands, size), x))

    def test_constant_skip_produces_no_correction(self):
        module = DirectionalWaveletDetailCompensation(4)
        module.residual_scale.data.fill_(0.5)
        skip = torch.ones(2, 4, 7, 9)
        up = torch.randn_like(skip)
        self.assertTrue(torch.equal(module(up, skip), up))

    def test_zero_initialization_and_gradients(self):
        torch.manual_seed(17)
        module = DirectionalWaveletDetailCompensation(4)
        up = torch.randn(2, 4, 7, 9, requires_grad=True)
        skip = torch.randn_like(up, requires_grad=True)
        original = skip.detach().clone()
        self.assertTrue(torch.equal(module(up, skip), up))
        module(up, skip).square().mean().backward()
        self.assertGreater(module.residual_scale.grad.abs().item(), 0.)
        # At nonzero residual scale the threshold, gates, projection and raw skip
        # must all learn; zero scale intentionally delays them by one update.
        module.zero_grad(set_to_none=True)
        module.residual_scale.data.fill_(0.2)
        module(up, skip).square().mean().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0., name)
        self.assertTrue(torch.isfinite(skip.grad).all())
        self.assertGreater(skip.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.equal(original, skip.detach()))

    def test_invalid_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            DirectionalWaveletDetailCompensation(4)(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 7, 8))


class ModelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, cls=EMCADNet, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(num_classes=kwargs.pop("num_classes", 3), pretrain=False, **kwargs)

    def assert_state_equal(self, a, b):
        self.assertEqual(list(a), list(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_both_off_match_historical_baseline(self):
        reference_cls = historical_model_class(BASELINE_REF)
        # PVT-B2 is the actual Synapse encoder; compare all model weights,
        # buffers, RNG state, four logits, training gradients and BN updates.
        torch.manual_seed(2222)
        reference = self.make_model(reference_cls, encoder="pvt_v2_b2", activation="relu6")
        reference_rng = torch.get_rng_state()
        torch.manual_seed(2222)
        current = self.make_model(encoder="pvt_v2_b2", activation="relu6", caa_mode="off", use_wavelet_detail=0)
        self.assertTrue(torch.equal(reference_rng, torch.get_rng_state()))
        self.assert_state_equal(reference.state_dict(), current.state_dict())
        current.load_state_dict(reference.state_dict(), strict=True)
        self.assertFalse(any(isinstance(m, (DirectionalWaveletDetailCompensation, ContentAwareAntiAliasUpsample))
                             for m in current.modules()))
        images = torch.randn(2, 1, 64, 64)
        reference.eval()
        current.eval()
        with torch.no_grad():
            for a, b in zip(reference(images), current(images)):
                self.assertTrue(torch.equal(a, b))
        reference.train()
        current.train()
        torch.manual_seed(42)
        outputs_ref = reference(images)
        torch.manual_seed(42)
        outputs_cur = current(images)
        for a, b in zip(outputs_ref, outputs_cur):
            self.assertTrue(torch.equal(a, b))
        sum(p.square().mean() for p in outputs_ref).backward()
        sum(p.square().mean() for p in outputs_cur).backward()
        for (name, a), (_, b) in zip(reference.named_parameters(), current.named_parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad, name)
            else:
                self.assertTrue(torch.equal(a.grad, b.grad), name)
        self.assert_state_equal(reference.state_dict(), current.state_dict())

    def test_both_off_binary_and_acdc_heads_match_baseline(self):
        reference_cls = historical_model_class(BASELINE_REF)
        for classes in (1, 4):
            with self.subTest(num_classes=classes):
                torch.manual_seed(2222)
                reference = self.make_model(reference_cls, num_classes=classes,
                                            encoder="pvt_v2_b2", activation="relu6").eval()
                torch.manual_seed(2222)
                current = self.make_model(num_classes=classes, encoder="pvt_v2_b2",
                                          activation="relu6", caa_mode="off",
                                          use_wavelet_detail=0).eval()
                self.assert_state_equal(reference.state_dict(), current.state_dict())
                current.load_state_dict(reference.state_dict(), strict=True)
                images = torch.randn(1, 3 if classes == 1 else 1, 64, 64)
                with torch.no_grad():
                    for a, b in zip(reference(images), current(images)):
                        self.assertEqual(a.shape[1], classes)
                        self.assertTrue(torch.equal(a, b))

    def test_wavelet_off_preserves_pre_edit_caa(self):
        reference_cls = historical_model_class(PRE_EDIT_REF)
        torch.manual_seed(2222)
        reference = self.make_model(reference_cls, encoder="pvt_v2_b0", caa_mode="caa")
        torch.manual_seed(2222)
        current = self.make_model(encoder="pvt_v2_b0", caa_mode="caa", use_wavelet_detail=0)
        self.assert_state_equal(reference.state_dict(), current.state_dict())
        reference.eval()
        current.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 64)
            for a, b in zip(reference(x), current(x)):
                self.assertTrue(torch.equal(a, b))

    def test_four_ablations_and_original_raw_skips(self):
        for caa_mode in ["off", "caa"]:
            torch.manual_seed(123)
            plain = self.make_model(encoder="pvt_v2_b0", caa_mode=caa_mode)
            torch.manual_seed(123)
            model = self.make_model(encoder="pvt_v2_b0", caa_mode=caa_mode, use_wavelet_detail=1)
            # Creating the new branch must not shift original weights/heads.
            for key, value in plain.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            model.train()
            records = {}
            hooks = [model.backbone.register_forward_hook(lambda m, a, out: records.update(encoder=out))]
            for index, block in enumerate(model.decoder.wavelet_details):
                hooks.append(block.register_forward_pre_hook(
                    lambda m, args, index=index: records.update({index: args[1]})))
                block.residual_scale.data.fill_(0.1)
            outputs = model(torch.randn(2, 3, 64, 64))
            self.assertEqual(len(outputs), 4)
            for prediction in outputs:
                self.assertEqual(tuple(prediction.shape), (2, 3, 64, 64))
                self.assertTrue(torch.isfinite(prediction).all())
            for index in range(3):
                self.assertIs(records[index], records["encoder"][2 - index])
            sum(out.square().mean() for out in outputs).backward()
            for block in model.decoder.wavelet_details:
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0., name)
            for hook in hooks:
                hook.remove()
            plain.eval()
            with torch.no_grad():
                self.assertTrue(all(torch.isfinite(x).all() for x in plain(torch.randn(1, 3, 64, 64))))

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(5)
        model = EMCAD(channels=[32, 24, 16, 8], expansion_factor=2,
                      caa_mode="caa", use_wavelet_detail=1).eval()
        for block in model.wavelet_details:
            block.residual_scale.data.fill_(0.2)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = EMCAD(channels=[32, 24, 16, 8], expansion_factor=2,
                         caa_mode="caa", use_wavelet_detail=1).eval()
        restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
        x = torch.randn(1, 32, 2, 2)
        skips = [torch.randn(1, 24, 4, 4), torch.randn(1, 16, 8, 8), torch.randn(1, 8, 16, 16)]
        with torch.no_grad():
            for a, b in zip(model(x, skips), restored(x, skips)):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()

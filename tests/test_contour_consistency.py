"""CPU regression tests; no dataset downloads or training claims.

Run: python -m unittest discover -s tests -p test_contour_consistency.py -v
"""
import ast
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import types
import unittest

import torch

from lib.contour_consistency import MultiScaleContourConsistency
from lib.networks import EMCADNet

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def historical_model(ref):
    """Load committed code without changing checkout or touching git metadata."""
    modules = []
    for path in ('lib/decoders.py', 'lib/networks.py'):
        source = subprocess.check_output([
            'git', '-c', 'safe.directory=' + ROOT.as_posix(), 'show', ref + ':' + path
        ], cwd=ROOT).decode('utf-8-sig')
        module = types.ModuleType('_historical_' + path.split('/')[-1][:-3])
        if path.endswith('networks.py'):
            source = source.replace('from lib.decoders import EMCAD',
                                    'from _historical_decoders import EMCAD')
        exec(compile(source, ref + ':' + path, 'exec'), module.__dict__)
        sys.modules[module.__name__] = module
        modules.append(module)
    return modules[-1].EMCADNet


def make(cls=EMCADNet, **kwargs):
    torch.manual_seed(2026)
    with contextlib.redirect_stdout(io.StringIO()):
        model = cls(num_classes=9, pretrain=False, **kwargs)
    return model


class ContourTests(unittest.TestCase):
    def test_zero_start_can_learn(self):
        m = MultiScaleContourConsistency([8, 4, 2])
        xs = [torch.randn(2, c, s, s) for c, s in [(8, 4), (4, 8), (2, 16)]]
        optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
        sum(y.square().mean() for y in m(xs)).backward()
        self.assertTrue(torch.isfinite(m.strength.grad).all())
        self.assertGreater(m.strength.grad.abs().sum(), 0)
        optimizer.step()
        optimizer.zero_grad()
        sum(y.square().mean() for y in m(xs)).backward()
        for branch in m.feedback:
            self.assertGreater(sum(p.grad.abs().sum() for p in branch.parameters()), 0)

    def test_constant_edge_and_odd_shapes(self):
        m = MultiScaleContourConsistency([8, 4, 2])
        self.assertLess(m.contour(torch.ones(1, 8, 7, 9), (7, 9)).abs().max(), 1e-5)
        step = torch.zeros(1, 8, 7, 9)
        step[..., 4:] = 1
        edge = m.contour(step, (7, 9))
        self.assertGreater(edge[..., 3:5].mean(), edge[..., :2].mean())
        xs = [torch.randn(2, c, h, w, requires_grad=True)
              for c, h, w in [(8, 5, 7), (4, 9, 13), (2, 17, 25)]]
        for x, y in zip(xs, m(xs)):
            self.assertTrue(torch.equal(x, y))
        with torch.no_grad():
            m.strength.fill_(0.2)
        ys = m(xs)
        changed = m([xs[0], xs[1], xs[2].flip(-1)])
        self.assertFalse(torch.equal(ys[0], changed[0]))
        sum(y.square().mean() for y in ys).backward()
        for parameter in m.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)
        for x in xs:
            self.assertTrue(torch.isfinite(x.grad).all())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(all(torch.isfinite(y).all() for y in m(xs)))

    def test_historical_off_and_caa_regression(self):
        # Latest baseline commit before CAA restoration; also compare pre-edit HEAD.
        for ref, caa in [('8bb7d52', 'off'), ('7adef9d', 'off'), ('7adef9d', 'caa')]:
            cls = historical_model(ref)
            old = make(cls, **({'caa_mode': caa} if ref == '7adef9d' else {}))
            old_rng = torch.get_rng_state().clone()
            new = make(caa_mode=caa, use_contour_consistency=0)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for name, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
            self.assertFalse(hasattr(new.decoder, 'contour_consistency'))
            x = torch.randn(1, 1, 64, 64)
            old.eval(); new.eval()
            with torch.no_grad():
                for a, b in zip(old(x), new(x)):
                    self.assertTrue(torch.equal(a, b), ref)
            # Training-mode output, gradients and running statistics regression.
            old.train(); new.train()
            torch.manual_seed(51)  # PVT stochastic depth must see the same RNG.
            old_outputs = old(x)
            old_outputs[-1].square().mean().backward()
            torch.manual_seed(51)
            new_outputs = new(x)
            new_outputs[-1].square().mean().backward()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_outputs, new_outputs)))
            for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
                if a.grad is not None:
                    self.assertTrue(torch.equal(a.grad, b.grad), name)
            for name, a in old.named_buffers():
                self.assertTrue(torch.equal(a, dict(new.named_buffers())[name]), name)

    def test_four_ablations_and_real_input_sizes(self):
        for caa in ['off', 'caa']:
            off = make(caa_mode=caa)
            rng = torch.get_rng_state().clone()
            on = make(caa_mode=caa, use_contour_consistency=1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in off.state_dict().items():
                self.assertTrue(torch.equal(value, on.state_dict()[name]), name)
            off.eval(); on.eval()
            with torch.no_grad():
                x = torch.randn(1, 1, 64, 64)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(off(x), on(x))))
            clone = make(caa_mode=caa, use_contour_consistency=1)
            clone.load_state_dict(on.state_dict(), strict=True)
        del off, clone
        for size, channels, classes in [(224, 1, 9), (224, 1, 4),
                                         (256, 3, 1), (352, 3, 1)]:
            with contextlib.redirect_stdout(io.StringIO()):
                model = EMCADNet(num_classes=classes, pretrain=False, caa_mode='caa',
                                 use_contour_consistency=1).eval()
            model.decoder.contour_consistency.strength.data.fill_(0.1)
            outputs = model(torch.randn(1, channels, size, size))
            self.assertEqual([tuple(x.shape) for x in outputs], [(1, classes, size, size)]*4)
            sum(x.square().mean() for x in outputs).backward()
            for p in model.decoder.contour_consistency.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            del model, outputs


if __name__ == '__main__':
    unittest.main()

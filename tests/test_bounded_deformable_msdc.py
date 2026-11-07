"""Checks for the optional d2/d1 bounded deformable branches."""

import unittest

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class BoundedDeformableMSDCTests(unittest.TestCase):
    def test_zero_offset_matches_regular_depthwise_convolution(self):
        from lib.bounded_deformable import BoundedDeformableDepthwiseConv

        for kernel_size in (3, 5):
            layer = BoundedDeformableDepthwiseConv(4, kernel_size)
            x = torch.randn(2, 4, 7, 9, requires_grad=True)
            expected = F.conv2d(x, layer.weight, padding=kernel_size // 2, groups=4)
            actual = layer(x)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            actual.square().mean().backward()
            self.assertIsNotNone(layer.offset_gain.grad)
            self.assertIsNotNone(layer.weight.grad)

    def test_disabled_is_baseline_and_enabled_only_at_d2_d1(self):
        from lib.bounded_deformable import BoundedDeformableDepthwiseConv
        from lib.decoders import EMCAD, ContentAwareAntiAliasUpsample

        common = dict(channels=[128, 64, 32, 16], kernel_sizes=[1, 3, 5],
                      expansion_factor=2, caa_mode="off")
        baseline = EMCAD(**common)
        disabled = EMCAD(**common, bounded_deformable_msdc=False)
        enabled = EMCAD(**common, bounded_deformable_msdc=True)
        self.assertEqual(list(baseline.state_dict()), list(disabled.state_dict()))
        self.assertFalse(any(isinstance(m, BoundedDeformableDepthwiseConv)
                             for m in disabled.modules()))
        self.assertFalse(any(isinstance(m, ContentAwareAntiAliasUpsample)
                             for m in disabled.modules()))
        for stage in (enabled.mscb4, enabled.mscb3):
            self.assertFalse(any(isinstance(m, BoundedDeformableDepthwiseConv)
                                 for m in stage.modules()))
        for stage in (enabled.mscb2, enabled.mscb1):
            self.assertEqual(sum(isinstance(m, BoundedDeformableDepthwiseConv)
                                 for m in stage.modules()), 2)


if __name__ == "__main__":
    unittest.main()

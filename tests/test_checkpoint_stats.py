"""Regression checks for filtering THOP statistics (no PyTorch required)."""

import unittest
from collections import OrderedDict

from utils.checkpoint_stats import without_profiling_stats


class CheckpointStatsTests(unittest.TestCase):
    def test_save_preserves_weights_buffers_metadata_and_source(self):
        weight, running_mean = object(), object()
        state = OrderedDict([
            ("weight", weight), ("bn.running_mean", running_mean),
            ("total_ops", 1), ("module.backbone.total_params", 2),
            ("total_ops_extra", 3), ("unexpected_weight", 4),
        ])
        state._metadata = {"": {"version": 1}}
        clean = without_profiling_stats(state)
        self.assertEqual(list(clean), [
            "weight", "bn.running_mean", "total_ops_extra", "unexpected_weight",
        ])
        self.assertIs(clean["weight"], weight)
        self.assertIs(clean["bn.running_mean"], running_mean)
        self.assertEqual(clean._metadata, state._metadata)
        self.assertEqual(len(state), 6)

    def test_load_retains_expected_stats_and_unrelated_mismatches(self):
        state = {"weight": 1, "total_ops": 2, "block.total_params": 3,
                 "unexpected_weight": 4}
        clean = without_profiling_stats(state, {"weight", "total_ops"})
        self.assertEqual(clean, {"weight": 1, "total_ops": 2,
                                 "unexpected_weight": 4})

    def test_clean_checkpoint_is_unchanged(self):
        state = {"weight": object(), "bn.num_batches_tracked": object()}
        self.assertEqual(without_profiling_stats(state), state)


if __name__ == "__main__":
    unittest.main()

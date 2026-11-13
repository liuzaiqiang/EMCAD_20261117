"""Online top-K selection uses the existing per-epoch Dice before costly metrics."""

import unittest

from utils.synapse_candidate_selection import (
    add_topk_entry, candidate_checkpoint_name, is_new_best_dice,
    should_enter_topk, validate_topk_config,
)


class SynapseCandidateSelectionTests(unittest.TestCase):
    def test_historical_dice_record_requires_strict_improvement(self):
        self.assertTrue(is_new_best_dice(0.8401, 0.8400))
        self.assertFalse(is_new_best_dice(0.8400, 0.8400))
        self.assertFalse(is_new_best_dice(0.8399, 0.8400))

    def test_threshold_and_online_board_rule(self):
        board = [{"epoch": 10, "rank_dice": 0.84}]
        self.assertFalse(should_enter_topk(0.8389, board, 2, 83.90))
        self.assertTrue(should_enter_topk(0.8390, board, 2, 83.90))
        board.append({"epoch": 11, "rank_dice": 0.839})
        self.assertFalse(should_enter_topk(0.8390, board, 2, 83.90))
        self.assertTrue(should_enter_topk(0.8391, board, 2, 83.90))

    def test_displaced_entry_and_tie_order(self):
        board = [{"epoch": 10, "rank_dice": 0.84},
                 {"epoch": 11, "rank_dice": 0.839}]
        new_board, evicted = add_topk_entry(board, {"epoch": 12, "rank_dice": 0.841}, 2)
        self.assertEqual([item["epoch"] for item in new_board], [12, 10])
        self.assertEqual([item["epoch"] for item in evicted], [11])
        tied_board, _ = add_topk_entry(board[:1], {"epoch": 12, "rank_dice": 0.84}, 2)
        self.assertEqual([item["epoch"] for item in tied_board], [10, 12])

    def test_online_board_matches_final_top_ten_after_300_epochs(self):
        scores = [0.80 + ((epoch * 37) % 100) / 1000 for epoch in range(1, 301)]
        board = []
        for epoch, score in enumerate(scores, start=1):
            if should_enter_topk(score, board, 10, 83.90):
                board, _ = add_topk_entry(
                    board, {"epoch": epoch, "rank_dice": score}, 10)
        eligible = [{"epoch": epoch, "rank_dice": score}
                    for epoch, score in enumerate(scores, start=1) if score >= 0.839]
        expected = sorted(eligible, key=lambda item: (-item["rank_dice"], item["epoch"]))[:10]
        self.assertEqual(board, expected)

    def test_invalid_configuration_fails_early(self):
        validate_topk_config(0, 0, float("nan"))
        for value in (-1, 101, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_topk_config(1, 10, value)
        for enabled, topk in ((2, 10), (-1, 10), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                validate_topk_config(enabled, topk, 83.90)

    def test_filename_identifies_epoch_and_validation_metrics(self):
        self.assertEqual(
            candidate_checkpoint_name(123, 0.83904, 12.345),
            "epoch_123_valdice_83.9040_valhd95_12.3450.pth",
        )
        self.assertNotEqual(
            candidate_checkpoint_name(123, 0.83904, 12.345),
            candidate_checkpoint_name(124, 0.83904, 12.345),
        )


if __name__ == "__main__":
    unittest.main()

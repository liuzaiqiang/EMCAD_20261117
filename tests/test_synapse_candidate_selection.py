"""Verify the Synapse TopN threshold, ranking, and metric-rich filenames."""

import unittest

from utils.synapse_candidate_selection import (
    add_topk_entry, candidate_checkpoint_name, should_enter_topk,
)


class SynapseCandidateSelectionTests(unittest.TestCase):
    def test_threshold_then_current_board_controls_full_validation(self):
        board = [{"epoch": 10, "rank_dice": 0.8420}]
        self.assertFalse(should_enter_topk(0.8409, board, 2, 84.10))
        self.assertTrue(should_enter_topk(0.8410, board, 2, 84.10))
        board.append({"epoch": 11, "rank_dice": 0.8410})
        self.assertFalse(should_enter_topk(0.8410, board, 2, 84.10))
        self.assertTrue(should_enter_topk(0.8411, board, 2, 84.10))

    def test_replacement_keeps_best_rank_dice_and_exposes_both_metrics(self):
        board = [{"epoch": 10, "rank_dice": 0.8420},
                 {"epoch": 11, "rank_dice": 0.8410}]
        updated, evicted = add_topk_entry(
            board, {"epoch": 12, "rank_dice": 0.8430}, 2)
        self.assertEqual([entry["epoch"] for entry in updated], [12, 10])
        self.assertEqual([entry["epoch"] for entry in evicted], [11])
        self.assertEqual(
            candidate_checkpoint_name(237, 0.840152, 21.6241),
            "epoch_237_valdice_84.0152_valhd95_21.6241.pth",
        )


if __name__ == "__main__":
    unittest.main()

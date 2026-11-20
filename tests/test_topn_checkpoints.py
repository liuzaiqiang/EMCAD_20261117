"""Check that Top-N weights share the run directory with best.pth."""

import json
import os
import tempfile
import unittest

from utils.topn_checkpoints import TopNCheckpoints


class TopNCheckpointsTests(unittest.TestCase):
    def test_weights_are_saved_beside_manifest_and_evicted_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            topn = TopNCheckpoints(directory, size=2)

            def save_checkpoint(path):
                with open(path, "wb") as stream:
                    stream.write(b"checkpoint")

            topn.record(1, 0.70, 12.0, save_checkpoint)
            topn.record(2, 0.80, 10.0, save_checkpoint)
            topn.record(3, 0.90, 8.0, save_checkpoint)

            with open(os.path.join(directory, "topn_manifest.json"), encoding="utf-8") as stream:
                manifest = json.load(stream)
            checkpoints = [entry["checkpoint"] for entry in manifest["entries"]]
            self.assertEqual(checkpoints, [
                "epoch_0003_dice_90.0000.pth",
                "epoch_0002_dice_80.0000.pth",
            ])
            self.assertFalse(os.path.exists(os.path.join(directory, "topn_weights")))
            self.assertFalse(os.path.exists(os.path.join(directory, "epoch_0001_dice_70.0000.pth")))
            for filename in checkpoints:
                self.assertTrue(os.path.isfile(os.path.join(directory, filename)))


if __name__ == "__main__":
    unittest.main()

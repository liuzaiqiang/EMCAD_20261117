"""Check Dice ranking and HD95 history without requiring image data."""

import csv
import json
import os
import tempfile
import unittest

from lib.topn_checkpoints import TopNCheckpoints


class TopNCheckpointTests(unittest.TestCase):
    def test_retain_best_and_record_every_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            def save(path):
                with open(path, "wb") as stream:
                    stream.write(b"checkpoint")

            topn = TopNCheckpoints(directory, 2, save)
            for epoch, dice, hd95 in ((1, 0.7, 8.0), (2, 0.6, 4.0), (3, 0.8, 3.0)):
                topn.record(epoch, dice, hd95)
            with open(topn.history, newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            with open(topn.manifest, encoding="utf-8") as stream:
                entries = json.load(stream)["entries"]
            self.assertEqual([int(row["epoch"]) for row in history], [1, 2, 3])
            self.assertEqual([row["epoch"] for row in entries], [3, 1])
            self.assertEqual(len([name for name in os.listdir(directory)
                                  if name.endswith(".pth")]), 2)


if __name__ == "__main__":
    unittest.main()

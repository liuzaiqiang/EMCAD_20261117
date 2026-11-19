"""Check that TopN preserves the best Dice models and HD95 trajectory."""

import csv
import json
import os
import tempfile
import unittest

from lib.topn_checkpoints import TopNCheckpoints


class TopNCheckpointTests(unittest.TestCase):
    def test_rank_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            def save(path):
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write("checkpoint")

            topn = TopNCheckpoints(directory, 2, save)
            topn.record(1, 0.7, 8.0)
            topn.record(2, 0.6, 4.0)
            topn.record(3, 0.8, 3.0)
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

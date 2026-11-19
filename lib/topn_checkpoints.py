"""Online Dice-ranked checkpoints with paired HD95 trajectory records."""

import csv
import json
import math
import os


class TopNCheckpoints:
    def __init__(self, directory, size, save_model):
        if size < 1:
            raise ValueError("topn_size must be positive")
        self.directory = directory
        self.size = size
        self.save_model = save_model
        self.board = []
        self.history = os.path.join(directory, "topn_history.csv")
        self.manifest = os.path.join(directory, "topn_manifest.json")

    def record(self, epoch, dice, hd95):
        if not math.isfinite(dice):
            raise ValueError("TopN validation Dice must be finite")
        row = {"epoch": int(epoch), "val_dice": float(dice),
               "val_hd95": float(hd95) if math.isfinite(hd95) else None}
        exists = os.path.isfile(self.history)
        with open(self.history, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "val_dice", "val_hd95"))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        qualifies = len(self.board) < self.size or dice > self.board[-1]["val_dice"]
        if not qualifies:
            return False
        filename = "epoch_{:04d}_dice_{:.4f}_hd95_{}.pth".format(
            epoch, dice * 100, "nan" if row["val_hd95"] is None else "{:.4f}".format(hd95))
        path = os.path.join(self.directory, filename)
        self.save_model(path)
        row["checkpoint"] = filename
        self.board = sorted([*self.board, row], key=lambda x: (-x["val_dice"], x["epoch"]))
        for removed in self.board[self.size:]:
            os.remove(os.path.join(self.directory, removed["checkpoint"]))
        self.board = self.board[:self.size]
        with open(self.manifest, "w", encoding="utf-8") as stream:
            json.dump({"topn_size": self.size, "ranking": "val_dice",
                       "entries": self.board}, stream, indent=2)
        return True

"""Dice-ranked checkpoint retention with a paired HD95 history."""

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
        self.entries = []
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
        if len(self.entries) >= self.size and dice <= self.entries[-1]["val_dice"]:
            return False
        filename = "epoch_{:04d}_dice_{:.4f}_hd95_{}.pth".format(
            epoch, dice * 100, "nan" if row["val_hd95"] is None else "{:.4f}".format(hd95))
        self.save_model(os.path.join(self.directory, filename))
        row["checkpoint"] = filename
        self.entries = sorted([*self.entries, row], key=lambda item: (-item["val_dice"], item["epoch"]))
        for removed in self.entries[self.size:]:
            os.remove(os.path.join(self.directory, removed["checkpoint"]))
        self.entries = self.entries[:self.size]
        with open(self.manifest, "w", encoding="utf-8") as stream:
            json.dump({"topn_size": self.size, "ranking": "val_dice", "entries": self.entries},
                      stream, indent=2)
        return True

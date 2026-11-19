"""Validation-only checkpoint ranking; never participates in model updates."""

import csv
import json
import math
import os


def validate_topn_config(enabled, size, threshold_percent=0.0):
    if enabled not in (0, 1) or isinstance(enabled, bool):
        raise ValueError("topn_enabled must be 0 or 1")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("topn_size must be a positive integer")
    if not math.isfinite(threshold_percent) or not 0 <= threshold_percent <= 100:
        raise ValueError("topn_dice_threshold must be in [0, 100]")


class TopNCheckpoints:
    """Keep the N highest validation Dice weights and record Dice/HD95 history."""

    def __init__(self, directory, size, threshold_percent=0.0):
        validate_topn_config(1, size, threshold_percent)
        self.directory = directory
        self.size = size
        self.threshold_percent = threshold_percent
        self.entries = []
        self.weights_dir = os.path.join(directory, "topn_weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self._manifest()

    def _manifest(self):
        path = os.path.join(self.directory, "topn_manifest.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump({"ranking_metric": "dice", "topn_size": self.size,
                       "dice_threshold_percent": self.threshold_percent,
                       "entries": self.entries}, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def record(self, epoch, dice, hd95, save_checkpoint):
        """Append every validation pass; save only a qualifying candidate."""
        dice = float(dice)
        hd95 = float(hd95)
        if not math.isfinite(dice):
            raise ValueError("validation Dice is non-finite")
        qualifies = (dice * 100 + 1e-10 >= self.threshold_percent and
                     (len(self.entries) < self.size or
                      dice > min(entry["dice"] for entry in self.entries)))
        path = os.path.join(self.directory, "topn_history.csv")
        with open(path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "dice", "hd95", "admitted"])
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "dice": dice, "hd95": hd95,
                             "admitted": int(qualifies)})
        if not qualifies:
            return False
        filename = "epoch_{:04d}_dice_{:.4f}.pth".format(epoch, dice * 100)
        checkpoint = os.path.join(self.weights_dir, filename)
        save_checkpoint(checkpoint)
        self.entries.append({"epoch": epoch, "dice": dice,
                             "hd95": hd95 if math.isfinite(hd95) else None,
                             "checkpoint": os.path.relpath(checkpoint, self.directory)})
        self.entries.sort(key=lambda entry: (-entry["dice"], entry["epoch"]))
        removed = self.entries[self.size:]
        self.entries = self.entries[:self.size]
        self._manifest()
        for entry in removed:
            os.remove(os.path.join(self.directory, entry["checkpoint"]))
        return True

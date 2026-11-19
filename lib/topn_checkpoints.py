"""Dice-ranked checkpoint set with per-epoch Dice/HD95 trajectory."""

import csv
import math
import os


class TopNCheckpoints:
    def __init__(self, directory, size, save_model):
        if size < 1:
            raise ValueError("topn_size must be positive")
        self.directory = os.path.join(directory, "topn")
        os.makedirs(self.directory, exist_ok=True)
        self.size = size
        self.save_model = save_model
        self.entries = []
        self.trajectory = os.path.join(self.directory, "trajectory.csv")

    def update(self, model, epoch, dice, hd95):
        if not math.isfinite(float(dice)):
            raise ValueError("TopN Dice must be finite")
        path = os.path.join(self.directory, "epoch_{:04d}.pth".format(epoch))
        candidate = (float(dice), int(epoch), float(hd95), path)
        admitted = len(self.entries) < self.size or candidate[:2] > self.entries[-1][:2]
        if admitted:
            self.save_model(model, path)
            self.entries.append(candidate)
            self.entries.sort(key=lambda item: (-item[0], -item[1]))
            if len(self.entries) > self.size:
                removed = self.entries.pop()
                os.remove(removed[3])
        exists = os.path.exists(self.trajectory)
        with open(self.trajectory, "a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(("epoch", "dice", "hd95", "topn_admitted", "checkpoint"))
            writer.writerow((epoch, dice, hd95, int(admitted), path if admitted else ""))
        return admitted

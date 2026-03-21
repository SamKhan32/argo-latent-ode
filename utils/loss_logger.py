"""
utils/loss_logger.py

Simple CSV loss logger. Each training stage writes its own file under results/.
Columns: epoch, train_loss, val_loss
"""

import os
import csv


class LossLogger:

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def log(self, epoch, train_loss, val_loss):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
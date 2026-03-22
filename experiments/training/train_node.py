"""
experiments/training/train_node.py

Stage 2 — Neural ODE training on latent trajectories.
Losses saved to results/node_losses.csv.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

from globals.config import ODE_LR, ODE_EPOCHS, BATCH_SIZE, LATENT_DIM
from data.datasets import ArgoLatentDataset
from models.architectures.ode import ODEFunc
from utils.loss_logger import LossLogger

WINDOW_SIZE = 5
STRIDE      = 2

ODE_RTOL = 1e-3
ODE_ATOL = 1e-4

T_GRID = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float32)


class SlidingWindowDataset(Dataset):

    def __init__(self, latent_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.records     = latent_dataset.records
        self.window_size = window_size
        self.windows     = []

        from collections import defaultdict
        device_records = defaultdict(list)
        for r in self.records:
            device_records[r["device_idx"]].append(r)

        for device_idx, recs in device_records.items():
            recs = sorted(recs, key=lambda r: r["t"])
            n = len(recs)
            for start in range(0, n - window_size + 1, stride):
                window = recs[start : start + window_size]
                times  = [r["t"] for r in window]
                if all(times[i] < times[i+1] for i in range(len(times)-1)):
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        p   = torch.stack([torch.tensor(r["p"],   dtype=torch.float32) for r in window])
        lat = torch.tensor([r["lat"] for r in window], dtype=torch.float32)
        lon = torch.tensor([r["lon"] for r in window], dtype=torch.float32)
        return {"p": p, "lat": lat, "lon": lon}


def train_ode(
    latent_path="checkpoints/latent_cycles.pt",
    checkpoint_dir="checkpoints",
    checkpoint_name="ode_best.pt",
    log_path="results/node_losses.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    t_grid = T_GRID.to(device)

    print(f"Loading latent cycles from {latent_path}")
    ckpt = torch.load(latent_path, map_location="cpu", weights_only=False)

    latent_train = ArgoLatentDataset(ckpt["train"])
    latent_val   = ArgoLatentDataset(ckpt["val"])

    print("Building train windows...")
    train_windows = SlidingWindowDataset(latent_train, WINDOW_SIZE, STRIDE)
    print(f"Train windows: {len(train_windows)}")

    print("Building val windows...")
    val_windows = SlidingWindowDataset(latent_val, WINDOW_SIZE, STRIDE)
    print(f"Val windows: {len(val_windows)}")

    train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_windows,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ode_func  = ODEFunc().to(device)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=ODE_LR)
    loss_fn   = nn.MSELoss()

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, ODE_EPOCHS + 1):
        t_start = time.time()

        ode_func.train()
        train_loss = 0.0

        for batch in train_loader:
            p   = batch["p"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            lat0 = lat[:, 0:1]
            lon0 = lon[:, 0:1]
            z0   = torch.cat([p[:, 0, :], lat0, lon0], dim=-1)

            z_pred = odeint(ode_func, z0, t_grid, method="dopri5",
                            rtol=ODE_RTOL, atol=ODE_ATOL)

            p_pred = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)
            loss   = loss_fn(p_pred, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ode_func.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                p   = batch["p"].to(device)
                lat = batch["lat"].to(device)
                lon = batch["lon"].to(device)

                lat0 = lat[:, 0:1]
                lon0 = lon[:, 0:1]
                z0   = torch.cat([p[:, 0, :], lat0, lon0], dim=-1)

                z_pred = odeint(ode_func, z0, t_grid, method="dopri5",
                                rtol=ODE_RTOL, atol=ODE_ATOL)

                p_pred    = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)
                val_loss += loss_fn(p_pred, p).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{ODE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": ode_func.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nNODE training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path


if __name__ == "__main__":
    train_ode()
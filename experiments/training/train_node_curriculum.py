"""
experiments/training/train_node_curriculum.py

Stage 2b — Curriculum Neural ODE training on latent trajectories.

Progressively increases window size across phases, allowing the ODE to
first learn short-horizon dynamics before being asked to generalize further.

Schedule is controlled by CURRICULUM_WINDOWS and CURRICULUM_WEIGHTS in
globals/config.py. Each phase gets its own cosine annealing cycle,
restarting fresh from the previous phase's weights.

Example with ODE_EPOCHS=200, CURRICULUM_WINDOWS=[5, 10, 20, 25],
CURRICULUM_WEIGHTS=[0.15, 0.20, 0.25, 0.40]:
    Phase 1: window=5,  epochs=30,  lr: ODE_LR -> 1e-6
    Phase 2: window=10, epochs=40,  lr: ODE_LR -> 1e-6
    Phase 3: window=20, epochs=50,  lr: ODE_LR -> 1e-6
    Phase 4: window=25, epochs=80,  lr: ODE_LR -> 1e-6

Losses saved to results/node_curriculum_losses.csv.

Usage:
    python run/main.py --stage ode_curriculum
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

from globals.config import (
    ODE_LR, ODE_EPOCHS, BATCH_SIZE, LATENT_DIM,
    STRIDE, CURRICULUM_WINDOWS, CURRICULUM_WEIGHTS,
)
from data.datasets import ArgoLatentDataset
from models.architectures.ode import ODEFunc
from utils.loss_logger import LossLogger

ODE_RTOL = 1e-4
ODE_ATOL = 1e-5


class SlidingWindowDataset(Dataset):

    def __init__(self, latent_dataset, window_size, stride=STRIDE):
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
                if all(times[i] < times[i+1] for i in range(len(times) - 1)):
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        p   = torch.stack([torch.tensor(r["p"],   dtype=torch.float32) for r in window])
        lat = torch.tensor([r["lat"] for r in window], dtype=torch.float32)
        lon = torch.tensor([r["lon"] for r in window], dtype=torch.float32)
        return {"p": p, "lat": lat, "lon": lon}


def build_phase_epochs(total_epochs, weights, n_phases):
    """
    Distribute total_epochs across phases using weights.
    Remainder from rounding goes to the last phase.
    """
    assert len(weights) == n_phases, "CURRICULUM_WEIGHTS must have same length as CURRICULUM_WINDOWS"
    assert abs(sum(weights) - 1.0) < 1e-6, "CURRICULUM_WEIGHTS must sum to 1.0"
    phase_epochs = [max(1, int(w * total_epochs)) for w in weights]
    phase_epochs[-1] += total_epochs - sum(phase_epochs)
    return phase_epochs


def train_ode_curriculum(
    latent_path="checkpoints/latent_cycles.pt",
    checkpoint_dir="checkpoints",
    checkpoint_name="ode_best.pt",
    log_path="results/node_curriculum_losses.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading latent cycles from {latent_path}")
    ckpt = torch.load(latent_path, map_location="cpu", weights_only=False)

    latent_train = ArgoLatentDataset(ckpt["train"])
    latent_val   = ArgoLatentDataset(ckpt["val"])

    ode_func  = ODEFunc().to(device)
    loss_fn   = nn.MSELoss()
    logger    = LossLogger(log_path, extras=["phase", "window_size"])

    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    n_phases     = len(CURRICULUM_WINDOWS)
    phase_epochs = build_phase_epochs(ODE_EPOCHS, CURRICULUM_WEIGHTS, n_phases)

    print(f"\nCurriculum schedule:")
    for i, (w, e) in enumerate(zip(CURRICULUM_WINDOWS, phase_epochs)):
        print(f"  Phase {i+1}: window={w:2d} ({w*10:3d} days)  epochs={e}")
    print(f"  Total epochs: {sum(phase_epochs)}")

    global_epoch = 0

    for phase_idx, (window_size, n_epochs) in enumerate(zip(CURRICULUM_WINDOWS, phase_epochs)):
        print(f"\n{'='*60}")
        print(f"Phase {phase_idx + 1}/{n_phases} — window={window_size} ({window_size * 10} days)  epochs={n_epochs}")
        print(f"{'='*60}")

        t_grid = torch.arange(0, window_size, dtype=torch.float32).to(device) * 10.0

        print("Building train windows...")
        train_windows = SlidingWindowDataset(latent_train, window_size, STRIDE)
        print(f"Train windows: {len(train_windows)}")

        print("Building val windows...")
        val_windows = SlidingWindowDataset(latent_val, window_size, STRIDE)
        print(f"Val windows: {len(val_windows)}")

        train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_windows,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # fresh optimizer and cosine scheduler each phase
        optimizer = torch.optim.Adam(ode_func.parameters(), lr=ODE_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )

        for epoch in range(1, n_epochs + 1):
            global_epoch += 1
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
                optimizer.zero_grad()  # clear gradients first

                z_pred = odeint(ode_func, z0, t_grid, method="dopri5",
                                rtol=ODE_RTOL, atol=ODE_ATOL)

                p_pred = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)
                loss   = loss_fn(p_pred, p)

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

            elapsed    = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  [{phase_idx+1}/{n_phases}] "
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"(global {global_epoch})  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={current_lr:.2e}  time={elapsed:.1f}s"
            )

            logger.log(global_epoch, train_loss, val_loss,
                       phase=phase_idx + 1,
                       window_size=window_size)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": ode_func.state_dict()}, ckpt_path)
                print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

            scheduler.step()

    print(f"\nCurriculum ODE training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path


if __name__ == "__main__":
    train_ode_curriculum()
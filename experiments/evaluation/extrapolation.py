"""
experiments/evaluation/extrapolation.py

Extrapolation benchmark — evaluates all dynamics models across increasing
forecast horizons. Each horizon is measured in cycles (1 cycle = 10 days).

For each horizon H:
  - ODE  : integrates over T_GRID = [0, 10, 20, ..., H*10]
  - GRU  : unrolls n_steps = H discrete steps

MSE is computed at each step along the trajectory, producing a
per-step error profile for each model and horizon.

Results saved to results/extrapolation_results.csv.

Usage:
    python run/main.py --stage extrapolation
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torchdiffeq import odeint

from config import LATENT_DIM, ODE_HIDDEN, BATCH_SIZE
from data.datasets import ArgoLatentDataset
from experiments.training.train_node import SlidingWindowDataset
from globals.baseline_registry import BASELINES, DEPTH_ONLY_VAL_LOSS
from torch.utils.data import DataLoader

ODE_RTOL = 1e-3
ODE_ATOL = 1e-4

# Horizons in number of cycles (1 cycle = 10 days)
HORIZONS = [5, 10, 15, 20, 30, 40, 50, 100]


def load_model(entry, device):
    """Instantiate and load a model from a registry entry."""
    model = entry["model_class"](**entry["kwargs"]).to(device)
    ckpt  = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_ode(model, p0, lat0, lon0, horizon, device):
    """
    Roll ODE forward for `horizon` steps.
    Returns (horizon+1, B, latent_dim) trajectory including t=0.
    """
    t_grid = torch.arange(0, horizon + 1, dtype=torch.float32).to(device) * 10.0
    z0     = torch.cat([p0, lat0, lon0], dim=-1)
    z_pred = odeint(model, z0, t_grid, method="dopri5", rtol=ODE_RTOL, atol=ODE_ATOL)
    return z_pred[:, :, :LATENT_DIM]   # (T, B, latent_dim)


def run_gru(model, p0, lat0, lon0, horizon):
    """
    Roll GRU forward for `horizon` steps.
    Returns (horizon+1, B, latent_dim) trajectory including t=0.
    """
    traj = model(p0, lat0.squeeze(-1), lon0.squeeze(-1), n_steps=horizon)
    return traj   # (T, B, latent_dim)


def evaluate_horizon(model, kind, val_loader, horizon, device):
    """
    Evaluate a model at a given horizon.
    Returns per-step MSE list of length (horizon+1).
    """
    loss_fn    = nn.MSELoss()
    step_losses = torch.zeros(horizon + 1, device=device)
    n_batches   = 0

    with torch.no_grad():
        for batch in val_loader:
            p   = batch["p"].to(device)     # (B, W, latent_dim) — W may be < horizon+1
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            p0   = p[:, 0, :]
            lat0 = lat[:, 0:1]
            lon0 = lon[:, 0:1]

            if kind == "ode":
                traj = run_ode(model, p0, lat0, lon0, horizon, device)
            else:
                traj = run_gru(model, p0, lat0, lon0, horizon)

            # traj: (horizon+1, B, latent_dim)
            # We only have ground truth for min(W, horizon+1) steps
            n_gt = min(p.shape[1], horizon + 1)
            for step in range(n_gt):
                step_losses[step] += loss_fn(traj[step], p[:, step, :]).item()

            n_batches += 1

    return (step_losses / max(n_batches, 1)).cpu().tolist()


def run_extrapolation(
    latent_path="checkpoints/latent_cycles.pt",
    output_path="results/extrapolation_results.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading latent cycles from {latent_path}")
    ckpt        = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent_val  = ArgoLatentDataset(ckpt["val"])

    # Use the largest horizon window for the dataset so we have long sequences
    max_horizon = max(HORIZONS)
    val_windows = SlidingWindowDataset(latent_val, window_size=max_horizon + 1, stride=max_horizon)
    val_loader  = DataLoader(val_windows, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Val windows (horizon={max_horizon}): {len(val_windows)}")

    rows = []

    for name, entry in BASELINES.items():
        print(f"\nEvaluating: {name}")
        model = load_model(entry, device)

        for horizon in HORIZONS:
            print(f"  horizon={horizon} ({horizon * 10} days)...")
            step_losses = evaluate_horizon(model, entry["kind"], val_loader, horizon, device)
            final_loss  = step_losses[horizon]

            rows.append({
                "model":        name,
                "horizon_steps": horizon,
                "horizon_days":  horizon * 10,
                "final_mse":     final_loss,
                "step_losses":   step_losses,
            })
            print(f"  -> final step MSE: {final_loss:.4f}")

    # Add depth-only baseline as flat reference
    for horizon in HORIZONS:
        rows.append({
            "model":         "depth_only",
            "horizon_steps": horizon,
            "horizon_days":  horizon * 10,
            "final_mse":     DEPTH_ONLY_VAL_LOSS,
            "step_losses":   [DEPTH_ONLY_VAL_LOSS] * (horizon + 1),
        })

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== Extrapolation Summary (final step MSE) ===")
    pivot = df.pivot(index="horizon_days", columns="model", values="final_mse")
    print(pivot.to_string())

    return df

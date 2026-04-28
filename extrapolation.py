"""
extrapolation.py

Extrapolation benchmark — evaluates all dynamics models across increasing
forecast horizons. Each horizon is measured in cycles (1 cycle = 10 days).

Results saved to {results_dir}/extrapolation_results.csv.

Usage:
    python run/main.py --stage extrapolation --results_dir results/20260428_143012
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torchdiffeq import odeint

from config import LATENT_DIM, ODE_HIDDEN, BATCH_SIZE
from utils.datasets import ArgoLatentDataset
from train.train_node import SlidingWindowDataset
from torch.utils.data import DataLoader

ODE_RTOL = 1e-3
ODE_ATOL = 1e-4

HORIZONS = [5, 10, 15, 20, 30, 40, 50, 100]


def load_model(entry, device):
    model = entry["model_class"](**entry["kwargs"]).to(device)
    ckpt  = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_ode(model, p0, lat0, lon0, horizon, device):
    t_grid = torch.arange(0, horizon + 1, dtype=torch.float32).to(device) * 10.0
    z0     = torch.cat([p0, lat0, lon0], dim=-1)
    z_pred = odeint(model, z0, t_grid, method="dopri5", rtol=ODE_RTOL, atol=ODE_ATOL)
    return z_pred[:, :, :LATENT_DIM]


def run_gru(model, p0, lat0, lon0, horizon):
    return model(p0, lat0.squeeze(-1), lon0.squeeze(-1), n_steps=horizon)


def evaluate_horizon(model, kind, val_loader, horizon, device):
    loss_fn     = nn.MSELoss()
    step_losses = torch.zeros(horizon + 1, device=device)
    n_batches   = 0

    with torch.no_grad():
        for batch in val_loader:
            p   = batch["p"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            p0   = p[:, 0, :]
            lat0 = lat[:, 0:1]
            lon0 = lon[:, 0:1]

            if kind == "ode":
                traj = run_ode(model, p0, lat0, lon0, horizon, device)
            else:
                traj = run_gru(model, p0, lat0, lon0, horizon)

            n_gt = min(p.shape[1], horizon + 1)
            for step in range(n_gt):
                step_losses[step] += loss_fn(traj[step], p[:, step, :]).item()

            n_batches += 1

    return (step_losses / max(n_batches, 1)).cpu().tolist()


def run_extrapolation(
    latent_path,
    output_path,
    results_dir=None,
):
    # build BASELINES from the results_dir checkpoints if no registry provided
    from models.ode import ODEFunc
    from models.gru import GRUDynamics

    rd = results_dir or os.path.dirname(output_path)

    baselines = {
        "ode": {
            "model_class": ODEFunc,
            "kwargs":      {"latent_dim": LATENT_DIM, "hidden": ODE_HIDDEN},
            "checkpoint":  os.path.join(rd, "ode_best.pt"),
            "kind":        "ode",
        },
        "gru": {
            "model_class": GRUDynamics,
            "kwargs":      {"latent_dim": LATENT_DIM, "hidden": ODE_HIDDEN},
            "checkpoint":  os.path.join(rd, "gru_best.pt"),
            "kind":        "gru",
        },
    }

    # load depth-only baseline val loss if available
    probe_baseline_path = os.path.join(rd, "probe_baseline_best.pt")
    depth_only_val_loss = None
    if os.path.exists(probe_baseline_path):
        ckpt = torch.load(probe_baseline_path, map_location="cpu", weights_only=False)
        depth_only_val_loss = ckpt.get("val_loss")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading latent cycles from {latent_path}")
    ckpt       = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent_val = ArgoLatentDataset(ckpt["val"])

    max_horizon = max(HORIZONS)
    val_windows = SlidingWindowDataset(latent_val, window_size=max_horizon + 1, stride=max_horizon)
    val_loader  = DataLoader(val_windows, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Val windows (horizon={max_horizon}): {len(val_windows)}")

    rows = []

    for name, entry in baselines.items():
        if not os.path.exists(entry["checkpoint"]):
            print(f"  Skipping {name} — checkpoint not found: {entry['checkpoint']}")
            continue

        print(f"\nEvaluating: {name}")
        model = load_model(entry, device)

        for horizon in HORIZONS:
            print(f"  horizon={horizon} ({horizon * 10} days)...")
            step_losses = evaluate_horizon(model, entry["kind"], val_loader, horizon, device)
            final_loss  = step_losses[horizon]

            rows.append({
                "model":         name,
                "horizon_steps": horizon,
                "horizon_days":  horizon * 10,
                "final_mse":     final_loss,
                "step_losses":   step_losses,
            })
            print(f"  -> final step MSE: {final_loss:.4f}")

    if depth_only_val_loss is not None:
        for horizon in HORIZONS:
            rows.append({
                "model":         "depth_only",
                "horizon_steps": horizon,
                "horizon_days":  horizon * 10,
                "final_mse":     depth_only_val_loss,
                "step_losses":   [depth_only_val_loss] * (horizon + 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    print("\n=== Extrapolation Summary (final step MSE) ===")
    pivot = df.pivot(index="horizon_days", columns="model", values="final_mse")
    print(pivot.to_string())

    return df

"""
results/visualization/plot_extrapolation.py

Plots extrapolation benchmark results — final step MSE vs forecast horizon
for all models, plus per-step error curves for each horizon.

Saves to results/visualization/extrapolation.png

Usage:
    python results/visualization/plot_extrapolation.py
"""

import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

CSV_PATH = "results/extrapolation_results.csv"
OUT_PATH = "results/visualization/extrapolation.png"

COLORS = {
    "ode":        "#2196F3",   # blue
    "gru":        "#FF9800",   # orange
    "depth_only": "#9E9E9E",   # grey
}

LABELS = {
    "ode":        "Latent ODE",
    "gru":        "Latent GRU",
    "depth_only": "Depth-only baseline",
}


def main():
    df = pd.read_csv(CSV_PATH)
    df["step_losses"] = df["step_losses"].apply(ast.literal_eval)

    models   = df["model"].unique()
    horizons = sorted(df["horizon_days"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: final step MSE vs horizon ────────────────────────────────────
    ax = axes[0]
    for model in models:
        sub = df[df["model"] == model].sort_values("horizon_days")
        ax.plot(
            sub["horizon_days"], sub["final_mse"],
            label=LABELS.get(model, model),
            color=COLORS.get(model, "black"),
            linewidth=2,
            linestyle="--" if model == "depth_only" else "-",
            marker="o" if model != "depth_only" else None,
            markersize=5,
        )

    ax.set_title("Final Step MSE vs Forecast Horizon", fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("MSE (latent space)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: per-step error curves for largest horizon ────────────────────
    ax = axes[1]
    max_horizon = df["horizon_days"].max()
    sub = df[df["horizon_days"] == max_horizon]

    for _, row in sub.iterrows():
        model      = row["model"]
        step_losses = row["step_losses"]
        steps       = list(range(len(step_losses)))
        days        = [s * 10 for s in steps]

        ax.plot(
            days, step_losses,
            label=LABELS.get(model, model),
            color=COLORS.get(model, "black"),
            linewidth=2,
            linestyle="--" if model == "depth_only" else "-",
        )

    ax.set_title(f"Per-Step MSE at {max_horizon}-Day Horizon", fontsize=13, fontweight="bold")
    ax.set_xlabel("Days from initial observation")
    ax.set_ylabel("MSE (latent space)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Extrapolation Benchmark — Latent ODE vs GRU", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== Final Step MSE by Horizon ===")
    pivot = df.pivot(index="horizon_days", columns="model", values="final_mse")
    pivot.columns = [LABELS.get(c, c) for c in pivot.columns]
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
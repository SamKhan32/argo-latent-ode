"""
results/visualization/plot_all_losses_log.py

Same as plot_all_losses.py but with log-scale y-axis.
Saves to results/visualization/all_losses_log.png

Usage:
    python results/visualization/plot_all_losses_log.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_PATH    = "results/visualization/all_losses_log.png"

RUNS = [
    {"file": "encoder_losses.csv",     "title": "Encoder (T/S Reconstruction)"},
    {"file": "node_losses.csv",        "title": "Neural ODE (Latent Dynamics)"},
    {"file": "gru_losses.csv",         "title": "GRU (Latent Dynamics)"},
    {"file": "probe_losses.csv",       "title": "ODE Probe (Oxygen)"},
    {"file": "gru_probe_losses.csv",   "title": "GRU Probe (Oxygen)"},
]


def load_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except FileNotFoundError:
        return None


def main():
    available = [(r, load_csv(os.path.join(RESULTS_DIR, r["file"])))
                 for r in RUNS]
    available = [(r, df) for r, df in available if df is not None]

    n = len(available)
    if n == 0:
        print("No loss CSVs found in results/")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = [axes] if n == 1 else axes.flatten()

    for ax, (run, df) in zip(axes, available):
        epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

        ax.plot(epochs, df["train_loss"], label="train", linewidth=1.5)
        ax.plot(epochs, df["val_loss"],   label="val",   linewidth=1.5, linestyle="--")

        ax.set_yscale("log")
        ax.set_title(run["title"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        best_val = df["val_loss"].min()
        best_ep  = df.loc[df["val_loss"].idxmin(), "epoch"] if "epoch" in df.columns \
                   else df["val_loss"].idxmin() + 1
        ax.axhline(best_val, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.annotate(f"best val: {best_val:.4f}",
                    xy=(best_ep, best_val),
                    xytext=(0.97, 0.95),
                    textcoords="axes fraction",
                    ha="right", va="top",
                    fontsize=8, color="red")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Training Loss Curves (Log Scale)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
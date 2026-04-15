"""
plot_results.py
---------------
Generates all publication-quality figures for the Argo Latent ODE project.
Script lives in: results/visualization/
CSVs live in:    results/

Produces:
  01_encoder_training.png        - Encoder/decoder T/S training curves
  02_reconstruction_training.png - NODE vs GRU T/S training on one plot (+ depth-only hline)
  03_node_curriculum.png         - NODE curriculum training with phase shading
  04_probe_training.png          - NODE probe vs GRU probe oxygen training on one plot
  05_extrapolation.png           - Extrapolation MSE vs horizon (NODE, GRU, depth-only)
  06_finetune_training.png       - Fine-tune training curves (total + component losses)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up

# ── Shared style ───────────────────────────────────────────────────────────────
COLORS = {
    "ode":        "#2563EB",   # blue   – primary model
    "gru":        "#DC2626",   # red    – GRU baseline
    "depth_only": "#6B7280",   # grey   – trivial baseline
    "train_ode":  "#93C5FD",   # light blue – NODE train
    "train_gru":  "#FCA5A5",   # light red  – GRU train
    "oxy":        "#059669",   # teal   – oxygen
    "ts_raw":     "#7C3AED",   # purple – raw T/S
    "ts_evo":     "#D97706",   # amber  – evolved T/S
}

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "lines.linewidth":   2.0,
})

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}")

def annotate_best(ax, epochs, values, color, fmt="{:.4f}"):
    """Annotate the best (min) val loss with a marker and label."""
    idx = values.idxmin()
    x, y = epochs.iloc[idx], values.iloc[idx]
    ax.scatter(x, y, color=color, s=60, zorder=5)
    ax.annotate(
        fmt.format(y),
        xy=(x, y), xytext=(8, -10),
        textcoords="offset points",
        fontsize=9, color=color, va="top", fontweight="bold",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 01 — Encoder training (solo — nothing to compare against)
# ══════════════════════════════════════════════════════════════════════════════
def plot_encoder_training():
    df = load("encoder_losses")
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df["epoch"], df["train_loss"], color=COLORS["train_ode"], alpha=0.8, label="Train")
    ax.plot(df["epoch"], df["val_loss"],   color=COLORS["ode"],               label="Val")
    annotate_best(ax, df["epoch"], df["val_loss"], COLORS["ode"])

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Encoder / Decoder — T/S Reconstruction Training")
    ax.legend()
    save(fig, "01_encoder_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02 — Reconstruction: NODE and GRU side-by-side subplots
# ══════════════════════════════════════════════════════════════════════════════
def plot_reconstruction_training():
    node = load("node_curriculum_losses")
    gru  = load("gru_losses")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("T/S Reconstruction Training", fontsize=14, fontweight="bold")

    for ax, df, label, vc, tc in [
        (axes[0], node, "Latent NODE", COLORS["ode"], COLORS["train_ode"]),
        (axes[1], gru,  "Latent GRU",  COLORS["gru"], COLORS["train_gru"]),
    ]:
        ax.plot(df["epoch"], df["train_loss"], color=tc,  alpha=0.6, label="Train")
        ax.plot(df["epoch"], df["val_loss"],   color=vc,            label="Val")
        annotate_best(ax, df["epoch"], df["val_loss"], vc)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.legend()

    axes[0].set_ylabel("MSE Loss (log scale)")
    fig.tight_layout()
    save(fig, "02_reconstruction_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03 — NODE curriculum phases (solo — GRU has no curriculum)
# ══════════════════════════════════════════════════════════════════════════════
def plot_node_curriculum():
    df = load("node_curriculum_losses")
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(df["epoch"], df["train_loss"], color=COLORS["train_ode"], alpha=0.6, label="Train")
    ax.plot(df["epoch"], df["val_loss"],   color=COLORS["ode"],               label="Val")
    annotate_best(ax, df["epoch"], df["val_loss"], COLORS["ode"])

    if "phase" in df.columns:
        phase_colors = ["#DBEAFE", "#FEF9C3", "#DCFCE7", "#FCE7F3"]
        for i, ph in enumerate(sorted(df["phase"].unique())):
            mask = df["phase"] == ph
            x0 = df.loc[mask, "epoch"].iloc[0]
            x1 = df.loc[mask, "epoch"].iloc[-1]
            ax.axvspan(x0, x1, alpha=0.25, color=phase_colors[i % len(phase_colors)],
                       label=f"Phase {ph}")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Latent NODE — Curriculum Training Phases")
    ax.legend(ncol=2)
    save(fig, "03_node_curriculum.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04 — Probe: NODE probe and GRU probe side-by-side subplots
# ══════════════════════════════════════════════════════════════════════════════
def plot_probe_training():
    node_p = load("probe_losses")
    gru_p  = load("gru_probe_losses")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Oxygen Probe Training", fontsize=14, fontweight="bold")

    for ax, df, label, vc, tc in [
        (axes[0], node_p, "Latent NODE Probe", COLORS["ode"], COLORS["train_ode"]),
        (axes[1], gru_p,  "Latent GRU Probe",  COLORS["gru"], COLORS["train_gru"]),
    ]:
        ax.plot(df["epoch"], df["train_loss"], color=tc,  alpha=0.6, label="Train")
        ax.plot(df["epoch"], df["val_loss"],   color=vc,            label="Val")
        annotate_best(ax, df["epoch"], df["val_loss"], vc)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.legend()

    axes[0].set_ylabel("MSE Loss (log scale)")
    fig.tight_layout()
    save(fig, "04_probe_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05 — Extrapolation benchmark
# ══════════════════════════════════════════════════════════════════════════════
def plot_extrapolation():
    df = load("extrapolation_results")

    fig, ax = plt.subplots(figsize=(9, 5))

    style = {
        "ode":        dict(color=COLORS["ode"],        marker="o", linestyle="-",  label="Latent NODE"),
        "gru":        dict(color=COLORS["gru"],        marker="s", linestyle="-",  label="Latent GRU"),
        "depth_only": dict(color=COLORS["depth_only"], marker="D", linestyle="--", label="Depth-Only Baseline"),
    }

    for model, grp in df.groupby("model"):
        s = style[model]
        ax.plot(grp["horizon_days"], grp["final_mse"],
                color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], markersize=7, label=s["label"])
        last = grp.iloc[-1]
        ax.annotate(f"{last['final_mse']:.4f}",
                    xy=(last["horizon_days"], last["final_mse"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=s["color"], fontweight="bold")

    train_horizon = df.loc[df["model"] == "ode", "horizon_days"].min()
    ax.axvline(train_horizon, color="#6B7280", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(train_horizon * 1.05, ax.get_ylim()[1],
            "Training horizon", va="top", fontsize=9, color="#6B7280")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(df["horizon_days"].unique())
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Final MSE")
    ax.set_title("Extrapolation Performance vs Forecast Horizon")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    save(fig, "05_extrapolation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06 — Fine-tune training  (total + component losses)
# ══════════════════════════════════════════════════════════════════════════════
def plot_finetune_training():
    df = load("finetune_losses")
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color=COLORS["train_ode"], alpha=0.8, label="Train")
    ax.plot(epochs, df["val_loss"],   color=COLORS["ode"],               label="Val (total)")
    annotate_best(ax, epochs, df["val_loss"], COLORS["ode"])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Fine-Tune — Total Loss")
    ax.legend()

    ax = axes[1]
    for col, color, label in [
        ("val_ts_raw", COLORS["ts_raw"], "Val T/S (raw)"),
        ("val_ts_evo", COLORS["ts_evo"], "Val T/S (evolved)"),
        ("val_oxy",    COLORS["oxy"],    "Val Oxygen"),
    ]:
        ax.plot(epochs, df[col], color=color, label=label)
        annotate_best(ax, epochs, df[col], color)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Fine-Tune — Val Loss Components")
    ax.legend()

    fig.tight_layout()
    save(fig, "06_finetune_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    plot_encoder_training()
    plot_reconstruction_training()
    plot_node_curriculum()
    plot_probe_training()
    plot_extrapolation()
    #plot_finetune_training()
    print(f"\nDone. All figures saved to: {OUT_DIR}")
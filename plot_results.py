"""
plot_results.py

Generates all training curve figures for a given results directory.

Usage:
    python plot_results.py --results_dir results/20260428_143012
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLORS = {
    "ode":        "#2563EB",
    "gru":        "#DC2626",
    "depth_only": "#6B7280",
    "train_ode":  "#93C5FD",
    "train_gru":  "#FCA5A5",
    "oxy":        "#059669",
    "ts_raw":     "#7C3AED",
    "ts_evo":     "#D97706",
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


def load(results_dir, name):
    return pd.read_csv(os.path.join(results_dir, f"{name}.csv"))


def save(fig, figures_dir, name):
    path = os.path.join(figures_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}")


def annotate_best(ax, epochs, values, color, fmt="{:.4f}"):
    idx = values.idxmin()
    x, y = epochs.iloc[idx], values.iloc[idx]
    ax.scatter(x, y, color=color, s=60, zorder=5)
    ax.annotate(
        fmt.format(y),
        xy=(x, y), xytext=(8, -10),
        textcoords="offset points",
        fontsize=9, color=color, va="top", fontweight="bold",
    )


def plot_encoder_training(results_dir, figures_dir):
    df = load(results_dir, "encoder_losses")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color=COLORS["train_ode"], alpha=0.8, label="Train")
    ax.plot(df["epoch"], df["val_loss"],   color=COLORS["ode"],               label="Val")
    annotate_best(ax, df["epoch"], df["val_loss"], COLORS["ode"])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Encoder / Decoder — Reconstruction Training")
    ax.legend()
    save(fig, figures_dir, "01_encoder_training.png")


def plot_reconstruction_training(results_dir, figures_dir):
    node = load(results_dir, "node_curriculum_losses")
    gru  = load(results_dir, "gru_losses")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Latent Dynamics Training", fontsize=14, fontweight="bold")

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
    save(fig, figures_dir, "02_reconstruction_training.png")


def plot_node_curriculum(results_dir, figures_dir):
    df = load(results_dir, "node_curriculum_losses")
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
    save(fig, figures_dir, "03_node_curriculum.png")


def plot_probe_training(results_dir, figures_dir):
    node_p = load(results_dir, "probe_losses")
    gru_p  = load(results_dir, "gru_probe_losses")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Target Variable Probe Training", fontsize=14, fontweight="bold")

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
    save(fig, figures_dir, "04_probe_training.png")


def plot_extrapolation(results_dir, figures_dir):
    df = load(results_dir, "extrapolation_results")
    fig, ax = plt.subplots(figsize=(9, 5))

    style = {
        "ode":        dict(color=COLORS["ode"],        marker="o", linestyle="-",  label="Latent NODE"),
        "gru":        dict(color=COLORS["gru"],        marker="s", linestyle="-",  label="Latent GRU"),
        "depth_only": dict(color=COLORS["depth_only"], marker="D", linestyle="--", label="Depth-Only Baseline"),
    }

    for model, grp in df.groupby("model"):
        if model not in style:
            continue
        s = style[model]
        ax.plot(grp["horizon_days"], grp["final_mse"],
                color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], markersize=7, label=s["label"])
        last = grp.iloc[-1]
        ax.annotate(f"{last['final_mse']:.4f}",
                    xy=(last["horizon_days"], last["final_mse"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=s["color"], fontweight="bold")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(df["horizon_days"].unique())
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Final MSE")
    ax.set_title("Extrapolation Performance vs Forecast Horizon")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    save(fig, figures_dir, "05_extrapolation.png")


def plot_finetune_training(results_dir, figures_dir):
    df     = load(results_dir, "finetune_losses")
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
        ("val_target", COLORS["oxy"],    "Val Target"),
    ]:
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, label=label)
            annotate_best(ax, epochs, df[col], color)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Fine-Tune — Val Loss Components")
    ax.legend()

    fig.tight_layout()
    save(fig, figures_dir, "06_finetune_training.png")


def main(results_dir):
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Generating figures for: {results_dir}")

    # only plot what exists
    def exists(name):
        return os.path.exists(os.path.join(results_dir, f"{name}.csv"))

    if exists("encoder_losses"):            plot_encoder_training(results_dir, figures_dir)
    if exists("node_curriculum_losses") and exists("gru_losses"):
                                            plot_reconstruction_training(results_dir, figures_dir)
    if exists("node_curriculum_losses"):    plot_node_curriculum(results_dir, figures_dir)
    if exists("probe_losses") and exists("gru_probe_losses"):
                                            plot_probe_training(results_dir, figures_dir)
    if exists("extrapolation_results"):     plot_extrapolation(results_dir, figures_dir)
    if exists("finetune_losses"):           plot_finetune_training(results_dir, figures_dir)

    print(f"\nDone. Figures saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Timestamped results directory to plot")
    args = parser.parse_args()
    main(args.results_dir)

"""
Latent Space Analysis
=====================
Run from project root:
    python -m experiments.latent_analysis.latent_analysis

Outputs figures to: experiments/latent_analysis/figures/
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# optional UMAP — gracefully skipped if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed — skipping UMAP plots. pip install umap-learn")

FIGURE_DIR = os.path.join("experiments", "latent_analysis", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

LATENT_PATH = os.path.join("checkpoints", "latent_cycles.pt")



def save(name):
    path = os.path.join(FIGURE_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


def load_records(latent_path=LATENT_PATH):
    print(f"Loading latent cycles from {latent_path} ...")
    ckpt = torch.load(latent_path, map_location="cpu", weights_only=False)
    train  = ckpt["train"]
    val    = ckpt["val"]
    probe  = ckpt.get("probe", [])
    all_records = train + val + probe
    print(f"  train={len(train)}  val={len(val)}  probe={len(probe)}  total={len(all_records)}")
    return all_records, train, val, probe


def records_to_arrays(records):
    P          = np.stack([np.array(r["p"],   dtype=np.float32) for r in records])
    lats       = np.array([r["lat"]        for r in records], dtype=np.float32)
    lons       = np.array([r["lon"]        for r in records], dtype=np.float32)
    ts         = np.array([r["t"]          for r in records], dtype=np.float64)
    device_ids = np.array([r["device_idx"] for r in records], dtype=np.int32)
    return P, lats, lons, ts, device_ids


def days_to_month(ts_days):
    """Convert days-since-2000 to approximate month (1–12)."""
    import pandas as pd
    epoch = pd.Timestamp("2000-01-01")
    timestamps = pd.to_datetime(ts_days * 86400 * 1e9 + epoch.value)
    return timestamps.month


# ── Figure 1: PCA variance explained ─────────────────────────────────────────

def plot_pca_variance(P):
    print("Plotting PCA variance explained ...")
    pca_full = PCA().fit(P)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_ * 100,
           color="steelblue", alpha=0.7, label="Per component")
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="tomato", marker="o",
            markersize=4, label="Cumulative")
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Variance Explained — Latent Space")
    ax.legend()
    ax.set_xlim(0.5, len(cumvar) + 0.5)
    save("01_pca_variance.png")


# ── Figure 2: PCA scatter colored by latitude ─────────────────────────────────

def plot_pca_by_lat(P, lats):
    print("Plotting PCA scatter by latitude ...")
    pca2 = PCA(n_components=2).fit_transform(P)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pca2[:, 0], pca2[:, 1], c=lats, cmap="RdYlBu",
                    s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Latitude (°N)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent Space (PCA) — colored by Latitude")
    save("02_pca_latitude.png")


# ── Figure 3: PCA scatter colored by longitude ────────────────────────────────

def plot_pca_by_lon(P, lons):
    print("Plotting PCA scatter by longitude ...")
    pca2 = PCA(n_components=2).fit_transform(P)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pca2[:, 0], pca2[:, 1], c=lons, cmap="twilight_shifted",
                    s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Longitude (°E)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent Space (PCA) — colored by Longitude")
    save("03_pca_longitude.png")


# ── Figure 4: PCA scatter colored by season ───────────────────────────────────

def plot_pca_by_season(P, ts):
    print("Plotting PCA scatter by season ...")
    pca2   = PCA(n_components=2).fit_transform(P)
    months = days_to_month(ts)

    season_map   = {12: 0, 1: 0, 2: 0,   # winter
                     3: 1,  4: 1, 5: 1,   # spring
                     6: 2,  7: 2, 8: 2,   # summer
                     9: 3, 10: 3, 11: 3}  # autumn
    season_labels = ["Winter", "Spring", "Summer", "Autumn"]
    season_colors = ["#4575b4", "#91cf60", "#fc8d59", "#d73027"]
    seasons = np.array([season_map[m] for m in months])

    fig, ax = plt.subplots(figsize=(7, 6))
    for s, label, color in zip(range(4), season_labels, season_colors):
        mask = seasons == s
        ax.scatter(pca2[mask, 0], pca2[mask, 1], c=color, s=4,
                   alpha=0.6, label=label, linewidths=0)
    ax.legend(markerscale=3)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent Space (PCA) — colored by Season")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_yscale("symlog", linthresh=0.5)
    save("04_pca_season.png")


# ── Figure 5: Geographic distribution of floats ───────────────────────────────

def plot_geographic(lats, lons, device_ids):
    print("Plotting geographic distribution ...")
    n_devices = len(np.unique(device_ids))
    colors = cm.tab20(np.linspace(0, 1, min(n_devices, 20)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, did in enumerate(np.unique(device_ids)):
        mask = device_ids == did
        ax.scatter(lons[mask], lats[mask], s=3, alpha=0.5,
                   color=colors[i % 20], linewidths=0)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Float Trajectories — {n_devices} floats")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    save("05_geographic.png")


# ── Figure 6: Latent trajectories of sample floats through PCA space ──────────

def plot_latent_trajectories(P, ts, device_ids, n_floats=6):
    print("Plotting latent trajectories through PCA space ...")
    pca2 = PCA(n_components=2).fit_transform(P)

    unique_devices = np.unique(device_ids)
    # pick floats with the most observations
    counts = {d: np.sum(device_ids == d) for d in unique_devices}
    top_devices = sorted(counts, key=counts.get, reverse=True)[:n_floats]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, did in zip(axes, top_devices):
        mask = device_ids == did
        pc   = pca2[mask]
        t    = ts[mask]
        order = np.argsort(t)
        pc = pc[order]

        # color by time
        t_norm = (t[order] - t[order].min()) / (t[order].max() - t[order].min() + 1e-8)
        for j in range(len(pc) - 1):
            ax.plot(pc[j:j+2, 0], pc[j:j+2, 1], color=cm.plasma(t_norm[j]),
                    linewidth=1.2, alpha=0.8)
        ax.scatter(pc[:, 0], pc[:, 1], c=t_norm, cmap="plasma",
                   s=20, zorder=5, linewidths=0)
        ax.scatter(*pc[0], color="green",  s=60, zorder=6, marker="^", label="start")
        ax.scatter(*pc[-1], color="red",   s=60, zorder=6, marker="v", label="end")
        ax.set_title(f"Float {did}  ({counts[did]} casts)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=7, markerscale=0.8)

    plt.suptitle("Latent Trajectories Through PCA Space (color = time)", fontsize=13)
    plt.tight_layout()
    save("06_latent_trajectories.png")


# ── Figure 7: Latent dimension activations over time ─────────────────────────

def plot_latent_dims_over_time(P, ts, n_dims=6):
    print("Plotting latent dimensions over time ...")
    order  = np.argsort(ts)
    P_sort = P[order]
    t_sort = ts[order]

    # convert to approximate year for x-axis
    import pandas as pd
    epoch = pd.Timestamp("2000-01-01")
    years = pd.to_datetime(t_sort * 86400 * 1e9 + epoch.value).year + \
            pd.to_datetime(t_sort * 86400 * 1e9 + epoch.value).dayofyear / 365.0

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.scatter(years, P_sort[:, i], s=1, alpha=0.3, color=f"C{i}")
        # rolling mean
        window = max(1, len(years) // 100)
        rolling = np.convolve(P_sort[:, i], np.ones(window)/window, mode="valid")
        ax.plot(years[window-1:], rolling, color=f"C{i}", linewidth=1.5, alpha=0.9)
        ax.set_ylabel(f"z[{i}]", fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[-1].set_xlabel("Year")
    plt.suptitle(f"First {n_dims} Latent Dimensions Over Time", fontsize=13)
    plt.tight_layout()
    save("07_latent_dims_time.png")


# ── Figure 8: UMAP (if available) ─────────────────────────────────────────────

def plot_umap(P, lats, lons):
    if not HAS_UMAP:
        return
    print("Plotting UMAP ...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(P)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=lats,
                           cmap="RdYlBu", s=3, alpha=0.6, linewidths=0)
    plt.colorbar(sc0, ax=axes[0], label="Latitude (°N)")
    axes[0].set_title("UMAP — colored by Latitude")

    sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=lons,
                           cmap="twilight_shifted", s=3, alpha=0.6, linewidths=0)
    plt.colorbar(sc1, ax=axes[1], label="Longitude (°E)")
    axes[1].set_title("UMAP — colored by Longitude")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.suptitle("UMAP of Latent Space", fontsize=13)
    plt.tight_layout()
    save("08_umap.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_records, train, val, probe = load_records()
    P, lats, lons, ts, device_ids = records_to_arrays(all_records)

    print(f"\nLatent vectors: {P.shape}  (n_casts x latent_dim)")
    print(f"Lat range:  {lats.min():.1f} → {lats.max():.1f}")
    print(f"Lon range:  {lons.min():.1f} → {lons.max():.1f}\n")

    plot_pca_variance(P)
    plot_pca_by_lat(P, lats)
    plot_pca_by_lon(P, lons)
    plot_pca_by_season(P, ts)
    plot_geographic(lats, lons, device_ids)
    plot_latent_trajectories(P, ts, device_ids)
    plot_latent_dims_over_time(P, ts)
    plot_umap(P, lats, lons)

    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
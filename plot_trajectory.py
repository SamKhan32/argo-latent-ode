import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

ARCHIVE_PATH   = "results/archive/good_results_041220206"
LATENT_PATH    = f"{ARCHIVE_PATH}/latent_cycles.pt"
CSV_PATH       = "data/processed/PFL1_interp72.csv"

# ── 1. Load latent records ──────────────────────────────────────────────────
data     = torch.load(LATENT_PATH, map_location="cpu", weights_only=False)
all_recs = data["train"] + data["val"] + data["probe"]

cast_meta = {r["cast_id"]: {"device_idx": r["device_idx"], "t": r["t"]}
             for r in all_recs}
cast_ids  = set(cast_meta.keys())

# ── 2. Load CSV and compute per-cast summaries ──────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
df = df[df["wod_unique_cast"].isin(cast_ids)]

def profile_summary(grp):
    T = grp["Temperature"].values
    S = grp["Salinity"].values
    O = grp["Oxygen"].values
    z = grp["z"].values

    def stratification(var, depths):
        mask_surf = depths <= 50
        mask_deep = depths >= 500
        if mask_surf.sum() < 2 or mask_deep.sum() < 2:
            return np.nan
        return np.nanmean(var[mask_surf]) - np.nanmean(var[mask_deep])

    return pd.Series({
        "T_strat": stratification(T, z),
        "S_strat": stratification(S, z),
        "O_mean":  np.nanmean(O),
    })

print("Computing profile summaries...")
summaries = df.groupby("wod_unique_cast").apply(profile_summary).reset_index()
summaries.rename(columns={"wod_unique_cast": "cast_id"}, inplace=True)
summaries["device_idx"] = summaries["cast_id"].map(lambda c: cast_meta[c]["device_idx"])
summaries["t"]          = summaries["cast_id"].map(lambda c: cast_meta[c]["t"])
summaries = summaries.dropna(subset=["T_strat", "S_strat", "O_mean", "t"])
print(f"Clean casts: {len(summaries)}")

# ── 3. Select floats ────────────────────────────────────────────────────────
MIN_CASTS = 8
counts    = summaries.groupby("device_idx").size()
good_devs = counts[counts >= MIN_CASTS].index.tolist()
plot_devs = good_devs[:3]

t_global_min = summaries["t"].min()
t_global_max = summaries["t"].max()

# ── 4. Build plotly figure ───────────────────────────────────────────────────
fig = go.Figure()

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

for i, dev in enumerate(plot_devs):
    sub    = summaries[summaries["device_idx"] == dev].sort_values("t")
    x      = sub["T_strat"].values
    y      = sub["S_strat"].values
    z      = sub["O_mean"].values
    t      = sub["t"].values
    color  = colors[i % len(colors)]

    # Line connecting casts in time order
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=3),
        name=f"Float {dev}",
        legendgroup=f"float_{dev}",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Scatter points colored by time
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=4,
            color=t,
            colorscale="Plasma",
            cmin=t_global_min,
            cmax=t_global_max,
            colorbar=dict(title="Time (days)", thickness=15) if i == 0 else None,
            showscale=(i == 0),
        ),
        name=f"Float {dev} casts",
        legendgroup=f"float_{dev}",
        showlegend=False,
        hovertemplate=(
            f"<b>Float {dev}</b><br>"
            "T strat: %{x:.2f} °C<br>"
            "S strat: %{y:.3f} PSU<br>"
            "Chl:     %{z:.3f} mg/m³<br>"
            "Time:    %{marker.color:.0f} days<br>"
            "<extra></extra>"
        ),
    ))

    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(size=8, color="lime", symbol="circle",
                    line=dict(color="black", width=1)),
        name="Start",
        legendgroup="start",
        showlegend=(i == 0),
        hovertemplate=f"<b>Float {dev} — START</b><br>t={t[0]:.0f} days<extra></extra>",
    ))

    # End marker
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="End",
        legendgroup="end",
        showlegend=(i == 0),
        hovertemplate=f"<b>Float {dev} — END</b><br>t={t[-1]:.0f} days<extra></extra>",
    ))

fig.update_layout(
    title="Float Trajectories in T/S/O Space<br><sup>Points colored by time (plasma scale) — rotate, zoom, hover freely</sup>",
    scene=dict(
        xaxis_title="Temperature Stratification (°C)",
        yaxis_title="Salinity Stratification (PSU)",
        zaxis_title="Mean Chlorophyll (mg/m³)",
    ),
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=0, r=0, b=0, t=60),
    width=1000,
    height=800,
)

out = f"{ARCHIVE_PATH}/trajectory_TSO_3d.html"
fig.write_html(out)
print(f"Saved raw trajectory plot to {out}")

# ── 5. Build latent vectors per cast, aligned to same floats ────────────────
# Map device_idx -> sorted list of (t, latent_vector)
from collections import defaultdict
latent_by_dev = defaultdict(list)
for r in all_recs:
    latent_by_dev[r["device_idx"]].append((r["t"], r["p"]))

# Sort each float's casts by time
for dev in latent_by_dev:
    latent_by_dev[dev].sort(key=lambda x: x[0])

# ── 6. PCA on all latent vectors from the selected floats ───────────────────
all_vecs = []
for dev in plot_devs:
    if len(latent_by_dev[dev]) >= MIN_CASTS:
        for t_val, p in latent_by_dev[dev]:
            all_vecs.append(p)

all_vecs = np.stack(all_vecs, axis=0)  # (N, 32)
pca = PCA(n_components=3)
pca.fit(all_vecs)
var_explained = pca.explained_variance_ratio_
print(f"PCA variance explained: PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}  PC3={var_explained[2]:.1%}  total={sum(var_explained):.1%}")

# ── 7. Build latent plotly figure ────────────────────────────────────────────
fig2 = go.Figure()

for i, dev in enumerate(plot_devs):
    casts = latent_by_dev[dev]
    if len(casts) < MIN_CASTS:
        continue

    t_arr = np.array([c[0] for c in casts])
    p_arr = np.stack([c[1] for c in casts], axis=0)  # (n_casts, 32)
    proj  = pca.transform(p_arr)                      # (n_casts, 3)

    x, y, z = proj[:, 0], proj[:, 1], proj[:, 2]
    color = colors[i % len(colors)]

    fig2.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=3),
        name=f"Float {dev}",
        legendgroup=f"float_{dev}",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig2.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=4,
            color=t_arr,
            colorscale="Plasma",
            cmin=t_global_min,
            cmax=t_global_max,
            colorbar=dict(title="Time (days)", thickness=15) if i == 0 else None,
            showscale=(i == 0),
        ),
        name=f"Float {dev} casts",
        legendgroup=f"float_{dev}",
        showlegend=False,
        hovertemplate=(
            f"<b>Float {dev}</b><br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<br>"
            "Time: %{marker.color:.0f} days<br>"
            "<extra></extra>"
        ),
    ))

    fig2.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(size=8, color="lime", symbol="circle",
                    line=dict(color="black", width=1)),
        name="Start",
        legendgroup="start",
        showlegend=(i == 0),
        hovertemplate=f"<b>Float {dev} — START</b><br>t={t_arr[0]:.0f} days<extra></extra>",
    ))

    fig2.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="End",
        legendgroup="end",
        showlegend=(i == 0),
        hovertemplate=f"<b>Float {dev} — END</b><br>t={t_arr[-1]:.0f} days<extra></extra>",
    ))

fig2.update_layout(
    title=f"Float Trajectories in Latent Space (PCA 3D)<br><sup>Variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%} — rotate, zoom, hover freely</sup>",
    scene=dict(
        xaxis_title=f"PC1 ({var_explained[0]:.1%})",
        yaxis_title=f"PC2 ({var_explained[1]:.1%})",
        zaxis_title=f"PC3 ({var_explained[2]:.1%})",
    ),
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=0, r=0, b=0, t=60),
    width=1000,
    height=800,
)

out2 = f"{ARCHIVE_PATH}/trajectory_latent_pca3d.html"
fig2.write_html(out2)
print(f"Saved latent trajectory plot to {out2}")

# ── 8. 2D PC1 vs time per float ──────────────────────────────────────────────
fig3 = go.Figure()

for i, dev in enumerate(plot_devs):
    casts = latent_by_dev[dev]
    if len(casts) < MIN_CASTS:
        continue

    t_arr = np.array([c[0] for c in casts])
    p_arr = np.stack([c[1] for c in casts], axis=0)
    pc1   = pca.transform(p_arr)[:, 0]

    color = colors[i % len(colors)]

    fig3.add_trace(go.Scatter(
        x=t_arr, y=pc1,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        name=f"Float {dev}",
        hovertemplate=(
            f"<b>Float {dev}</b><br>"
            "Time: %{x:.0f} days<br>"
            "PC1:  %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

fig3.update_layout(
    title=f"Latent Trajectory — PC1 vs Time<br><sup>PC1 captures {var_explained[0]:.1%} of latent variance</sup>",
    xaxis_title="Time (days since epoch)",
    yaxis_title=f"PC1 ({var_explained[0]:.1%} variance)",
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=60, r=20, b=60, t=60),
    width=1000,
    height=500,
)

out3 = f"{ARCHIVE_PATH}/trajectory_latent_pc1_time.html"
fig3.write_html(out3)
print(f"Saved PC1 vs time plot to {out3}")
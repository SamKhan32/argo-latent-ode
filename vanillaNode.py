"""
vanillaNode.py

Vanilla Neural ODE vs GRU baseline operating directly on raw T/S profiles.
No encoder/decoder — state is a flattened (73 depth levels x 2 variables) = 146-dim vector.
Scope: T/S reconstruction and extrapolation only (no oxygen probe).

Mirrors train_dynamics.py structure but loads sequences from PFL1_interp72.csv
instead of latent_cycles.pt.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from globals.config import INTERP_PATH, ODE_HIDDEN, BATCH_SIZE, SEED
DEVICE      = "cuda"
RESULTS_DIR = "results/vanilla"
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_DEPTHS   = 73          # fixed depth grid levels
N_VARS     = 2           # Temperature, Salinity
PROFILE_DIM = N_DEPTHS * N_VARS   # 146

INPUT_VARS  = ["Temperature", "Salinity"]
WINDOW_SIZE = 10          # profiles per training subsequence
EXTRAP_STEPS = 5          # how many steps ahead to evaluate extrapolation
BATCH_SIZE  = 32
N_EPOCHS    = 75
LR          = 1e-3
VAL_FRAC    = 0.15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_profile_sequences(csv_path: str):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["WMO_ID", "datetime", "z"]).reset_index(drop=True)

    t_mean, t_std = df["Temperature"].mean(), df["Temperature"].std()
    s_mean, s_std = df["Salinity"].mean(),    df["Salinity"].std()
    stats = dict(t_mean=t_mean, t_std=t_std, s_mean=s_mean, s_std=s_std)

    sequences = []
    min_len = WINDOW_SIZE + EXTRAP_STEPS

    for wmo_id, float_df in df.groupby("WMO_ID"):
        cast_groups = float_df.groupby("wod_unique_cast", sort=True)
        casts = sorted(cast_groups.groups.keys())

        if len(casts) < min_len:
            continue

        profiles = []
        times    = []
        t0       = None

        for cast_key in casts:
            cast = cast_groups.get_group(cast_key).sort_values("z")

            if len(cast) != N_DEPTHS:
                continue

            temp = (cast["Temperature"].values - t_mean) / (t_std + 1e-8)
            sal  = (cast["Salinity"].values    - s_mean) / (s_std + 1e-8)
            profile = np.concatenate([temp, sal]).astype(np.float32)
            profiles.append(profile)

            dt = cast["datetime"].iloc[0]
            if t0 is None:
                t0 = dt
            times.append((dt - t0).total_seconds() / 86400.0)

        if len(profiles) < min_len:
            continue

        sequences.append({
            "profiles": np.stack(profiles),
            "times":    np.array(times, dtype=np.float32),
            "wmo_id":   str(wmo_id),
        })

    print(f"Loaded {len(sequences)} floats  "
          f"(T μ={t_mean:.2f} σ={t_std:.2f}, S μ={s_mean:.2f} σ={s_std:.2f})")
    return sequences, stats

def train_val_split(sequences, val_frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * val_frac))
    val_idx  = idx[:n_val]
    train_idx = idx[n_val:]
    train = [sequences[i] for i in train_idx]
    val   = [sequences[i] for i in val_idx]
    print(f"Train floats: {len(train)}   Val floats: {len(val)}")
    return train, val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProfileSequenceDataset(Dataset):
    """
    Yields sliding windows of (context_profiles, context_times, target_profiles, target_times).

    context: first WINDOW_SIZE profiles
    target:  next EXTRAP_STEPS profiles (for extrapolation eval)

    During training only the context window is used for the reconstruction loss.
    """
    def __init__(self, sequences, window=WINDOW_SIZE, extrap=EXTRAP_STEPS):
        self.samples = []
        for seq in sequences:
            P = seq["profiles"]   # (T, 146)
            T = seq["times"]      # (T,)
            n = len(P)
            for start in range(0, n - window - extrap + 1):
                ctx_p = P[start : start + window]
                ctx_t = T[start : start + window]
                tgt_p = P[start + window : start + window + extrap]
                tgt_t = T[start + window : start + window + extrap]
                # Shift times so context starts at 0
                t0 = ctx_t[0]
                self.samples.append((
                    torch.tensor(ctx_p),
                    torch.tensor(ctx_t - t0),
                    torch.tensor(tgt_p),
                    torch.tensor(tgt_t - t0),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """f_theta(z) — the RHS of dz/dt = f(z). Time-invariant."""
    def __init__(self, dim=PROFILE_DIM, hidden=ODE_HIDDEN):
        super().__init__()
        layers = []
        in_dim = dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.nfe = 0  # number of function evaluations (diagnostic)

    def forward(self, t, z):
        self.nfe += 1
        return self.net(z)


class VanillaNODE(nn.Module):
    """
    Neural ODE operating directly on flattened T/S profiles.
    Given the first profile as initial condition, integrates forward to
    all requested time points.
    """
    def __init__(self, dim=PROFILE_DIM, hidden=ODE_HIDDEN):
        super().__init__()
        self.func = ODEFunc(dim, hidden)

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z0: (batch, 146)  initial profile
        t:  (T,)          time points to solve at (must start at 0)
        Returns: (T, batch, 146)
        """
        self.func.nfe = 0
        return odeint(self.func, z0, t, method="dopri5",
                      rtol=1e-4, atol=1e-5)


class VanillaGRU(nn.Module):
    """
    GRU baseline operating directly on 146-dim profile sequences.
    Takes full context sequence, predicts next-step profiles auto-regressively
    during extrapolation.
    """
    def __init__(self, dim=PROFILE_DIM, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(dim, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, 146)
        Returns: (batch, T, 146)  one-step-ahead predictions
        """
        h, _ = self.gru(x)          # (batch, T, hidden)
        return self.out(h)           # (batch, T, 146)

    def extrapolate(self, ctx: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        ctx:    (batch, T_ctx, 146)
        Returns (batch, n_steps, 146)
        """
        _, h = self.gru(ctx)          # seed hidden state from context
        preds = []
        x = ctx[:, -1:, :]           # last context profile as first input
        for _ in range(n_steps):
            out, h = self.gru(x, h)
            x = self.out(out)        # predicted profile becomes next input
            preds.append(x)
        return torch.cat(preds, dim=1)  # (batch, n_steps, 146)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, model_type="node"):
    model.train()
    total_loss = 0.0
    for ctx_p, ctx_t, _, _ in loader:
        ctx_p = ctx_p.to(device)   # (B, W, 146)
        ctx_t = ctx_t.to(device)   # (B, W)

        optimizer.zero_grad()

        if model_type == "node":
            # Use first profile as z0, integrate to all context time points
            # Build a common time grid for the batch (use mean times)
            t_grid = ctx_t.mean(dim=0)  # (W,) — shared time axis
            z0     = ctx_p[:, 0, :]     # (B, 146)
            pred   = model(z0, t_grid)  # (W, B, 146)
            pred   = pred.permute(1, 0, 2)  # (B, W, 146)
            loss   = nn.functional.mse_loss(pred, ctx_p)

        else:  # gru
            # Teacher-forced: predict t+1 from t
            inp  = ctx_p[:, :-1, :]   # (B, W-1, 146)
            tgt  = ctx_p[:, 1:,  :]   # (B, W-1, 146)
            pred = model(inp)          # (B, W-1, 146)
            loss = nn.functional.mse_loss(pred, tgt)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, model_type="node"):
    model.eval()
    total_loss = 0.0
    for ctx_p, ctx_t, _, _ in loader:
        ctx_p = ctx_p.to(device)
        ctx_t = ctx_t.to(device)

        if model_type == "node":
            t_grid = ctx_t.mean(dim=0)
            z0     = ctx_p[:, 0, :]
            pred   = model(z0, t_grid).permute(1, 0, 2)
            loss   = nn.functional.mse_loss(pred, ctx_p)
        else:
            inp  = ctx_p[:, :-1, :]
            tgt  = ctx_p[:, 1:,  :]
            pred = model(inp)
            loss = nn.functional.mse_loss(pred, tgt)

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_extrapolation(model, loader, device, model_type="node"):
    """MSE on EXTRAP_STEPS profiles beyond the context window."""
    model.eval()
    total_loss = 0.0
    for ctx_p, ctx_t, tgt_p, tgt_t in loader:
        ctx_p = ctx_p.to(device)
        ctx_t = ctx_t.to(device)
        tgt_p = tgt_p.to(device)
        tgt_t = tgt_t.to(device)

        if model_type == "node":
            # Integrate from z0 through context + extrap time points
            all_t  = torch.cat([ctx_t.mean(0), tgt_t.mean(0)], dim=0)  # (W+E,)
            z0     = ctx_p[:, 0, :]
            traj   = model(z0, all_t).permute(1, 0, 2)  # (B, W+E, 146)
            pred   = traj[:, WINDOW_SIZE:, :]            # (B, E, 146)
        else:
            pred = model.extrapolate(ctx_p, EXTRAP_STEPS)  # (B, E, 146)

        loss = nn.functional.mse_loss(pred, tgt_p)
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    sequences, stats = load_profile_sequences(INTERP_PATH)
    train_seqs, val_seqs = train_val_split(sequences)

    train_ds = ProfileSequenceDataset(train_seqs)
    val_ds   = ProfileSequenceDataset(val_seqs)
    print(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    for model_type in ["node", "gru"]:
        print(f"\n{'='*50}")
        print(f"Training Vanilla {model_type.upper()}")
        print(f"{'='*50}")

        if model_type == "node":
            model = VanillaNODE().to(device)
        else:
            model = VanillaGRU().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5)

        train_losses, val_losses = [], []
        best_val = float("inf")
        best_state = None

        for epoch in range(1, N_EPOCHS + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer, device, model_type)
            vl_loss = eval_epoch(model, val_loader,   device,      model_type)
            scheduler.step(vl_loss)

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
                      f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
                      f"best={best_val:.4f}  ({elapsed:.1f}s)")

        # Load best weights
        model.load_state_dict(best_state)
        model.to(device)

        # Extrapolation
        extrap_loss = eval_extrapolation(model, val_loader, device, model_type)
        print(f"\nVanilla {model_type.upper()} — best val recon: {best_val:.4f}  "
              f"extrap MSE: {extrap_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(RESULTS_DIR, f"vanilla_{model_type}_best.pt")
        torch.save({"model_state": best_state,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_val": best_val,
                    "extrap_loss": extrap_loss}, ckpt_path)
        print(f"Saved checkpoint → {ckpt_path}")

        results[model_type] = {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_val":     best_val,
            "extrap_loss":  extrap_loss,
        }

    # --- Summary ---
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for mt, r in results.items():
        print(f"Vanilla {mt.upper():4s}  best_val_recon={r['best_val']:.4f}  "
              f"extrap_MSE={r['extrap_loss']:.4f}")

    # --- Plot ---
    _plot_training_curves(results)


def _plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for mt, r in results.items():
        label = f"Vanilla {mt.upper()}"
        axes[0].semilogy(r["train_losses"], label=f"{label} train")
        axes[0].semilogy(r["val_losses"],   label=f"{label} val", linestyle="--")
    axes[0].set_title("Reconstruction Loss (T/S MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    mts   = list(results.keys())
    bvals = [results[m]["best_val"]    for m in mts]
    extrs = [results[m]["extrap_loss"] for m in mts]
    x = np.arange(len(mts))
    w = 0.35
    axes[1].bar(x - w/2, bvals, w, label="Best Val Recon")
    axes[1].bar(x + w/2, extrs, w, label="Extrap MSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in mts])
    axes[1].set_yscale("log")
    axes[1].set_title("NODE vs GRU Final Comparison")
    axes[1].set_ylabel("MSE (log scale)")
    axes[1].legend()
    axes[1].grid(True, axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("results", "vanilla_node_vs_gru.png")
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
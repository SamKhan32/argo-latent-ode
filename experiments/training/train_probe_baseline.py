"""
experiments/training/train_probe_baseline.py

Depth-only baseline for the oxygen probe.

Same architecture and training loop as train_probe.py, but the decoder
receives ONLY depth in meters as input — no latent vector. If the probe
head can't beat this baseline, the latent space is not contributing
anything beyond the mean O2-depth relationship.

Usage:
    python run/main.py --stage probe_baseline
or directly:
    python experiments/training/train_probe_baseline.py
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs.config1 import (
    LATENT_DIM, DEPTH_GRID, DECODER_HIDDEN,
    BATCH_SIZE, SEED, TARGET_VARS,
)
from experiments.training.train_probe import SlidingWindowProbeDataset, masked_mse

PROBE_LR     = 1e-3
PROBE_EPOCHS = 100

torch.manual_seed(SEED)


# ── depth-only decoder ────────────────────────────────────────────────────────

class DepthOnlyDecoder(nn.Module):
    """
    Predicts TARGET_VARS from depth alone — no latent vector.
    Same hidden architecture as OxygenDecoderHead for a fair comparison.
    Input: scalar depth in meters. Output: (n_target_vars,) per depth level.
    """

    def __init__(self, hidden=DECODER_HIDDEN):
        super().__init__()
        n_out = len(TARGET_VARS)

        layers = []
        in_dim = 1   # depth only
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_out)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, depth_levels):
        """
        depth_levels : (depth,) float tensor — DEPTH_GRID in meters
        returns      : (depth, n_target_vars)

        Note: no batch dim — same prediction for every sample since there
        is no per-cast information. We expand to (B, D, n_vars) in the
        training loop for loss computation.
        """
        d = depth_levels.view(-1, 1)   # (D, 1)
        return self.mlp(d)             # (D, n_target_vars)


# ── training loop ─────────────────────────────────────────────────────────────

def train_probe_baseline(
    probe_dataset,
    checkpoint_dir="checkpoints",
    checkpoint_name="probe_baseline_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)  # (D,)

    print("Building probe windows...")
    window_ds = SlidingWindowProbeDataset(probe_dataset)
    print(f"Probe windows: {len(window_ds)}")

    n_val   = max(1, int(0.2 * len(window_ds)))
    n_train = len(window_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        window_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = DepthOnlyDecoder(hidden=DECODER_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE_LR)

    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        # ── train ──
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            target = batch["target"].to(device)   # (B, W, D, n_tgt)
            B, W, D, n_tgt = target.shape

            # same prediction for every sample — expand across B*W
            oxy_pred = model(depth_tensor)                          # (D, n_tgt)
            oxy_pred = oxy_pred.unsqueeze(0).expand(B * W, -1, -1) # (B*W, D, n_tgt)

            target_flat = target.reshape(B * W, D, n_tgt)

            loss = masked_mse(oxy_pred, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(device)
                B, W, D, n_tgt = target.shape

                oxy_pred    = model(depth_tensor)
                oxy_pred    = oxy_pred.unsqueeze(0).expand(B * W, -1, -1)
                target_flat = target.reshape(B * W, D, n_tgt)

                val_loss += masked_mse(oxy_pred, target_flat).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{PROBE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nBaseline training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")
    return ckpt_path
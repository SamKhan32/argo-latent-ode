"""
experiments/training/train_gru_probe.py

Stage: GRU probe — frozen encoder + frozen GRU dynamics, train oxygen decoder head.

Direct parallel to train_probe.py but using GRUDynamics instead of ODEFunc.
Losses saved to results/gru_probe_losses.csv.
"""

import os
import time
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from globals.config import (
    LATENT_DIM, DEPTH_GRID,
    BATCH_SIZE, SEED, DECODER_HIDDEN, ODE_HIDDEN,
)
from models.architectures.gru import GRUDynamics
from models.architectures.probe_decoder import OxygenDecoderHead
from experiments.training.train_probe import (
    SlidingWindowProbeDataset,
    masked_mse,
    encode_profiles,
)
from utils.loss_logger import LossLogger

PROBE_LR     = 1e-3
PROBE_EPOCHS = 100
WINDOW_SIZE  = 5

torch.manual_seed(SEED)


def train_gru_probe(
    probe_dataset,
    encoder,
    gru,
    checkpoint_dir="checkpoints",
    checkpoint_name="gru_probe_best.pt",
    log_path="results/gru_probe_losses.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    for model in (encoder, gru):
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    encoder = encoder.to(device)
    gru     = gru.to(device)

    print("Building probe windows...")
    window_ds = SlidingWindowProbeDataset(probe_dataset, WINDOW_SIZE)
    print(f"Probe windows: {len(window_ds)}")

    n_val   = max(1, int(0.2 * len(window_ds)))
    n_train = len(window_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        window_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probe_head = OxygenDecoderHead(latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN).to(device)
    optimizer  = torch.optim.Adam(probe_head.parameters(), lr=PROBE_LR)

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    n_steps = WINDOW_SIZE - 1

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        probe_head.train()
        train_loss = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)
            mask    = batch["mask"].to(device)
            target  = batch["target"].to(device)
            lat     = batch["lat"].to(device)
            lon     = batch["lon"].to(device)

            B, W, D, n_in = profile.shape

            profile_flat = profile.reshape(B * W, D, n_in)
            mask_flat    = mask.reshape(B * W, D, n_in)
            p_flat       = encode_profiles(encoder, profile_flat, mask_flat, device)
            p0           = p_flat.reshape(B, W, LATENT_DIM)[:, 0, :]   # (B, latent_dim)

            # evolve with GRU
            p_traj      = gru(p0, lat[:, 0], lon[:, 0], n_steps)       # (W, B, latent_dim)
            p_pred      = p_traj.permute(1, 0, 2)                       # (B, W, latent_dim)
            p_pred_flat = p_pred.reshape(B * W, LATENT_DIM)

            target_flat = target.reshape(B * W, D, target.shape[-1])
            oxy_pred    = probe_head(p_pred_flat, depth_tensor)
            loss        = masked_mse(oxy_pred, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        probe_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)
                target  = batch["target"].to(device)
                lat     = batch["lat"].to(device)
                lon     = batch["lon"].to(device)

                B, W, D, n_in = profile.shape

                profile_flat = profile.reshape(B * W, D, n_in)
                mask_flat    = mask.reshape(B * W, D, n_in)
                p_flat       = encode_profiles(encoder, profile_flat, mask_flat, device)
                p0           = p_flat.reshape(B, W, LATENT_DIM)[:, 0, :]

                p_traj      = gru(p0, lat[:, 0], lon[:, 0], n_steps)
                p_pred      = p_traj.permute(1, 0, 2)
                p_pred_flat = p_pred.reshape(B * W, LATENT_DIM)

                target_flat = target.reshape(B * W, D, target.shape[-1])
                oxy_pred    = probe_head(p_pred_flat, depth_tensor)
                val_loss   += masked_mse(oxy_pred, target_flat).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{PROBE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": probe_head.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nGRU probe training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path

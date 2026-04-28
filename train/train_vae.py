"""
experiments/training/train_vae.py

Stage 1 (VAE variant) — encoder/decoder training on T/S reconstruction.
Losses saved to results/vae_losses.csv.

Differences from train_encoder.py:
  - Uses VAE instead of Autoencoder
  - forward() returns (recon, mu, log_var)
  - Loss is vae_loss (masked MSE + beta-weighted KL) instead of masked MSE + L2
  - Validation loss is reconstruction-only for a fair comparison with the autoencoder
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from config import (
    LOW_DRIFT_PATH, INTERP_PATH, DEPTH_GRID,
    ENCODER_LR, ENCODER_EPOCHS, BATCH_SIZE, LATENT_DIM,
    ENCODER_HIDDEN, DECODER_HIDDEN,
)
from utils.split import build_splits
from utils.datasets import ArgoProfileDataset
from models.vae import VAE, vae_loss
from utils.loss_logger import LossLogger


def train_vae(
    checkpoint_dir="checkpoints",
    checkpoint_name="vae_best.pt",
    log_path="results/vae_losses.csv",
    beta=1.0,
):
    """
    beta : KL weight in vae_loss.
           1.0  = standard VAE
           <1.0 = beta-VAE (more disentangled latent dims, slightly worse recon)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    df, _ = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test", stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = VAE(latent_dim=LATENT_DIM,
                    encoder_hidden=ENCODER_HIDDEN,
                    decoder_hidden=DECODER_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, ENCODER_EPOCHS + 1):
        t_start = time.time()

        # -- training --
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss    = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)   # (B, D, n_vars)
            mask    = batch["mask"].to(device)       # (B, D, n_vars)

            recon, mu, log_var = model(profile, mask, depth_tensor)
            loss, recon_l, kl_l = vae_loss(recon, profile, mask, mu, log_var, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss       += loss.item()
            train_recon_loss += recon_l.item()
            train_kl_loss    += kl_l.item()

        train_loss       /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss    /= len(train_loader)

        # -- validation --
        # use reconstruction loss only so this number is comparable to the
        # autoencoder's val loss when you plot them side by side
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)

                recon, mu, log_var = model(profile, mask, depth_tensor)
                _, recon_l, _ = vae_loss(recon, profile, mask, mu, log_var, beta=beta)
                val_loss += recon_l.item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:3d}/{ENCODER_EPOCHS}  "
            f"train={train_loss:.4f} (recon={train_recon_loss:.4f} kl={train_kl_loss:.4f})  "
            f"val_recon={val_loss:.4f}  time={elapsed:.1f}s"
        )

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(ckpt_path, stats=train_ds.stats)
            print(f"  -> saved checkpoint (val_recon={best_val_loss:.4f})")

    print(f"\nVAE training complete. Best val recon loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path


if __name__ == "__main__":
    train_vae()

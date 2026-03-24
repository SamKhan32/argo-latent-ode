"""
experiments/training/train_finetune.py

Stage 4 — joint finetuning of encoder + ODE + probe head.

Loads pretrained autoencoder and ODE checkpoints, freezes the T/S decoder,
and finetunes encoder + ODE + probe head jointly with three loss terms:

    loss = ts_recon_raw_loss
         + LAMBDA_ODE * ts_recon_evolved_loss
         + LAMBDA_OXY * oxygen_loss

ts_recon_raw_loss     : decoder(encoder(T/S)) vs T/S  — clean encoder signal
ts_recon_evolved_loss : decoder(ODE(p0))      vs T/S  — interpretability signal
oxygen_loss           : probe_head(ODE(p0))   vs O2   — joint supervision signal

Oxygen targets are z-scored using training data stats before loss computation
to bring the oxygen loss to the same scale as T/S losses.

Losses saved to results/finetune_losses.csv.

Usage:
    python run/main.py --stage finetune
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from globals.config import (
    LATENT_DIM, DEPTH_GRID, BATCH_SIZE, SEED,
    DECODER_HIDDEN, ODE_HIDDEN,
    PROBE_LR, PROBE_EPOCHS,
    LAMBDA_ODE, LAMBDA_OXY,
)
from data.datasets import ArgoJointWindowDataset
from models.architectures.autoencoder import Autoencoder
from models.architectures.ode import ODEFunc
from models.architectures.probe_decoder import OxygenDecoderHead
from utils.loss_logger import LossLogger

WINDOW_SIZE = 5
STRIDE      = 2

ODE_RTOL = 1e-3
ODE_ATOL = 1e-4

T_GRID = torch.arange(0, WINDOW_SIZE, dtype=torch.float32) * 10.0

torch.manual_seed(SEED)


def masked_mse_ts(pred, target, mask):
    """MSE using explicit boolean mask — for T/S reconstruction loss."""
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return ((pred - target)[mask] ** 2).mean()


def masked_mse_oxy(pred, target):
    """MSE using NaN mask — for oxygen loss where missing values are NaN."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return ((pred - target)[mask] ** 2).mean()


def compute_oxy_stats(train_ds):
    """Compute oxygen mean and std from training windows for normalization."""
    all_oxy = []
    for idx in range(len(train_ds)):
        item  = train_ds[idx]
        target = item["target"]   # (W, D, n_target)
        valid  = target[~torch.isnan(target)]
        if len(valid) > 0:
            all_oxy.append(valid)
    all_oxy  = torch.cat(all_oxy)
    oxy_mean = all_oxy.mean().item()
    oxy_std  = all_oxy.std().item()
    oxy_std  = oxy_std if oxy_std > 1e-6 else 1.0
    return oxy_mean, oxy_std


def _forward(encoder, decoder, ode_func, probe_head,
             profile, mask, target, lat, lon,
             depth_tensor, t_grid, device,
             oxy_mean, oxy_std):
    """
    Shared forward pass for train and val loops.
    Returns (loss, loss_ts_raw, loss_ts_evo, loss_oxy).
    """
    B, W, D, n_in = profile.shape

    profile_flat = profile.reshape(B * W, D, n_in)
    mask_flat    = mask.reshape(B * W, D, n_in)
    p_flat       = encoder(profile_flat, mask_flat)          # (B*W, latent_dim)
    p            = p_flat.reshape(B, W, LATENT_DIM)          # (B, W, latent_dim)

    # T/S reconstruction from raw encoder output
    recon_raw   = decoder(p_flat, depth_tensor)              # (B*W, D, n_vars)
    loss_ts_raw = masked_mse_ts(recon_raw, profile_flat, mask_flat.bool())

    # ODE rollout from first cycle
    lat0 = lat[:, 0:1]
    lon0 = lon[:, 0:1]
    z0   = torch.cat([p[:, 0, :], lat0, lon0], dim=-1)

    z_pred      = odeint(ode_func, z0, t_grid, method="dopri5",
                         rtol=ODE_RTOL, atol=ODE_ATOL)
    p_pred      = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)  # (B, W, latent_dim)
    p_pred_flat = p_pred.reshape(B * W, LATENT_DIM)

    # T/S reconstruction from ODE-evolved latent
    recon_evo   = decoder(p_pred_flat, depth_tensor)
    loss_ts_evo = masked_mse_ts(recon_evo, profile_flat, mask_flat.bool())

    # Oxygen prediction from ODE-evolved latent — normalize target
    target_flat = target.reshape(B * W, D, target.shape[-1])
    target_norm = (target_flat - oxy_mean) / oxy_std
    oxy_pred    = probe_head(p_pred_flat, depth_tensor)
    loss_oxy    = masked_mse_oxy(oxy_pred, target_norm)

    loss = loss_ts_raw + LAMBDA_ODE * loss_ts_evo + LAMBDA_OXY * loss_oxy

    return loss, loss_ts_raw, loss_ts_evo, loss_oxy


def train_finetune(
    probe_dataset,
    autoencoder_checkpoint="checkpoints/autoencoder_best.pt",
    ode_checkpoint="checkpoints/ode_best.pt",
    probe_checkpoint="checkpoints/probe_head_best.pt",
    checkpoint_dir="checkpoints",
    checkpoint_name="finetune_best.pt",
    log_path="results/finetune_losses.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)
    t_grid       = T_GRID.to(device)

    # ── Load pretrained models ────────────────────────────────────────────────
    print("Loading pretrained checkpoints...")
    autoencoder, _ = Autoencoder.load(autoencoder_checkpoint, device=device)
    encoder  = autoencoder.encoder.to(device)
    decoder  = autoencoder.decoder.to(device)

    ode_func = ODEFunc(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    ode_ckpt = torch.load(ode_checkpoint, map_location=device, weights_only=False)
    ode_func.load_state_dict(ode_ckpt["model_state"])

    probe_head = OxygenDecoderHead(latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN).to(device)
    if os.path.exists(probe_checkpoint):
        probe_ckpt = torch.load(probe_checkpoint, map_location=device, weights_only=False)
        probe_head.load_state_dict(probe_ckpt["model_state"])
        print("  Loaded pretrained probe head.")
    else:
        print("  No probe checkpoint found — probe head initialised from scratch.")

    # ── Freeze T/S decoder only ───────────────────────────────────────────────
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Building joint windows...")
    window_ds = ArgoJointWindowDataset(probe_dataset, WINDOW_SIZE, STRIDE)
    print(f"Joint windows: {len(window_ds)}")

    n_val   = max(1, int(0.2 * len(window_ds)))
    n_train = len(window_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        window_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # ── Oxygen normalization stats from training data ─────────────────────────
    print("Computing oxygen normalization stats...")
    oxy_mean, oxy_std = compute_oxy_stats(train_ds)
    print(f"Oxygen stats — mean: {oxy_mean:.2f}, std: {oxy_std:.2f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = (
        list(encoder.parameters()) +
        list(ode_func.parameters()) +
        list(probe_head.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=PROBE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROBE_EPOCHS, eta_min=1e-6
    )

    logger        = LossLogger(log_path, extras=["val_ts_raw", "val_ts_evo", "val_oxy"])
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        encoder.train()
        ode_func.train()
        probe_head.train()

        train_loss = train_ts_raw = train_ts_evo = train_oxy = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)
            mask    = batch["mask"].to(device)
            target  = batch["target"].to(device)
            lat     = batch["lat"].to(device)
            lon     = batch["lon"].to(device)

            loss, loss_ts_raw, loss_ts_evo, loss_oxy = _forward(
                encoder, decoder, ode_func, probe_head,
                profile, mask, target, lat, lon,
                depth_tensor, t_grid, device,
                oxy_mean, oxy_std,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss   += loss.item()
            train_ts_raw += loss_ts_raw.item()
            train_ts_evo += loss_ts_evo.item()
            train_oxy    += loss_oxy.item()

        n = len(train_loader)
        train_loss   /= n
        train_ts_raw /= n
        train_ts_evo /= n
        train_oxy    /= n

        # ── Validate ──────────────────────────────────────────────────────────
        encoder.eval()
        ode_func.eval()
        probe_head.eval()

        val_loss = val_ts_raw = val_ts_evo = val_oxy = 0.0

        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)
                target  = batch["target"].to(device)
                lat     = batch["lat"].to(device)
                lon     = batch["lon"].to(device)

                loss, loss_ts_raw, loss_ts_evo, loss_oxy = _forward(
                    encoder, decoder, ode_func, probe_head,
                    profile, mask, target, lat, lon,
                    depth_tensor, t_grid, device,
                    oxy_mean, oxy_std,
                )

                val_loss   += loss.item()
                val_ts_raw += loss_ts_raw.item()
                val_ts_evo += loss_ts_evo.item()
                val_oxy    += loss_oxy.item()

        n = len(val_loader)
        val_loss   /= n
        val_ts_raw /= n
        val_ts_evo /= n
        val_oxy    /= n

        elapsed    = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{PROBE_EPOCHS}  "
            f"loss={val_loss:.4f}  "
            f"ts_raw={val_ts_raw:.4f}  "
            f"ts_evo={val_ts_evo:.4f}  "
            f"oxy={val_oxy:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        logger.log(epoch, train_loss, val_loss,
                   val_ts_raw=val_ts_raw,
                   val_ts_evo=val_ts_evo,
                   val_oxy=val_oxy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder_state":    encoder.state_dict(),
                "ode_state":        ode_func.state_dict(),
                "probe_head_state": probe_head.state_dict(),
                "oxy_mean":         oxy_mean,
                "oxy_std":          oxy_std,
            }, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

        scheduler.step()

    print(f"\nFinetuning complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path
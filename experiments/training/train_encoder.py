import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config1 import (
    ENCODER_LR, ENCODER_EPOCHS, BATCH_SIZE,
    LOW_DRIFT_PATH, INTERP_PATH, DEPTH_GRID,
)
from data.split import build_splits
from data.datasets import ArgoProfileDataset
from models.architectures.autoencoder import Autoencoder


## Loss ##

def masked_mse(recon, target, mask):
    """MSE computed only over valid (non-missing) observations."""
    mask = mask.float()
    loss = ((recon - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    return loss


## Collate ##
## Since data is now on a fixed depth grid, all profiles have the same depth
## dimension — no padding needed. collate_fn just stacks tensors normally.

def collate_fn(batch):
    return {
        "profile": torch.stack([item["profile"] for item in batch]),
        "mask":    torch.stack([item["mask"]    for item in batch]),
        "lat":     torch.stack([item["lat"]     for item in batch]),
        "lon":     torch.stack([item["lon"]     for item in batch]),
        "t":       torch.stack([item["t"]       for item in batch]),
        "wmo_id":  [item["wmo_id"]  for item in batch],
        "cast_id": [item["cast_id"] for item in batch],
    }


## Training loop ##

def train_encoder(checkpoint_dir="checkpoints", checkpoint_name="autoencoder_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # fixed depth grid as a tensor — passed to decoder each forward call
    depth_levels = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    # --- data ---
    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test", stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # --- model ---
    model     = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)

    # --- training ---
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, ENCODER_EPOCHS + 1):

        ## train ##
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            profile = batch["profile"].to(device)
            mask    = batch["mask"].to(device)

            recon, p = model(profile, mask, depth_levels)
            loss     = masked_mse(recon, profile, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ## validate ##
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)
                recon, _ = model(profile, mask, depth_levels)
                val_loss += masked_mse(recon, profile, mask).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{ENCODER_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        ## checkpoint ##
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(checkpoint_path, stats=train_ds.stats)
            print(f"  -> saved best checkpoint (val={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_encoder()
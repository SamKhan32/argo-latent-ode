import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint  # switched from odeint_adjoint

from configs.config1 import ODE_LR, ODE_EPOCHS, BATCH_SIZE, LATENT_DIM
from data.datasets import ArgoLatentDataset
from models.architectures.ode import ODEFunc

WINDOW_SIZE = 5
STRIDE      = 2

ODE_RTOL = 1e-3
ODE_ATOL = 1e-4


## SlidingWindowDataset ##
## Builds all valid windows of size WINDOW_SIZE from ArgoLatentDataset
## One item = one window of consecutive casts from the same float

class SlidingWindowDataset(Dataset):

    def __init__(self, latent_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.records     = latent_dataset.records
        self.window_size = window_size
        self.windows     = []

        from collections import defaultdict
        device_records = defaultdict(list)
        for r in self.records:
            device_records[r["device_idx"]].append(r)

        for device_idx, recs in device_records.items():
            recs = sorted(recs, key=lambda r: r["t"])
            n = len(recs)
            for start in range(0, n - window_size + 1, stride):
                window = recs[start : start + window_size]
                times  = [r["t"] for r in window]
                # skip windows with duplicate or non-increasing timestamps
                if all(times[i] < times[i+1] for i in range(len(times)-1)):
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t0     = window[0]["t"]

        p   = torch.stack([torch.tensor(r["p"],   dtype=torch.float32) for r in window])
        lat = torch.tensor([r["lat"] for r in window], dtype=torch.float32)
        lon = torch.tensor([r["lon"] for r in window], dtype=torch.float32)
        t   = torch.tensor([r["t"] - t0 for r in window], dtype=torch.float32)  # relative time

        return {"p": p, "lat": lat, "lon": lon, "t": t}   # all (window_size, ...)


## Training loop ##

def train_ode(
    latent_path="checkpoints/latent_cycles.pt",
    checkpoint_dir="checkpoints",
    checkpoint_name="ode_best.pt",
    encoder_checkpoint=None,
    debug=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load precomputed latent datasets ---
    print(f"Loading latent cycles from {latent_path}")
    ckpt = torch.load(latent_path, map_location="cpu", weights_only=False)

    latent_train = ArgoLatentDataset(ckpt["train"])
    latent_val   = ArgoLatentDataset(ckpt["val"])

    print("Building train windows...")
    train_windows = SlidingWindowDataset(latent_train, WINDOW_SIZE, STRIDE)
    print(f"Train windows: {len(train_windows)}")

    print("Building val windows...")
    val_windows = SlidingWindowDataset(latent_val, WINDOW_SIZE, STRIDE)
    print(f"Val windows: {len(val_windows)}")

    train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_windows,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ode_func  = ODEFunc().to(device)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=ODE_LR)
    loss_fn   = nn.MSELoss()

    best_val_loss  = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    for epoch in range(1, ODE_EPOCHS + 1):
        t_start = time.time()

        ## train ##
        ode_func.train()
        train_loss = 0.0

        for batch in train_loader:
            p   = batch["p"].to(device)     # (batch, window, latent_dim)
            lat = batch["lat"].to(device)   # (batch, window)
            lon = batch["lon"].to(device)   # (batch, window)
            t   = batch["t"].to(device)     # (batch, window)

            batch_loss = 0.0
            for i in range(p.shape[0]):
                t_i  = t[i]
                p_i  = p[i]
                lat0 = lat[i, 0].reshape(1, 1)
                lon0 = lon[i, 0].reshape(1, 1)
                z0   = torch.cat([p_i[0:1], lat0, lon0], dim=-1)

                z_pred = odeint(ode_func, z0, t_i, method="dopri5",
                                rtol=ODE_RTOL, atol=ODE_ATOL)
                p_pred = z_pred[:, 0, :LATENT_DIM]

                batch_loss += loss_fn(p_pred, p_i)

            loss = batch_loss / p.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ## validate ##
        ode_func.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                p   = batch["p"].to(device)
                lat = batch["lat"].to(device)
                lon = batch["lon"].to(device)
                t   = batch["t"].to(device)

                batch_loss = 0.0
                for i in range(p.shape[0]):
                    t_i  = t[i]
                    p_i  = p[i]
                    lat0 = lat[i, 0].reshape(1, 1)
                    lon0 = lon[i, 0].reshape(1, 1)
                    z0   = torch.cat([p_i[0:1], lat0, lon0], dim=-1)

                    z_pred = odeint(ode_func, z0, t_i, method="dopri5",
                                    rtol=ODE_RTOL, atol=ODE_ATOL)
                    p_pred = z_pred[:, 0, :LATENT_DIM]

                    batch_loss += loss_fn(p_pred, p_i)

                val_loss += (batch_loss / p.shape[0]).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{ODE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")


        ## checkpoint ##
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": ode_func.state_dict()}, checkpoint_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_ode()
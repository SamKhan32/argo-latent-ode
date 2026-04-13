import torch
import numpy as np

ckpt = torch.load('checkpoints/latent_cycles.pt', map_location='cpu', weights_only=False)

# check lat/lon distribution of train vs val
train_lats = np.array([r['lat'] for r in ckpt['train']])
val_lats   = np.array([r['lat'] for r in ckpt['val']])
train_lons = np.array([r['lon'] for r in ckpt['train']])
val_lons   = np.array([r['lon'] for r in ckpt['val']])

print("Train lat:", train_lats.mean().round(2), "±", train_lats.std().round(2))
print("Val   lat:", val_lats.mean().round(2),   "±", val_lats.std().round(2))
print("Train lon:", train_lons.mean().round(2), "±", train_lons.std().round(2))
print("Val   lon:", val_lons.mean().round(2),   "±", val_lons.std().round(2))

# check how many unique floats in val
val_devices = set(r['device_idx'] for r in ckpt['val'])
train_devices = set(r['device_idx'] for r in ckpt['train'])
print(f"\nTrain floats: {len(train_devices)}, Val floats: {len(val_devices)}")
print(f"Overlap: {len(train_devices & val_devices)}")

val_norms = np.linalg.norm(
    np.stack([r['p'] for r in ckpt['val']]), axis=1
)
print("Val latent norm — mean:", val_norms.mean().round(2),
      "max:", val_norms.max().round(2),
      "% > 10:", (val_norms > 10).mean().round(3))

train_norms = np.linalg.norm(
    np.stack([r['p'] for r in ckpt['train']]), axis=1
)
print("Train latent norm — mean:", train_norms.mean().round(2),
      "max:", train_norms.max().round(2),
      "% > 10:", (train_norms > 10).mean().round(3))
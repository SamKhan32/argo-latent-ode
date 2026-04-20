import argparse
import torch

from globals.config import (
    LOW_DRIFT_PATH, INTERP_PATH,
    LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN, ODE_HIDDEN,
)
from data.split import build_splits
from data.datasets import ArgoProfileDataset, ArgoLatentDataset, ArgoProbeDataset
from models.architectures.vae import VAE                        # <-- VAE
from models.architectures.ode import ODEFunc
from models.architectures.gru import GRUDynamics
from experiments.training.train_vae import train_vae            # <-- needs to be written
from experiments.training.train_node import train_ode
from experiments.training.train_probe import train_probe
from experiments.training.train_probe_baseline import train_probe_baseline
from experiments.training.train_gru import train_gru
from experiments.training.train_gru_probe import train_gru_probe
from experiments.evaluation.extrapolation import run_extrapolation
from experiments.training.train_node_curriculum import train_ode_curriculum
from utils.seeding import set_seed

## checkpoint paths — separate from autoencoder so nothing gets overwritten
VAE_CHECKPOINT    = "checkpoints/vae_best.pt"
LATENT_PATH       = "checkpoints/vae_latent_cycles.pt"
ODE_CHECKPOINT    = "checkpoints/vae_ode_best.pt"
GRU_CHECKPOINT    = "checkpoints/vae_gru_best.pt"

set_seed()


## Stages ##

def stage_split():
    print("=== Stage: split ===")
    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    print("Split complete.")
    return df, split_map


def stage_vae_encoder():
    print("=== Stage: vae_encoder ===")
    # train_vae() should mirror train_encoder() but:
    #   - construct VAE instead of Autoencoder
    #   - call vae_loss(recon, target, mask, mu, log_var) instead of masked MSE
    #   - unpack (recon, mu, log_var) from model.forward()
    #   - save to VAE_CHECKPOINT
    return train_vae()


def stage_encode(checkpoint_path=VAE_CHECKPOINT,
                 latent_path=LATENT_PATH):
    print("=== Stage: encode ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, _ = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test",  stats=train_ds.stats)
    probe_ds = ArgoProfileDataset(df, split="probe", stats=train_ds.stats)

    # VAE.load has same signature as Autoencoder.load
    model, _ = VAE.load(checkpoint_path, device=device)

    all_wmo_ids = df["WMO_ID"].unique()
    wmo_to_idx  = {wmo: i for i, wmo in enumerate(sorted(all_wmo_ids))}

    # encoder interface is unchanged — still takes (profile, mask) -> mu
    # at eval time reparameterize() returns mu directly (no sampling)
    latent_train = ArgoLatentDataset.from_encoder(train_ds, model.encoder, device, wmo_to_idx)
    latent_val   = ArgoLatentDataset.from_encoder(val_ds,   model.encoder, device, wmo_to_idx)
    latent_probe = ArgoLatentDataset.from_encoder(probe_ds, model.encoder, device, wmo_to_idx)

    print(f"Latent train: {len(latent_train)} casts")
    print(f"Latent val:   {len(latent_val)} casts")
    print(f"Latent probe: {len(latent_probe)} casts")

    torch.save({
        "train":      latent_train.records,
        "val":        latent_val.records,
        "probe":      latent_probe.records,
        "wmo_to_idx": wmo_to_idx,
    }, latent_path)
    print(f"Saved VAE latent cycles to {latent_path}")

    return latent_train, latent_val, latent_probe, wmo_to_idx


def stage_ode(latent_path=LATENT_PATH):
    print("=== Stage: ode ===")
    return train_ode(latent_path=latent_path)


def stage_ode_curriculum(latent_path=LATENT_PATH):
    print("=== Stage: ode_curriculum ===")
    return train_ode_curriculum(latent_path=latent_path)


def _load_probe_dataset(vae_checkpoint=VAE_CHECKPOINT):
    """Helper — build probe dataset with consistent normalization stats."""
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    model, _ = VAE.load(vae_checkpoint, device=device)
    print(f"Probe casts: {len(probe_ds)}")
    return probe_ds, model.encoder, device


def stage_probe(
    vae_checkpoint=VAE_CHECKPOINT,
    ode_checkpoint=ODE_CHECKPOINT,
):
    print("=== Stage: probe ===")
    probe_ds, encoder, device = _load_probe_dataset(vae_checkpoint)

    ode_func = ODEFunc(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    ode_ckpt = torch.load(ode_checkpoint, map_location=device, weights_only=False)
    ode_func.load_state_dict(ode_ckpt["model_state"])

    return train_probe(probe_ds, encoder, ode_func)


def stage_probe_baseline(vae_checkpoint=VAE_CHECKPOINT):
    print("=== Stage: probe_baseline ===")
    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    print(f"Probe casts: {len(probe_ds)}")
    return train_probe_baseline(probe_ds)


def stage_gru(latent_path=LATENT_PATH):
    print("=== Stage: gru ===")
    return train_gru(latent_path=latent_path)


def stage_gru_probe(
    vae_checkpoint=VAE_CHECKPOINT,
    gru_checkpoint=GRU_CHECKPOINT,
):
    print("=== Stage: gru_probe ===")
    probe_ds, encoder, device = _load_probe_dataset(vae_checkpoint)

    gru      = GRUDynamics(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    gru_ckpt = torch.load(gru_checkpoint, map_location=device, weights_only=False)
    gru.load_state_dict(gru_ckpt["model_state"])

    return train_gru_probe(probe_ds, encoder, gru)


def stage_extrapolation(latent_path=LATENT_PATH):
    print("=== Stage: extrapolation ===")
    return run_extrapolation(latent_path=latent_path)


## Main ##

STAGES = [
    "split", "vae_encoder", "encode", "ode", "ode_curriculum",
    "probe", "probe_baseline", "gru", "gru_probe", "extrapolation", "all",
]


def main():
    parser = argparse.ArgumentParser(description="VAE variant of the Ocean Dynamics Latent ODE pipeline")
    parser.add_argument("--stage",          type=str, choices=STAGES, default="all")
    parser.add_argument("--checkpoint",     type=str, default=VAE_CHECKPOINT)
    parser.add_argument("--ode_checkpoint", type=str, default=ODE_CHECKPOINT)
    parser.add_argument("--gru_checkpoint", type=str, default=GRU_CHECKPOINT)
    parser.add_argument("--latent",         type=str, default=LATENT_PATH)
    args = parser.parse_args()

    if args.stage == "split":
        stage_split()
    elif args.stage == "vae_encoder":
        stage_vae_encoder()
    elif args.stage == "encode":
        stage_encode(args.checkpoint, args.latent)
    elif args.stage == "ode":
        stage_ode(args.latent)
    elif args.stage == "ode_curriculum":
        stage_ode_curriculum(args.latent)
    elif args.stage == "probe":
        stage_probe(args.checkpoint, args.ode_checkpoint)
    elif args.stage == "probe_baseline":
        stage_probe_baseline(args.checkpoint)
    elif args.stage == "gru":
        stage_gru(args.latent)
    elif args.stage == "gru_probe":
        stage_gru_probe(args.checkpoint, args.gru_checkpoint)
    elif args.stage == "extrapolation":
        stage_extrapolation(args.latent)
    elif args.stage == "all":
        stage_split()
        checkpoint_path = stage_vae_encoder()
        stage_encode(checkpoint_path, args.latent)
        stage_ode_curriculum(args.latent)
        stage_gru(args.latent)
        stage_probe(checkpoint_path, args.ode_checkpoint)
        stage_probe_baseline(checkpoint_path)
        stage_gru_probe(checkpoint_path, args.gru_checkpoint)


if __name__ == "__main__":
    main()

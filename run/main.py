import argparse
import torch

from configs.config1 import (
    LOW_DRIFT_PATH, INTERP_PATH,
    LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN, ODE_HIDDEN,
)
from data.split import build_splits
from data.datasets import ArgoProfileDataset, ArgoLatentDataset, ArgoProbeDataset
from models.architectures.autoencoder import Autoencoder
from models.architectures.ode import ODEFunc
from models.architectures.gru import GRUDynamics
from experiments.training.train_encoder import train_encoder
from experiments.training.train_node import train_ode
from experiments.training.train_probe import train_probe
from experiments.training.train_probe_baseline import train_probe_baseline
from experiments.training.train_gru import train_gru
from experiments.training.train_gru_probe import train_gru_probe


## Stages ##

def stage_split():
    print("=== Stage: split ===")
    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    print("Split complete.")
    return df, split_map


def stage_encoder():
    print("=== Stage: encoder ===")
    return train_encoder()


def stage_encode(checkpoint_path="checkpoints/autoencoder_best.pt",
                 latent_path="checkpoints/latent_cycles.pt"):
    print("=== Stage: encode ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, _ = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test",  stats=train_ds.stats)
    probe_ds = ArgoProfileDataset(df, split="probe", stats=train_ds.stats)

    model, _ = Autoencoder.load(checkpoint_path, device=device)

    all_wmo_ids = df["WMO_ID"].unique()
    wmo_to_idx  = {wmo: i for i, wmo in enumerate(sorted(all_wmo_ids))}

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
    print(f"Saved latent cycles to {latent_path}")

    return latent_train, latent_val, latent_probe, wmo_to_idx


def stage_ode(latent_path="checkpoints/latent_cycles.pt"):
    print("=== Stage: ode ===")
    return train_ode(latent_path=latent_path)


def _load_probe_dataset(autoencoder_checkpoint):
    """Helper — build probe dataset with consistent normalization stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _     = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds  = ArgoProfileDataset(df, split="train")
    probe_ds  = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    model, _  = Autoencoder.load(autoencoder_checkpoint, device=device)
    print(f"Probe casts: {len(probe_ds)}")
    return probe_ds, model.encoder, device


def stage_probe(
    autoencoder_checkpoint="checkpoints/autoencoder_best.pt",
    ode_checkpoint="checkpoints/ode_best.pt",
):
    print("=== Stage: probe ===")
    probe_ds, encoder, device = _load_probe_dataset(autoencoder_checkpoint)

    ode_func = ODEFunc(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    ode_ckpt = torch.load(ode_checkpoint, map_location=device, weights_only=False)
    ode_func.load_state_dict(ode_ckpt["model_state"])

    return train_probe(probe_ds, encoder, ode_func)


def stage_probe_baseline(autoencoder_checkpoint="checkpoints/autoencoder_best.pt"):
    print("=== Stage: probe_baseline ===")
    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    print(f"Probe casts: {len(probe_ds)}")
    return train_probe_baseline(probe_ds)


def stage_gru(latent_path="checkpoints/latent_cycles.pt"):
    print("=== Stage: gru ===")
    return train_gru(latent_path=latent_path)


def stage_gru_probe(
    autoencoder_checkpoint="checkpoints/autoencoder_best.pt",
    gru_checkpoint="checkpoints/gru_best.pt",
):
    print("=== Stage: gru_probe ===")
    probe_ds, encoder, device = _load_probe_dataset(autoencoder_checkpoint)

    gru = GRUDynamics(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    gru_ckpt = torch.load(gru_checkpoint, map_location=device, weights_only=False)
    gru.load_state_dict(gru_ckpt["model_state"])

    return train_gru_probe(probe_ds, encoder, gru)


## Main ##

STAGES = ["split", "encoder", "encode", "ode", "probe", "probe_baseline", "gru", "gru_probe", "all"]

def main():
    parser = argparse.ArgumentParser(description="Ocean Dynamics Latent ODE pipeline")
    parser.add_argument("--stage",          type=str, choices=STAGES, default="all")
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/autoencoder_best.pt")
    parser.add_argument("--ode_checkpoint", type=str, default="checkpoints/ode_best.pt")
    parser.add_argument("--gru_checkpoint", type=str, default="checkpoints/gru_best.pt")
    parser.add_argument("--latent",         type=str, default="checkpoints/latent_cycles.pt")
    args = parser.parse_args()

    if args.stage == "split":
        stage_split()
    elif args.stage == "encoder":
        stage_encoder()
    elif args.stage == "encode":
        stage_encode(args.checkpoint, args.latent)
    elif args.stage == "ode":
        stage_ode(args.latent)
    elif args.stage == "probe":
        stage_probe(args.checkpoint, args.ode_checkpoint)
    elif args.stage == "probe_baseline":
        stage_probe_baseline(args.checkpoint)
    elif args.stage == "gru":
        stage_gru(args.latent)
    elif args.stage == "gru_probe":
        stage_gru_probe(args.checkpoint, args.gru_checkpoint)
    elif args.stage == "all":
        stage_split()
        checkpoint_path = stage_encoder()
        stage_encode(checkpoint_path, args.latent)
        stage_ode(args.latent)
        stage_probe(checkpoint_path, args.ode_checkpoint)
        stage_probe_baseline(checkpoint_path)
        stage_gru(args.latent)
        stage_gru_probe(checkpoint_path, args.gru_checkpoint)


if __name__ == "__main__":
    main()

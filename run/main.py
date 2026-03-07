import argparse
import torch

from configs.config1 import LOW_DRIFT_PATH, INTERP_PATH
from data.split import build_splits
from data.datasets import ArgoProfileDataset, ArgoLatentDataset
from models.architectures.autoencoder import Autoencoder
from experiments.training.train_encoder import train_encoder
from experiments.training.train_node import train_ode


## Stages ##

def stage_split():
    print("=== Stage: split ===")
    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    print("Split complete.")
    return df, split_map


def stage_encoder():
    print("=== Stage: encoder ===")
    checkpoint_path = train_encoder()
    return checkpoint_path


def stage_encode(checkpoint_path="checkpoints/autoencoder_best.pt",
                 latent_path="checkpoints/latent_cycles.pt"):
    """Run encoder over all splits and save ArgoLatentDatasets to disk."""
    print("=== Stage: encode ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test",  stats=train_ds.stats)
    probe_ds = ArgoProfileDataset(df, split="probe", stats=train_ds.stats)

    model, stats = Autoencoder.load(checkpoint_path, device=device)

    # integer-encode WMO_IDs consistently across all splits
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
    checkpoint_path = train_ode(latent_path=latent_path)
    return checkpoint_path


## Main ##

STAGES = ["split", "encoder", "encode", "ode", "all"]

def main():
    parser = argparse.ArgumentParser(description="Ocean Dynamics Latent ODE pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default="all",
        help=f"Pipeline stage to run: {STAGES}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/autoencoder_best.pt",
        help="Path to autoencoder checkpoint (used by encode/ode stages)",
    )
    parser.add_argument(
        "--latent",
        type=str,
        default="checkpoints/latent_cycles.pt",
        help="Path to save/load latent cycles (used by encode/ode stages)",
    )
    args = parser.parse_args()

    if args.stage == "split":
        stage_split()

    elif args.stage == "encoder":
        stage_encoder()

    elif args.stage == "encode":
        stage_encode(args.checkpoint, args.latent)

    elif args.stage == "ode":
        stage_ode(args.latent)

    elif args.stage == "all":
        stage_split()
        checkpoint_path = stage_encoder()
        stage_encode(checkpoint_path, args.latent)
        stage_ode(args.latent)


if __name__ == "__main__":
    main()
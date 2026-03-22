"""
experiments/evaluation/baseline_registry.py

Registry of dynamics models for the extrapolation benchmark.
To add a new baseline, add an entry to BASELINES following the same structure.

Each entry:
    model_class   : class to instantiate
    checkpoint    : default checkpoint path
    kwargs        : constructor kwargs
    kind          : "ode" | "gru" — controls how forward pass is called
"""

from models.architectures.ode import ODEFunc
from models.architectures.gru import GRUDynamics
from globals.config import LATENT_DIM, ODE_HIDDEN

BASELINES = {
    "ode": {
        "model_class": ODEFunc,
        "checkpoint":  "checkpoints/ode_best.pt",
        "kwargs":      {"latent_dim": LATENT_DIM, "hidden": ODE_HIDDEN},
        "kind":        "ode",
    },
    "gru": {
        "model_class": GRUDynamics,
        "checkpoint":  "checkpoints/gru_best.pt",
        "kwargs":      {"latent_dim": LATENT_DIM, "hidden": ODE_HIDDEN},
        "kind":        "gru",
    },
}

# Depth-only baseline val loss — drawn as a flat reference line in plots.
DEPTH_ONLY_VAL_LOSS = 0.0274

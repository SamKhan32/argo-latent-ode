# Latent ODE Modeling of Oceanographic Float Data

## Overview

This project applies Latent Neural Ordinary Differential Equations (Latent ODEs) to oceanographic profiling float data (Argo). The model learns continuous-time latent dynamics from sparse, irregularly sampled vertical ocean profiles, with the goal of reconstructing observed variables and — critically — generalizing to unseen variables via a swappable decoder head.

---

## Motivation

Oceanographic float data presents a core challenge for standard deep learning: profiles are irregularly spaced in time, sparse across the water column, and heterogeneous across cruises and regions. Classical numerical ocean models address this with dense grids and strong physical assumptions that may not hold for complex or poorly characterized variables.

This project investigates whether a learned latent dynamical system can:

- Capture the underlying continuous-time structure of ocean profiles
- Reconstruct observed variables from a compact latent representation
- Generalize to variables unseen during training by swapping the decoder head

---

## Method

The model follows the Latent ODE framework:

1. **Encoder**
   Maps irregularly sampled Argo float profiles to a distribution over latent initial states z₀ via a variational encoder.

2. **Neural ODE**
   Evolves the latent state forward in continuous time:

   ```
   dz/dt = f_θ(z, t)
   ```

   where f_θ is a neural network. Irregular sampling is handled natively by the ODE solver.

3. **Decoder**
   Maps the latent trajectory back to observed variables. The decoder is designed to be swappable — after training, a new decoder head can be attached and trained to reconstruct variables not seen during the original training phase.

Training uses a variational objective (ELBO) with reconstruction and KL divergence terms.

---

## Research Questions

- Can a Latent ODE learn physically meaningful dynamics from sparse Argo profiles?
- Does the latent space generalize — i.e., can a new decoder head reconstruct unseen ocean variables from the pre-trained latent trajectory?
- How does model performance vary with latent dimension size, ODE network depth, and solver choice?

---

## Data

- **Source:** Argo profiling float dataset
- **Variables:** Temperature, salinity, dissolved oxygen, and additional biogeochemical tracers (where available)
- **Structure:** Depth-resolved vertical profiles, irregularly sampled in time and space

Preprocessing includes:

- Converting ragged profile structures into pointwise tabular format
- Group-aware train/validation/test splits by cast/profile index
- Normalization using training statistics only

---

## Experiments

- Reconstruction accuracy on held-out profiles
- Latent trajectory smoothness and stability
- Decoder transfer: training a new head to reconstruct unseen variables from frozen latent dynamics
- Sensitivity analysis over latent dimension and ODE depth

Baselines include discrete-time RNN/GRU models and classical interpolation approaches.

---

## Technologies

- Python
- PyTorch
- torchdiffeq
- xarray / pandas / NumPy

---

## Status

Active development. Ongoing experimentation with latent dimensionality, solver selection, training stability, and decoder transfer.

---

## Acknowledgments

Developed under the mentorship of Dr. Xuyang Li, Assistant Professor, College of Engineering, UNC Charlotte. Supported by the Office of Undergraduate Research.
# Latent ODE Modeling of Sparse Oceanographic Contaminant Transport

## Overview

This project applies **Latent Neural Ordinary Differential Equations (Latent ODEs)** to model sparse, irregularly sampled oceanographic data. The objective is to learn continuous-time latent dynamics governing contaminant-related and biogeochemical variables (e.g., PFAS/PFOS proxies, dissolved oxygen, temperature–salinity structure) from ship-based observational datasets.

Oceanographic measurements are typically:

- Irregularly sampled in time and depth  
- Spatially sparse across large domains  
- Heterogeneous across casts and cruises  

Latent ODEs provide a principled framework for continuous-time modeling under these constraints.

---

## Motivation

Classical numerical ocean models (e.g., PDE-based solvers) require dense spatial grids and strong physical parameterization. In contrast, many contaminant datasets consist of sparse vertical profiles collected over limited time windows.

This project investigates whether:

- A learned latent dynamical system can capture underlying transport structure  
- Continuous-time neural dynamics improve generalization under irregular sampling  
- Latent space evolution provides a compact representation of contaminant behavior  

---

## Method

The model follows the Latent ODE framework:

1. **Encoder**  
   Maps irregularly sampled observations to an inferred latent initial state \( z_0 \)

2. **Neural ODE**  
   Evolves latent state via  
   \[
   \frac{dz}{dt} = f_\theta(z, t)
   \]  
   where \( f_\theta \) is a neural network

3. **Decoder**  
   Maps latent trajectory back to observed variables

Training is performed using a variational objective (ELBO) with reconstruction and KL terms.

Irregular sampling is handled directly through the ODE solver.

---

## Data

The dataset consists of ship-based oceanographic casts containing:

- Depth-resolved measurements  
- Temperature and salinity  
- Dissolved oxygen and/or contaminant-relevant variables  
- Sparse temporal and spatial coverage  

Preprocessing includes:

- Converting ragged cast structures into pointwise tabular format  
- Group-aware train/validation/test splits (by cast index)  
- Normalization using training statistics only  
- Depth alignment and optional binning  

---

## Experiments

Experiments evaluate:

- Reconstruction accuracy under irregular sampling  
- Latent trajectory smoothness and stability  
- Generalization to unseen casts  
- Sensitivity to latent dimension and ODE network depth  

Baselines may include:

- Discrete-time RNN/GRU models  
- Static feedforward regressors  
- Classical interpolation approaches  

---

## Repository Structure

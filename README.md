# Latent ODE Modeling of Argo Float Data

Undergraduate research project at UNC Charlotte (Dr. Xuyang Li, College of Engineering). Applies Latent Neural ODEs to BGC-Argo oceanographic profiling float data to learn continuous-time ocean dynamics in a compressed latent space.

---

## What it does

Encodes sparse, irregularly sampled vertical ocean profiles (Temperature, Salinity, Oxygen) into a latent representation, trains a Neural ODE on the latent dynamics, then probes whether that latent space transfers to predicting variables the model was never trained on — currently Chlorophyll.

---

## Preliminary results

- The latent space is nearly 1-dimensional (PC1 explains ~88% of variance) and spontaneously recovers seasonal structure from depth profiles alone — time is never an input
- A frozen linear probe on the latent space beats a direct MLP baseline on Chlorophyll prediction, confirming that T/S/O covariance with Chlorophyll is preserved through compression
- NODE outperforms GRU on short-horizon latent trajectory reconstruction; GRU wins at long horizons where NODE diverges due to compounding integration error

---

## Future directions

- Expand to combined PFL1/2/3 dataset to increase Chlorophyll float coverage
- Frozen probe comparison: T/S-only encoder vs T/S/O encoder on Chlorophyll
- Publication target: NeurIPS/ICLR climate-ML workshop or AAAI student track

---

## Stack

Python, PyTorch, torchdiffeq, xarray, pandas. Runs on UNC Charlotte SLURM cluster.
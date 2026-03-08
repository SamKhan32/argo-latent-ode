import numpy as np

## Data paths ##
LOW_DRIFT_PATH = "data/processed/PFL1_low_drift_devices.csv"
PFL1_PATH      = "data/processed/PFL1_preprocessed.csv"
INTERP_PATH    = "data/processed/PFL1_interp72.csv"    # output of interpolate_depth_grid.py

## Variables ##
INPUT_VARS       = ['Temperature', 'Salinity']   # encoder inputs (X)
TARGET_VARS      = ['Oxygen']                          # held-out reconstruction target (Y)
ALL_VARS         = ['Temperature', 'Salinity', 'Oxygen', 'Nitrate', 'pH', 'Chlorophyll']  # all sensor vars to interpolate
MIN_TARGET_PROBE = 8                                   # guaranteed floats with target coverage in probe set

## Data ##
DEPTH_STRIDE = 1   # only used with PFL1_PATH (raw data) — not needed with INTERP_PATH, kept at 1 because we use grids now.

## Interpolation grid (73 levels, 0–2000m) ##
DEPTH_GRID = np.concatenate([
    np.arange(0,    200,  10),    # 0, 10, 20, ... 190   (21 levels — mixed layer)
    np.arange(200,  1000, 25),    # 200, 225, ... 975     (32 levels — thermocline)
    np.arange(1000, 2001, 50),    # 1000, 1050, ... 2000  (21 levels — deep water)
])  # 73 levels total

## Split ##
TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
PROBE_FRAC = 0.10
SEED       = 42

## Model hyperparameters ##
LATENT_DIM     = 16         # dimension of latent profile vector p
ENCODER_HIDDEN = [64, 64]   # hidden layer sizes in encoder MLP
DECODER_HIDDEN = [64, 64]   # hidden layer sizes in decoder MLP
ODE_HIDDEN     = [64, 64]   # hidden layer sizes in ODE function f(p, lat, lon, t)

## Training ##
ENCODER_LR     = 1e-3
ENCODER_EPOCHS = 10
ODE_LR         = 1e-3
ODE_EPOCHS     = 100
BATCH_SIZE     = 32
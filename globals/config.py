import numpy as np

## Data paths ##
INTERP_PATH    = "data/processed/PFL1_interp72.csv"
LOW_DRIFT_PATH = "data/processed/PFL1_low_drift_devices.csv"

## Variables ##
INPUT_VARS       = ['Temperature', 'Salinity', 'Oxygen']
TARGET_VARS      = ['Chlorophyll']
ALL_VARS         = ['Temperature', 'Salinity', 'Oxygen', 'Nitrate', 'pH', 'Chlorophyll']
MIN_TARGET_PROBE = 4

## Data ##
DEPTH_STRIDE = 1
DEVICE      = "cuda"
RESULTS_DIR = "results/vanilla"

## Interpolation grid (73 levels, 0–2000m) ##
DEPTH_GRID = np.concatenate([
    np.arange(0,    200,  10),
    np.arange(200,  1000, 25),
    np.arange(1000, 2001, 50),
])

## Split ##
TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
PROBE_FRAC = 0.10
SEED       = 42

## Model hyperparameters ##
LATENT_DIM     = 32
ENCODER_HIDDEN = [128, 128]
DECODER_HIDDEN = [64, 64]
ODE_HIDDEN     = [128, 128, 128]
LAMBDA_ODE     = 0.5
LAMBDA_OXY     = 0.5

## Training ##
ENCODER_LR     = 1e-3
ENCODER_EPOCHS = 80
ODE_LR         = 2e-4      # was 5e-4, reduce to dampen spikes
ODE_EPOCHS     = 150       # a bit more room since LR is lower
BATCH_SIZE     = 32
PROBE_LR       = 1e-4
PROBE_EPOCHS   = 100
WINDOW_SIZE    = 25
STRIDE         = 2

CURRICULUM_WINDOWS  = [10]
CURRICULUM_WEIGHTS  = [1.0]
import numpy as np

## Data paths ##
#LOW_DRIFT_PATH = "data/processed/all_low_drift_oxygen_devices.csv"
#PFL1_PATH      = "data/processed/PFL1_preprocessed.csv"
INTERP_PATH    = "data/processed/PFL1_interp72.csv"  # split.py handles PFL2/3 internally

LOW_DRIFT_PATH = "data/processed/PFL1_low_drift_devices.csv"

## Variables ##
INPUT_VARS       = ['Temperature', 'Salinity']
TARGET_VARS      = ['Oxygen']
ALL_VARS         = ['Temperature', 'Salinity', 'Oxygen', 'Nitrate', 'pH', 'Chlorophyll']
MIN_TARGET_PROBE = 4

## Data ##
DEPTH_STRIDE = 1
DEVICE      = "cuda"
RESULTS_DIR = "results/vanilla"
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
LATENT_DIM     = 12                      # reduce capacity
ENCODER_HIDDEN = [64, 64]
DECODER_HIDDEN = [32, 32]
ODE_HIDDEN     = [128, 128]                # much smaller, smoother dynamics

LAMBDA_ODE     = 1.0                     # stronger regularization on dynamics
LAMBDA_OXY     = 0.5

ODE_METHOD = 'rk4'                       # keep fixed-step, but see note below

## Training ##
ENCODER_LR     = 5e-4                    # cut in half
ENCODER_EPOCHS = 100                     # longer, but more stable

ODE_LR         = 1e-4                    # major reduction (key change)
ODE_EPOCHS     = 150                     # compensate with longer training

BATCH_SIZE     = 64                      # reduce gradient noise

PROBE_LR       = 2e-5                    # more conservative
PROBE_EPOCHS   = 120

WINDOW_SIZE = 20                         # slightly shorter horizon
STRIDE      = 2

CURRICULUM_WINDOWS  = [20]
CURRICULUM_WEIGHTS  = [1.0]    # gradual increase in difficulty
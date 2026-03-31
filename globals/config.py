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
MIN_TARGET_PROBE = 8

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
LATENT_DIM     = 32
ENCODER_HIDDEN = [128, 128]
DECODER_HIDDEN = [64, 64]
ODE_HIDDEN     = [256, 256, 256]  # increased from [128, 128, 128]
LAMBDA_ODE     = 0.5
LAMBDA_OXY     = 0.5

## Training ##
ENCODER_LR     = 1e-3
ENCODER_EPOCHS = 80               # increased from 40 — oscillation fix + cosine annealing
ODE_LR         = 5e-4
ODE_EPOCHS     = 200              # increased from 100 — more room for curriculum phases
BATCH_SIZE     = 32
PROBE_LR       = 1e-4
PROBE_EPOCHS   = 150              # increased from 100 — probe was still converging at 100

WINDOW_SIZE = 25
STRIDE      = 2

CURRICULUM_WINDOWS  = [5, 10, 20, 25]
CURRICULUM_WEIGHTS  = [0.15, 0.20, 0.25, 0.40]  # epoch fractions per phase
# gives: 30, 40, 50, 80 epochs for windows 5, 10, 20, 25
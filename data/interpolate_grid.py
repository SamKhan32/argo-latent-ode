"""
data/interpolate_depth_grid.py

Interpolates each cast onto the fixed depth grid defined in config1.py,
replacing raw high-resolution profiles with a consistent 74-level representation.

Saves to INTERP_PATH — does NOT overwrite PFL1_PATH.

Run from project root:
    python -m data.interpolate_depth_grid
"""

from configs.config1 import (
    LOW_DRIFT_PATH,
    PFL1_PATH,
    INTERP_PATH,
    DEPTH_GRID,
    ALL_VARS,
)

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

CAST_COL = 'wod_unique_cast'
DEPTH_COL = 'z'
WMO_COL  = 'WMO_ID'
META_COLS = ['date', 'GMT_time', 'lat', 'lon', WMO_COL, 'time']


def interpolate_cast(cast_df):
    """
    Interpolate one cast onto DEPTH_GRID.

    For each variable, fits a linear interpolator over the cast's valid
    (non-NaN) depth/value pairs and evaluates it at each grid level.
    Grid levels outside the cast's depth range get NaN (no extrapolation).
    Returns a DataFrame with one row per grid level.
    """
    cast_df = cast_df.sort_values(DEPTH_COL).drop_duplicates(DEPTH_COL)

    result = {DEPTH_COL: DEPTH_GRID.copy().astype(float)}

    # copy cast-level metadata from first row (constant across depth levels)
    first = cast_df.iloc[0]
    result[CAST_COL] = first[CAST_COL]
    for col in META_COLS:
        if col in cast_df.columns:
            result[col] = first[col]

    # interpolate each sensor variable onto the fixed grid
    for var in ALL_VARS:
        if var not in cast_df.columns:
            result[var] = np.nan
            continue

        valid = cast_df[[DEPTH_COL, var]].dropna()

        if len(valid) < 2:
            result[var] = np.nan
            continue

        f = interp1d(
            valid[DEPTH_COL].values,
            valid[var].values,
            kind='linear',
            bounds_error=False,   # NaN outside data range — no extrapolation
            fill_value=np.nan,
        )
        result[var] = f(DEPTH_GRID)

    return pd.DataFrame(result)


def main():
    print("Loading data...")
    df        = pd.read_csv(PFL1_PATH)
    low_drift = pd.read_csv(LOW_DRIFT_PATH)

    valid_wmos = set(low_drift[WMO_COL])
    df = df[df[WMO_COL].isin(valid_wmos)].copy()

    n_casts = df[CAST_COL].nunique()
    print(f"Rows loaded:          {len(df):,}")
    print(f"Unique casts:         {n_casts:,}")
    print(f"Grid levels:          {len(DEPTH_GRID)}  ({DEPTH_GRID.min():.0f}m – {DEPTH_GRID.max():.0f}m)")
    print(f"Expected output rows: {n_casts * len(DEPTH_GRID):,}")

    casts = list(df.groupby(CAST_COL, sort=False))
    results = []

    print(f"\nInterpolating {len(casts)} casts...")
    for i, (cast_id, cast_df) in enumerate(casts):
        if i % 1000 == 0:
            print(f"  {i}/{len(casts)}  ({100*i/len(casts):.1f}%)")
        results.append(interpolate_cast(cast_df))

    print("Concatenating...")
    out = pd.concat(results, ignore_index=True)

    # consistent column ordering
    col_order = [CAST_COL, WMO_COL, DEPTH_COL] + \
                [c for c in META_COLS if c != WMO_COL and c in out.columns] + \
                [v for v in ALL_VARS if v in out.columns]
    out = out[[c for c in col_order if c in out.columns]]

    print(f"Saving to {INTERP_PATH}...")
    out.to_csv(INTERP_PATH, index=False)

    print(f"\nDone.")
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
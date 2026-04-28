"""
data/interpolate_depth_grid.py

Interpolates each cast onto the fixed depth grid defined in globals/config.py,
replacing raw high-resolution profiles with a consistent 73-level representation.

By default runs on PFL1 using paths from config. Pass --pfl to run on PFL2/PFL3.

Run from project root:
    python -m data.interpolate_depth_grid                        # PFL1 (default)
    python -m data.interpolate_depth_grid --pfl 2               # PFL2
    python -m data.interpolate_depth_grid --pfl 3               # PFL3
"""

import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from config import (
    DEPTH_GRID,
    ALL_VARS,
)

CAST_COL  = 'wod_unique_cast'
DEPTH_COL = 'z'
WMO_COL   = 'WMO_ID'
META_COLS = ['date', 'GMT_time', 'lat', 'lon', 'WMO_ID', 'time']


def interpolate_cast(cast_df):
    cast_df = cast_df.sort_values(DEPTH_COL).drop_duplicates(DEPTH_COL)

    result = {DEPTH_COL: DEPTH_GRID.copy().astype(float)}

    first = cast_df.iloc[0]
    result[CAST_COL] = first[CAST_COL]
    for col in META_COLS:
        if col in cast_df.columns:
            result[col] = first[col]

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
            bounds_error=False,
            fill_value=np.nan,
        )
        result[var] = f(DEPTH_GRID)

    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description="Interpolate PFL data onto fixed depth grid")
    parser.add_argument("--pfl", type=int, choices=[1, 2, 3], default=1,
                        help="Which PFL dataset to process (1, 2, or 3)")
    args = parser.parse_args()

    n = args.pfl
    preprocessed_path = f"data/processed/PFL{n}_preprocessed.csv"
    low_drift_path    = f"data/processed/PFL{n}_low_drift_devices.csv"
    output_path       = f"data/processed/PFL{n}_interp72.csv"
    df        = pd.read_csv(preprocessed_path, low_memory=False)


    # For PFL1 use the combined all_low_drift_oxygen_devices if available,
    # otherwise fall back to PFL1_low_drift_devices
    if n == 1:
        low_drift_path = "data/processed/PFL1_low_drift_devices.csv"

    print(f"Processing PFL{n}")
    print(f"  Input:     {preprocessed_path}")
    print(f"  Low-drift: {low_drift_path}")
    print(f"  Output:    {output_path}")

    df        = pd.read_csv(preprocessed_path, low_memory=False)
    low_drift = pd.read_csv(low_drift_path)
    # construct time from date + GMT_time if not already present
    if 'time' not in df.columns:
        df['time'] = pd.to_datetime(df['date'], format='%Y%m%d') + \
                    pd.to_timedelta(df['GMT_time'], unit='ns')
    valid_wmos = set(low_drift[WMO_COL])
    df = df[df[WMO_COL].isin(valid_wmos)].copy()

    n_casts = df[CAST_COL].nunique()
    print(f"\nRows loaded:          {len(df):,}")
    print(f"Unique casts:         {n_casts:,}")
    print(f"Grid levels:          {len(DEPTH_GRID)}  ({DEPTH_GRID.min():.0f}m – {DEPTH_GRID.max():.0f}m)")
    print(f"Expected output rows: {n_casts * len(DEPTH_GRID):,}")

    casts   = list(df.groupby(CAST_COL, sort=False))
    results = []

    print(f"\nInterpolating {len(casts)} casts...")
    for i, (cast_id, cast_df) in enumerate(casts):
        if i % 1000 == 0:
            print(f"  {i}/{len(casts)}  ({100*i/len(casts):.1f}%)")
        results.append(interpolate_cast(cast_df))

    print("Concatenating...")
    out = pd.concat(results, ignore_index=True)

    col_order = [CAST_COL, WMO_COL, DEPTH_COL] + \
                [c for c in META_COLS if c != WMO_COL and c in out.columns] + \
                [v for v in ALL_VARS if v in out.columns]
    out = out[[c for c in col_order if c in out.columns]]

    print(f"Saving to {output_path}...")
    out.to_csv(output_path, index=False)

    print(f"\nDone.")
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
    
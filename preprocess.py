"""
preprocess.py

Full preprocessing pipeline: NetCDF -> interpolated, concatenated CSV.

Steps:
    1. Convert raw .nc files to flat CSVs  (nc_to_csv)
    2. Compute per-float drift statistics  (compute_drift)
    3. Filter to low-drift floats          (filter_low_drift)
    4. Filter to floats with O2 coverage   (filter_oxygen)
    5. Interpolate to fixed depth grid     (interpolate)
    6. Concatenate PFL1/2/3 into one file  (concatenate)

Output:
    data/processed/PFL_all_interp72.csv   <- what INTERP_PATH in config.py points to
    data/processed/all_low_drift_oxygen_devices.csv

Run from project root:
    python preprocess.py
"""

import argparse
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from config import (
    DEPTH_GRID,
    ALL_VARS,
    MAX_AVG_DRIFT_KM,
    MIN_CASTS,
    MIN_OXYGEN_CYCLES,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

ORIGINAL_DIR  = "data/original/"
PROCESSED_DIR = "data/processed/"

PFL_SOURCES = {
    "PFL1": ORIGINAL_DIR + "PFL1.nc",
    "PFL2": ORIGINAL_DIR + "PFL2.nc",
    "PFL3": ORIGINAL_DIR + "PFL3.nc",
}

CAST_COL  = "wod_unique_cast"
DEPTH_COL = "z"
WMO_COL   = "WMO_ID"
META_COLS = ["date", "GMT_time", "lat", "lon", "WMO_ID", "time"]


# ── Step 1: nc -> csv ──────────────────────────────────────────────────────────

def nc_to_csv(input_path, output_path):
    print(f"\n[nc_to_csv] {input_path} -> {output_path}")
    ds = xr.open_dataset(input_path)

    exclude = {"Primary_Investigator", "Principal_Investigator"}
    variables = [
        var.replace("_row_size", "")
        for var in ds.data_vars
        if var.endswith("_row_size")
        and var.replace("_row_size", "") in ds.data_vars
        and var.replace("_row_size", "") not in exclude
    ]
    print(f"  Variables detected: {variables}")

    row_boundaries = {}
    for var in variables:
        sizes = np.nan_to_num(ds[f"{var}_row_size"].values, nan=0).astype(int)
        row_boundaries[var] = np.cumsum(sizes)

    if "z_row_size" in ds.data_vars:
        z_sizes = np.nan_to_num(ds["z_row_size"].values, nan=0).astype(int)
        z_boundaries = np.cumsum(z_sizes)
    else:
        z_boundaries = None

    n_casts = len(ds["casts"].values)
    cast_tables = []

    for cast_idx in range(n_casts):
        if z_boundaries is not None:
            start = 0 if cast_idx == 0 else z_boundaries[cast_idx - 1]
            end   = z_boundaries[cast_idx]
            n     = end - start
            z_data = ds["z"].isel(z_obs=slice(start, end)).values
        else:
            ref = variables[0]
            start = 0 if cast_idx == 0 else row_boundaries[ref][cast_idx - 1]
            end   = row_boundaries[ref][cast_idx]
            n     = end - start
            z_data = None

        if n == 0:
            continue

        cast_data = {"castIndex": np.full(n, cast_idx)}

        for field in ["wod_unique_cast", "date", "GMT_time", "lat", "lon", "WMO_ID"]:
            if field in ds:
                cast_data[field] = np.full(n, ds[field].isel(casts=cast_idx).item())

        if z_data is not None:
            cast_data[DEPTH_COL] = z_data

        for var in variables:
            if var not in row_boundaries:
                continue
            s = 0 if cast_idx == 0 else row_boundaries[var][cast_idx - 1]
            e = row_boundaries[var][cast_idx]
            obs_dim = f"{var}_obs"
            if obs_dim in ds[var].dims and (e - s) > 0:
                var_data = ds[var].isel({obs_dim: slice(s, e)}).values
                if len(var_data) < n:
                    padded = np.full(n, np.nan)
                    padded[:len(var_data)] = var_data
                    cast_data[var] = padded
                else:
                    cast_data[var] = var_data[:n]
            else:
                cast_data[var] = np.full(n, np.nan)

        cast_tables.append(pd.DataFrame(cast_data))

        if (cast_idx + 1) % 500 == 0:
            print(f"  {cast_idx + 1}/{n_casts} casts processed")

    df = pd.concat(cast_tables, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}  shape={df.shape}")
    return df


# ── Step 2: drift statistics ───────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def compute_drift(df):
    cast_locs = df.groupby(CAST_COL).first().reset_index()
    cast_locs = cast_locs.dropna(subset=["lat", "lon", WMO_COL])

    rows = []
    for wmo_id, group in cast_locs.groupby(WMO_COL):
        if len(group) < 2:
            continue
        sort_col = "time" if "time" in group.columns else CAST_COL
        group = group.sort_values(sort_col)
        lats, lons = group["lat"].values, group["lon"].values
        distances = [haversine(lats[i], lons[i], lats[i+1], lons[i+1])
                     for i in range(len(lats) - 1)]
        rows.append({
            WMO_COL:                      wmo_id,
            "n_casts":                    len(group),
            "total_distance_km":          np.sum(distances),
            "straight_line_distance_km":  haversine(lats[0], lons[0], lats[-1], lons[-1]),
            "avg_distance_per_cast_km":   np.mean(distances),
            "max_distance_per_cast_km":   np.max(distances),
            "min_distance_per_cast_km":   np.min(distances),
            "std_distance_per_cast_km":   np.std(distances),
            "start_lat":                  lats[0],
            "start_lon":                  lons[0],
            "end_lat":                    lats[-1],
            "end_lon":                    lons[-1],
        })
    return pd.DataFrame(rows)


# ── Step 3: low-drift filter ───────────────────────────────────────────────────

def filter_low_drift(drift_df):
    return drift_df[
        (drift_df["avg_distance_per_cast_km"] <= MAX_AVG_DRIFT_KM) &
        (drift_df["n_casts"] >= MIN_CASTS)
    ].copy()


# ── Step 4: O2 filter ──────────────────────────────────────────────────────────

def filter_oxygen(df, low_drift_wmo_ids):
    ld_df = df[df[WMO_COL].isin(low_drift_wmo_ids)]
    o2_counts = (
        ld_df.groupby([WMO_COL, CAST_COL])["Oxygen"]
        .apply(lambda x: x.notna().any())
        .groupby(WMO_COL)
        .sum()
    )
    return o2_counts[o2_counts >= MIN_OXYGEN_CYCLES].index.tolist()


# ── Step 5: interpolate to depth grid ─────────────────────────────────────────

def interpolate_cast(cast_df):
    cast_df = cast_df.sort_values(DEPTH_COL).drop_duplicates(DEPTH_COL)
    result  = {DEPTH_COL: DEPTH_GRID.copy().astype(float)}

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
            valid[DEPTH_COL].values, valid[var].values,
            kind="linear", bounds_error=False, fill_value=np.nan,
        )
        result[var] = f(DEPTH_GRID)

    return pd.DataFrame(result)


def interpolate(df):
    if "time" not in df.columns:
        df["time"] = (
            pd.to_datetime(df["date"], format="%Y%m%d") +
            pd.to_timedelta(df["GMT_time"], unit="ns")
        )
    casts   = list(df.groupby(CAST_COL, sort=False))
    results = []
    print(f"  Interpolating {len(casts)} casts...")
    for i, (_, cast_df) in enumerate(casts):
        results.append(interpolate_cast(cast_df))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(casts)}")
    return pd.concat(results, ignore_index=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_pfl(name, nc_path):
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    preprocessed_path = PROCESSED_DIR + f"{name}_preprocessed.csv"

    # Step 1: nc -> csv
    df = nc_to_csv(nc_path, preprocessed_path)

    # Step 2 & 3: drift + low-drift filter
    print(f"\n[drift] Computing drift statistics...")
    drift_df  = compute_drift(df)
    low_drift = filter_low_drift(drift_df)
    print(f"  Total floats:     {len(drift_df)}")
    print(f"  Low-drift floats: {len(low_drift)}")

    # Step 4: O2 filter
    o2_wmo_ids = filter_oxygen(df, low_drift[WMO_COL].tolist())
    print(f"  O2 floats (>={MIN_OXYGEN_CYCLES} cycles): {len(o2_wmo_ids)}")

    low_drift_o2 = low_drift[low_drift[WMO_COL].isin(o2_wmo_ids)].copy()
    low_drift_o2["source"] = name

    # Step 5: interpolate
    print(f"\n[interpolate] Filtering to {len(o2_wmo_ids)} O2 floats...")
    df_filtered = df[df[WMO_COL].isin(o2_wmo_ids)].copy()
    interp_df   = interpolate(df_filtered)
    interp_df["source"] = name

    return low_drift_o2, interp_df


def main():
    all_low_drift = []
    all_interp    = []

    for name, nc_path in PFL_SOURCES.items():
        low_drift_o2, interp_df = process_pfl(name, nc_path)
        all_low_drift.append(low_drift_o2)
        all_interp.append(interp_df)

    # Save combined low-drift O2 device list
    combined_ld = pd.concat(all_low_drift, ignore_index=True)
    combined_ld = combined_ld.drop_duplicates(subset=WMO_COL, keep="first")
    ld_out = PROCESSED_DIR + "all_low_drift_oxygen_devices.csv"
    combined_ld.to_csv(ld_out, index=False)
    print(f"\n[output] Low-drift O2 devices: {ld_out}  ({len(combined_ld)} floats)")
    for src in ["PFL1", "PFL2", "PFL3"]:
        n = (combined_ld["source"] == src).sum()
        print(f"  {src}: {n}")

    # Save concatenated interpolated data — this is INTERP_PATH
    combined_interp = pd.concat(all_interp, ignore_index=True)
    interp_out = PROCESSED_DIR + "PFL_all_interp72.csv"
    combined_interp.to_csv(interp_out, index=False)
    print(f"\n[output] Combined interpolated data: {interp_out}  shape={combined_interp.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()

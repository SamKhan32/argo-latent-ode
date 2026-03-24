"""
data_cleaning/expand_oxygen_floats.py

Applies the same low-drift filter used for PFL1 to PFL2 and PFL3,
then filters for floats with oxygen coverage and merges with PFL1
to produce an expanded low-drift oxygen float list.

Output:
    data/processed/PFL2_device_drift_statistics.csv
    data/processed/PFL3_device_drift_statistics.csv
    data/processed/PFL2_low_drift_devices.csv
    data/processed/PFL3_low_drift_devices.csv
    data/processed/all_low_drift_oxygen_devices.csv  <- main output

Usage:
    python data_cleaning/expand_oxygen_floats.py
"""

import pandas as pd
import numpy as np

PROCESSED_DIR     = "data/processed/"
MAX_AVG_DRIFT_KM  = 50
MIN_CASTS         = 5
MIN_OXYGEN_CYCLES = 5   # minimum cycles with oxygen obs to count as an O2 float


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def calculate_device_drift(df):
    cast_locations = df.groupby("wod_unique_cast").first().reset_index()
    cast_locations = cast_locations.dropna(subset=["lat", "lon", "WMO_ID"])

    drift_data = []
    for wmo_id, device_casts in cast_locations.groupby("WMO_ID"):
        if len(device_casts) < 2:
            continue

        device_casts = device_casts.sort_values("time") if "time" in device_casts.columns \
                       else device_casts.sort_values("wod_unique_cast")

        lats = device_casts["lat"].values
        lons = device_casts["lon"].values

        distances = [haversine(lats[i], lons[i], lats[i+1], lons[i+1])
                     for i in range(len(lats) - 1)]

        total_distance    = np.sum(distances)
        straight_line     = haversine(lats[0], lons[0], lats[-1], lons[-1])

        drift_data.append({
            "WMO_ID":                    wmo_id,
            "n_casts":                   len(device_casts),
            "total_distance_km":         total_distance,
            "straight_line_distance_km": straight_line,
            "avg_distance_per_cast_km":  np.mean(distances),
            "max_distance_per_cast_km":  np.max(distances),
            "min_distance_per_cast_km":  np.min(distances),
            "std_distance_per_cast_km":  np.std(distances),
            "start_lat":                 lats[0],
            "start_lon":                 lons[0],
            "end_lat":                   lats[-1],
            "end_lon":                   lons[-1],
        })

    return pd.DataFrame(drift_data)


def filter_low_drift(drift_df):
    return drift_df[
        (drift_df["avg_distance_per_cast_km"] <= MAX_AVG_DRIFT_KM) &
        (drift_df["n_casts"] >= MIN_CASTS)
    ]


def get_oxygen_floats(df, low_drift_wmo_ids):
    """Return WMO_IDs that are low-drift AND have enough oxygen observations."""
    ld_df = df[df["WMO_ID"].isin(low_drift_wmo_ids)]

    # count cycles with at least one non-NaN oxygen reading
    o2_counts = (
        ld_df.groupby(["WMO_ID", "wod_unique_cast"])["Oxygen"]
        .apply(lambda x: x.notna().any())
        .groupby("WMO_ID")
        .sum()
    )
    return o2_counts[o2_counts >= MIN_OXYGEN_CYCLES].index.tolist()


def process_pfl(name, path):
    print(f"\n{'='*50}")
    print(f"Processing {name}: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Floats: {df['WMO_ID'].nunique()}")

    drift_df = calculate_device_drift(df)
    drift_df.to_csv(PROCESSED_DIR + f"{name}_device_drift_statistics.csv", index=False)
    print(f"  Drift stats saved.")

    low_drift = filter_low_drift(drift_df)
    low_drift.to_csv(PROCESSED_DIR + f"{name}_low_drift_devices.csv", index=False)
    print(f"  Low-drift floats: {len(low_drift)}")

    o2_wmo_ids = get_oxygen_floats(df, low_drift["WMO_ID"].tolist())
    print(f"  Low-drift O2 floats (>={MIN_OXYGEN_CYCLES} O2 cycles): {len(o2_wmo_ids)}")

    result = low_drift[low_drift["WMO_ID"].isin(o2_wmo_ids)].copy()
    result["source"] = name
    return result


def main():
    # PFL1 low drift devices already computed — just load and tag
    print("Loading PFL1 low-drift devices...")
    pfl1_ld  = pd.read_csv(PROCESSED_DIR + "PFL1_low_drift_devices.csv")
    pfl1_df  = pd.read_csv(PROCESSED_DIR + "PFL1_preprocessed.csv", low_memory=False)
    pfl1_o2  = get_oxygen_floats(pfl1_df, pfl1_ld["WMO_ID"].tolist())
    pfl1_out = pfl1_ld[pfl1_ld["WMO_ID"].isin(pfl1_o2)].copy()
    pfl1_out["source"] = "PFL1"
    print(f"  PFL1 low-drift O2 floats: {len(pfl1_out)}")

    # PFL2 and PFL3
    pfl2_out = process_pfl("PFL2", PROCESSED_DIR + "PFL2_preprocessed.csv")
    pfl3_out = process_pfl("PFL3", PROCESSED_DIR + "PFL3_preprocessed.csv")

    # Combine all
    combined = pd.concat([pfl1_out, pfl2_out, pfl3_out], ignore_index=True)

    # Drop duplicate WMO_IDs (float appears in multiple PFL files)
    before = len(combined)
    combined = combined.drop_duplicates(subset="WMO_ID", keep="first")
    print(f"\nDuplicates removed: {before - len(combined)}")

    combined.to_csv(PROCESSED_DIR + "all_low_drift_oxygen_devices.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Total unique low-drift O2 floats: {len(combined)}")
    print(f"  PFL1: {(combined['source'] == 'PFL1').sum()}")
    print(f"  PFL2: {(combined['source'] == 'PFL2').sum()}")
    print(f"  PFL3: {(combined['source'] == 'PFL3').sum()}")
    print(f"\nSaved to: {PROCESSED_DIR}all_low_drift_oxygen_devices.csv")


if __name__ == "__main__":
    main()
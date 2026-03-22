"""
diagnose_redundancy.py

Checks for redundant/duplicate casts in the PFL1 dataset.
Run from the root of the ocean_dynamics_2 project directory.

    python diagnose_redundancy.py
"""

from globals.config import LOW_DRIFT_PATH, PFL1_PATH, INPUT_VARS, TARGET_VARS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# all sensor columns we care about for similarity checking
ALL_VARS = INPUT_VARS + TARGET_VARS   # ['Temperature', 'Salinity', 'z', 'Oxygen']

TIME_COL = "time"
WMO_COL  = "WMO_ID"
CAST_COL = "wod_unique_cast"


# ── LOAD ───────────────────────────────────────────────────────────────────────

def load_data():
    df        = pd.read_csv(PFL1_PATH)
    low_drift = pd.read_csv(LOW_DRIFT_PATH)

    valid_wmos = set(low_drift["WMO_ID"])
    df = df[df[WMO_COL].isin(valid_wmos)].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    print(f"Rows loaded:    {len(df)}")
    print(f"Unique floats:  {df[WMO_COL].nunique()}")
    print(f"Unique casts:   {df[CAST_COL].nunique()}")
    return df


def cast_summary(df):
    """One row per cast — mean sensor values as a fingerprint, first time value."""
    agg = {TIME_COL: "first", WMO_COL: "first"}
    for var in ALL_VARS:
        if var in df.columns:
            agg[var] = "mean"
    return df.groupby(CAST_COL).agg(agg).reset_index()


# ── CHECKS ─────────────────────────────────────────────────────────────────────

def check_time_gaps(summary):
    """
    For each float, compute hours between consecutive casts.
    Argo floats normally profile every ~10 days, so anything
    under 24 hours is worth investigating.
    """
    print("\n=== TIME GAP ANALYSIS ===")
    print("(gaps between consecutive casts from the same float)\n")

    all_gaps = []
    for _, group in summary.groupby(WMO_COL):
        gaps = group.sort_values(TIME_COL)[TIME_COL].diff().dt.total_seconds() / 3600
        all_gaps.extend(gaps.dropna().tolist())

    all_gaps = np.array(all_gaps)

    print(f"Total consecutive cast pairs:  {len(all_gaps)}")
    print(f"Median gap:                    {np.median(all_gaps):.1f} hours")
    print(f"Mean gap:                      {np.mean(all_gaps):.1f} hours")
    print(f"\nGaps under 1 hour:             {(all_gaps < 1).sum()}   ({100*(all_gaps < 1).mean():.2f}%)")
    print(f"Gaps under 6 hours:            {(all_gaps < 6).sum()}   ({100*(all_gaps < 6).mean():.2f}%)")
    print(f"Gaps under 24 hours:           {(all_gaps < 24).sum()}   ({100*(all_gaps < 24).mean():.2f}%)")
    print(f"Gaps over 30 days (720 hours): {(all_gaps > 720).sum()}   ({100*(all_gaps > 720).mean():.2f}%)")

    # histogram clipped to 2 weeks so the normal ~10-day peak is visible
    plt.figure(figsize=(9, 4))
    plt.hist(all_gaps[all_gaps <= 24*14], bins=100, edgecolor="none", color="steelblue")
    plt.xlabel("Hours between consecutive casts (same float)")
    plt.ylabel("Count")
    plt.title("Inter-cast time gaps (clipped to 2 weeks)")
    plt.tight_layout()
    plt.savefig("gap_distribution.png", dpi=150)
    plt.close()
    print("\nSaved: gap_distribution.png")

    return all_gaps


def check_value_similarity(summary, time_threshold_hours=6):
    """
    Among cast pairs that are close in time, check how different
    the mean sensor readings actually are.
    Tiny gap + tiny delta = likely duplicate or near-duplicate.
    """
    print(f"\n=== VALUE SIMILARITY FOR CLOSE PAIRS (< {time_threshold_hours}h apart) ===\n")

    flagged = []
    for _, group in summary.groupby(WMO_COL):
        group = group.sort_values(TIME_COL).reset_index(drop=True)
        for i in range(len(group) - 1):
            gap_h = (group.loc[i+1, TIME_COL] - group.loc[i, TIME_COL]).total_seconds() / 3600
            if gap_h < time_threshold_hours:
                row = {
                    WMO_COL:    group.loc[i, WMO_COL],
                    "cast_a":   group.loc[i,   CAST_COL],
                    "cast_b":   group.loc[i+1, CAST_COL],
                    "gap_hours": round(gap_h, 4),
                }
                for var in ALL_VARS:
                    if var in group.columns:
                        v1, v2 = group.loc[i, var], group.loc[i+1, var]
                        if not (np.isnan(v1) or np.isnan(v2)):
                            row[f"delta_{var}"] = round(abs(v1 - v2), 6)
                flagged.append(row)

    if not flagged:
        print("No cast pairs found within the time threshold — likely no duplicates.")
        return

    pairs_df = pd.DataFrame(flagged)
    print(f"Close pairs found: {len(pairs_df)}\n")
    print("Mean absolute sensor differences within close pairs:")
    for var in ALL_VARS:
        col = f"delta_{var}"
        if col in pairs_df.columns:
            print(f"  {var:<15}  mean Δ = {pairs_df[col].mean():.5f}   max Δ = {pairs_df[col].max():.5f}")

    # rank by most suspicious: shortest gap + smallest temperature difference
    if "delta_Temperature" in pairs_df.columns:
        pairs_df["suspicion_score"] = pairs_df["gap_hours"] + pairs_df["delta_Temperature"]
        top = pairs_df.nsmallest(10, "suspicion_score")
        show = [WMO_COL, "cast_a", "cast_b", "gap_hours"] + \
               [f"delta_{v}" for v in ALL_VARS if f"delta_{v}" in pairs_df.columns]
        print(f"\nTop 10 most suspicious pairs (tiny gap + tiny temp diff):")
        print(top[show].to_string(index=False))

    pairs_df.to_csv("close_cast_pairs.csv", index=False)
    print("\nFull list saved: close_cast_pairs.csv")


def check_depth_level_counts(df):
    """
    How many raw depth rows does each cast have (before DEPTH_STRIDE subsampling)?
    Outlier casts with thousands of levels can slow down batching significantly.
    """
    print(f"\n=== RAW DEPTH LEVELS PER CAST (before DEPTH_STRIDE subsampling) ===\n")
    counts = df.groupby(CAST_COL).size()

    print(f"Mean:    {counts.mean():.1f}")
    print(f"Median:  {counts.median():.1f}")
    print(f"Max:     {counts.max()}")
    print(f"Min:     {counts.min()}")
    print(f"\nCasts with > 200 depth levels:  {(counts > 200).sum()}")
    print(f"Casts with > 500 depth levels:  {(counts > 500).sum()}")
    print(f"Casts with > 1000:              {(counts > 1000).sum()}")

    plt.figure(figsize=(9, 4))
    counts.clip(upper=500).hist(bins=80, edgecolor="none", color="coral")
    plt.xlabel("Raw depth levels per cast (clipped at 500)")
    plt.ylabel("Number of casts")
    plt.title("Distribution of raw depth levels per cast")
    plt.tight_layout()
    plt.savefig("depth_level_distribution.png", dpi=150)
    plt.close()
    print("Saved: depth_level_distribution.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df      = load_data()
    summary = cast_summary(df)

    check_time_gaps(summary)
    check_value_similarity(summary, time_threshold_hours=6)
    check_depth_level_counts(df)
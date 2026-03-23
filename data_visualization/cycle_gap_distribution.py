"""
data_visualization/cycle_gap_distribution.py
Prints distribution of time gaps between consecutive cycles per float.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/PFL1_interp72.csv")
df["time"] = pd.to_datetime(df["time"])
# get one row per cycle
cycles = df.groupby(["WMO_ID", "wod_unique_cast"])["time"].first().reset_index()
cycles = cycles.sort_values(["WMO_ID", "time"])

# compute gaps within each float
cycles["gap_days"] = cycles.groupby("WMO_ID")["time"].diff().dt.total_seconds() / 86400
gaps = cycles["gap_days"].dropna()

print(f"Min gap:    {gaps.min():.1f} days")
print(f"Max gap:    {gaps.max():.1f} days")
print(f"Median gap: {gaps.median():.1f} days")
print(f"Mean gap:   {gaps.mean():.1f} days")
print(f"\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {gaps.quantile(p/100):.1f} days")

print(f"\nGaps > 20 days: {(gaps > 20).sum()} ({100*(gaps > 20).mean():.1f}%)")
print(f"Gaps > 30 days: {(gaps > 30).sum()} ({100*(gaps > 30).mean():.1f}%)")
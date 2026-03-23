import pandas as pd

df = pd.read_csv("data/processed/PFL1_interp72.csv")

counts = df.groupby("WMO_ID")["wod_unique_cast"].nunique().sort_values()
print(f"Min cycles:    {counts.min()}")
print(f"Max cycles:    {counts.max()}")
print(f"Median cycles: {counts.median()}")
print(f"Mean cycles:   {counts.mean():.1f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {counts.quantile(p/100):.0f}")

print(f"\nFloats with >= 10 cycles: {(counts >= 10).sum()}")
print(f"Floats with >= 20 cycles: {(counts >= 20).sum()}")
print(f"Floats with >= 40 cycles: {(counts >= 40).sum()}")

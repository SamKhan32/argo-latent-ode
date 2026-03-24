import pandas as pd
df = pd.read_csv("data/processed/PFL1_interp72.csv")
coverage = df.groupby("WMO_ID")["Oxygen"].apply(lambda x: x.notna().mean())
print(f"Floats with >0% oxygen:   {(coverage > 0).sum()}")
print(f"Floats with >10% oxygen:  {(coverage > 0.1).sum()}")
print(f"Floats with >50% oxygen:  {(coverage > 0.5).sum()}")
print(f"Floats with >90% oxygen:  {(coverage > 0.9).sum()}")

import pandas as pd
from globals.config import LOW_DRIFT_PATH, INTERP_PATH, TARGET_VARS

low_drift_df = pd.read_csv(LOW_DRIFT_PATH)
pfl_df = pd.read_csv(INTERP_PATH)

low_drift_wmos = set(low_drift_df["WMO_ID"].unique())
pfl_filtered = pfl_df[pfl_df["WMO_ID"].isin(low_drift_wmos)]

target_cols = [v for v in TARGET_VARS if v in pfl_df.columns]
print(f"TARGET_VARS found in df: {target_cols}")

o2_wmo_ids = set(pfl_filtered[pfl_filtered[target_cols].notna().any(axis=1)]["WMO_ID"].unique())
print(f"O2 floats in low-drift: {len(o2_wmo_ids)} / {len(low_drift_wmos)}")

# check coverage per float
coverage = pfl_filtered.groupby("WMO_ID")["Oxygen"].apply(lambda x: x.notna().mean())
print(f"\nFloats with >0% oxygen: {(coverage > 0).sum()}")
print(f"Floats with >5% oxygen: {(coverage > 0.05).sum()}")
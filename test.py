import xarray as xr
import numpy as np
import pandas as pd

PFL1_NC  = "data/original/PFL1.nc"
PFL1_CSV = "data/processed/PFL1_preprocessed.csv"

print("Loading .nc file...")
ds = xr.open_dataset(PFL1_NC, decode_timedelta=False)

# extract wod_unique_cast and time — one row per cast
wod_ids   = ds["wod_unique_cast"].values
times     = ds["time"].values  # datetime64[ns]

cast_time_df = pd.DataFrame({
    "wod_unique_cast": wod_ids,
    "time":            times,
})

print(f"Extracted {len(cast_time_df)} cast timestamps")
print(cast_time_df.head())

# merge into existing CSV on wod_unique_cast
print("\nLoading CSV...")
df = pd.read_csv(PFL1_CSV)
print(f"CSV shape before merge: {df.shape}")

df = df.merge(cast_time_df, on="wod_unique_cast", how="left")
print(f"CSV shape after merge:  {df.shape}")
print(f"Missing time values:    {df['time'].isna().sum()}")

df.to_csv(PFL1_CSV, index=False)
print("Saved.")
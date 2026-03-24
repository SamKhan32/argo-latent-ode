import pandas as pd
df = pd.read_csv("data/processed/PFL1_preprocessed.csv", low_memory=False, nrows=200)
cast0 = df[df['castIndex'] == 0]
print(cast0[['z', 'Temperature', 'Oxygen']].head())
import xarray as xr
import numpy as np

ds = xr.open_dataset("data/original/PFL1.nc")
ox_sizes = np.nan_to_num(ds["Oxygen_row_size"].values, nan=0).astype(int)
print(f"Oxygen_row_size[:5]: {ox_sizes[:5]}")
print(f"Oxygen_row_size sum: {ox_sizes.sum():,}")
print(f"First cast oxygen slice: {ds['Oxygen'].isel(Oxygen_obs=slice(0, ox_sizes[0])).values[:5]}")
import pandas as pd
df = pd.read_csv("data/processed/PFL1_interp72.csv", nrows=50000)
print(df['Oxygen'].mean(), df['Oxygen'].median(), df['Oxygen'].isna().mean())

import pandas as pd
import numpy as np

def assign_ocean_region(lat, lon):
    if lat > 50:
        return "subpolar"
    elif lat > 35:
        return "northwest_atlantic" if lon < -40 else "northeast_atlantic"
    elif lat > 25:
        return "subtropical_west" if lon < -40 else "subtropical_east"
    else:
        return "tropics"

ld = pd.read_csv("data/processed/PFL1_low_drift_devices.csv")
df = pd.read_csv("data/processed/PFL1_interp72.csv")

ld["region"] = ld.apply(lambda r: assign_ocean_region(r["start_lat"], r["start_lon"]), axis=1)

o2_wmos = df.groupby("WMO_ID")["Oxygen"].apply(lambda x: x.notna().any())
o2_wmos = o2_wmos[o2_wmos].index

o2_floats = ld[ld["WMO_ID"].isin(o2_wmos)]
print(o2_floats["region"].value_counts())
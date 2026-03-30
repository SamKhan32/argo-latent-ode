# script to check how many casts all the variables possible
# header: castIndex,wod_unique_cast,date,GMT_time,lat,lon,WMO_ID,z,Temperature,Salinity,Oxygen,Pressure,Chlorophyll,Nitrate,pH
# ran from project root
# all PFL processed datasets are named like this PFLX_preprocessed.csv, where X is a number.
import numpy as np
import pandas as pd
DATA_PATH = "data/processed/"
PFL1 = DATA_PATH + "PFL1_preprocessed.csv"
PFL2 = DATA_PATH + "PFL2_preprocessed.csv"
PFL3 = DATA_PATH + "PFL3_preprocessed.csv"
VARS_TO_CHECK = ['Temperature', 'Salinity', 'Oxygen', 'Pressure', 'Chlorophyll']
PK_CAST = 'castIndex'  # unique identifier for each cast, used to group by casts
pfl1_df = pd.read_csv(PFL1)
pfl2_df = pd.read_csv(PFL2)
pfl3_df = pd.read_csv(PFL3)

# check how many casts have all variables present (non-NaN) in their profiles, and how many casts have every variable present at in their observations
def check_casts_with_all_vars(df, vars_to_check):
    # group by cast and check if all variables are present in each cast
    casts_with_all_vars = df.groupby(PK_CAST).apply(lambda x: x[vars_to_check].notna().all().all())
    num_casts_with_all_vars = casts_with_all_vars.sum()
    total_casts = casts_with_all_vars.shape[0]
    return num_casts_with_all_vars, total_casts
pfl1_casts_with_all_vars, pfl1_total_casts = check_casts_with_all_vars(pfl1_df, VARS_TO_CHECK)
pfl2_casts_with_all_vars, pfl2_total_casts = check_casts_with_all_vars(pfl2_df, VARS_TO_CHECK)
pfl3_casts_with_all_vars, pfl3_total_casts = check_casts_with_all_vars(pfl3_df, VARS_TO_CHECK)
print(f"PFL1: {pfl1_casts_with_all_vars}/{pfl1_total_casts} casts have all variables present")
print(f"PFL2: {pfl2_casts_with_all_vars}/{pfl2_total_casts} casts have all variables present")
print(f"PFL3: {pfl3_casts_with_all_vars}/{pfl3_total_casts} casts have all variables present")

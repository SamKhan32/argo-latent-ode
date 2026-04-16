import pandas as pd
df = pd.read_csv('data/processed/PFL1_interp72.csv')
for col in ['Temperature','Salinity','Oxygen','Chlorophyll','Nitrate','pH']:
    has_var = df.groupby('WMO_ID')[col].apply(lambda x: x.notna().any())
    print(f"{col}: {has_var.sum()} / {has_var.shape[0]} floats have at least one reading")
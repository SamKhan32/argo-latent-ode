import pandas as pd
df = pd.read_csv("data/processed/PFL1_preprocessed.csv", low_memory=False, nrows=200)
cast0 = df[df['castIndex'] == 0]
print(cast0[['z', 'Temperature', 'Oxygen']].head())
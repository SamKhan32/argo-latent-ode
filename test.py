import pandas as pd
df = pd.read_csv('data/processed/PFL1_preprocessed.csv')

# Get per-float stats
stats = df.sort_values('date').groupby('WMO_ID').agg(
    start_lat=('lat', 'first'),
    start_lon=('lon', 'first'),
    end_lat=('lat', 'last'),
    end_lon=('lon', 'last'),
    n_casts=('lat', 'count')
).reset_index()

# Compute straight line distance using haversine
import numpy as np
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

stats['straight_line_km'] = haversine(stats.start_lat, stats.start_lon, stats.end_lat, stats.end_lon)
print(stats.nlargest(10, 'straight_line_km')[['WMO_ID','straight_line_km','n_casts']])
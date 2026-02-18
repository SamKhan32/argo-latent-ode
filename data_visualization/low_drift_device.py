import pandas as pd
import numpy as np

processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations"

drift_df = pd.read_csv(processed_dir + "device_drift_statistics.csv")
pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")

print("=== UNDERSTANDING YOUR DATA DISTRIBUTION ===\n")

# Current selection
low_drift_devices = drift_df[
    (drift_df['avg_distance_per_cast_km'] <= 50) &
    (drift_df['n_casts'] >= 5)
]

print(f"Total devices in dataset: {drift_df['WMO_ID'].nunique()}")
print(f"Low-drift devices (≤50km, ≥5 casts): {len(low_drift_devices)}")
print(f"Percentage of devices: {len(low_drift_devices)/len(drift_df)*100:.1f}%\n")

# Check observations per device
total_obs = len(pfl_table)
low_drift_obs = pfl_table[pfl_table['WMO_ID'].isin(low_drift_devices['WMO_ID'])]

print(f"Total observations: {total_obs:,}")
print(f"Low-drift observations: {len(low_drift_obs):,}")
print(f"Percentage of observations: {len(low_drift_obs)/total_obs*100:.1f}%\n")

# This means those 50 devices are very "productive"
avg_obs_per_device_all = total_obs / drift_df['WMO_ID'].nunique()
avg_obs_per_device_low_drift = len(low_drift_obs) / len(low_drift_devices)

print(f"Average observations per device (all): {avg_obs_per_device_all:.0f}")
print(f"Average observations per device (low-drift): {avg_obs_per_device_low_drift:.0f}")
print(f"Low-drift devices are {avg_obs_per_device_low_drift/avg_obs_per_device_all:.1f}x more productive!\n")

print("="*60 + "\n")

# Let's see the top contributors
print("=== TOP 20 MOST PRODUCTIVE LOW-DRIFT DEVICES ===\n")
low_drift_device_obs = []
for wmo_id in low_drift_devices['WMO_ID']:
    n_obs = len(pfl_table[pfl_table['WMO_ID'] == wmo_id])
    n_casts = drift_df[drift_df['WMO_ID'] == wmo_id]['n_casts'].values[0]
    avg_drift = drift_df[drift_df['WMO_ID'] == wmo_id]['avg_distance_per_cast_km'].values[0]
    low_drift_device_obs.append({
        'WMO_ID': wmo_id,
        'observations': n_obs,
        'casts': n_casts,
        'avg_drift_km': avg_drift,
        'obs_per_cast': n_obs / n_casts
    })

top_devices = pd.DataFrame(low_drift_device_obs).sort_values('observations', ascending=False).head(20)
print(top_devices.to_string(index=False))

print("\n" + "="*60 + "\n")

# Cumulative contribution
print("=== CUMULATIVE CONTRIBUTION ===\n")
sorted_devices = pd.DataFrame(low_drift_device_obs).sort_values('observations', ascending=False)
sorted_devices['cumulative_obs'] = sorted_devices['observations'].cumsum()
sorted_devices['cumulative_pct'] = sorted_devices['cumulative_obs'] / len(low_drift_obs) * 100

print("How many devices do you need to get X% of low-drift observations?\n")
for pct in [50, 75, 90, 95, 99, 100]:
    n_devices = len(sorted_devices[sorted_devices['cumulative_pct'] <= pct]) + 1
    n_devices = min(n_devices, len(sorted_devices))
    actual_pct = sorted_devices.iloc[n_devices-1]['cumulative_pct']
    print(f"  {pct}% of data: {n_devices} devices (actual: {actual_pct:.1f}%)")

print("\n" + "="*60 + "\n")

# Is 50 devices enough for ML?
print("=== IS 50 DEVICES ENOUGH FOR ML? ===\n")

print("It depends on your use case:")
print("✓ For spatial modeling: 50 devices might be limiting if they don't cover your")
print("  geographic region well. You'd want spatial diversity.")
print("✓ For temporal modeling: If these 50 devices have many profiles over time,")
print("  you might have enough temporal variation.")
print("✓ For general oceanographic relationships (T-S-P-O2): 75% of observations")
print("  is probably plenty, even from 50 devices.")
print("\nKey question: Are you modeling spatial patterns or physical relationships?")

print("\n" + "="*60 + "\n")

# Analyze spatial coverage
print("=== SPATIAL COVERAGE OF LOW-DRIFT DEVICES ===\n")
low_drift_casts = pfl_table[pfl_table['WMO_ID'].isin(low_drift_devices['WMO_ID'])].groupby('castIndex').first()

print(f"Latitude range: {low_drift_casts['lat'].min():.2f}° to {low_drift_casts['lat'].max():.2f}°")
print(f"Longitude range: {low_drift_casts['lon'].min():.2f}° to {low_drift_casts['lon'].max():.2f}°")
print(f"\nNumber of unique cast locations: {len(low_drift_casts)}")
print(f"Average distance between consecutive casts: {low_drift_devices['avg_distance_per_cast_km'].mean():.1f} km")

# Compare to full dataset
all_casts = pfl_table.groupby('castIndex').first()
print(f"\nComparison to full dataset:")
print(f"  Full dataset lat range: {all_casts['lat'].min():.2f}° to {all_casts['lat'].max():.2f}°")
print(f"  Full dataset lon range: {all_casts['lon'].min():.2f}° to {all_casts['lon'].max():.2f}°")
print(f"  Low-drift covers {(low_drift_casts['lat'].max()-low_drift_casts['lat'].min())/(all_casts['lat'].max()-all_casts['lat'].min())*100:.1f}% of lat range")
print(f"  Low-drift covers {(low_drift_casts['lon'].max()-low_drift_casts['lon'].min())/(all_casts['lon'].max()-all_casts['lon'].min())*100:.1f}% of lon range")

print("\n" + "="*60 + "\n")

# Recommendation
print("=== RECOMMENDATION ===\n")
print("You have 75% of observations from 50 devices. This is actually GOOD if:")
print("1. Your ML task is learning T-S-P-O2 relationships (physical oceanography)")
print("2. You care more about data quality (low drift) than device diversity")
print("3. The 50 devices still cover your region of interest spatially")
print("\nConsider relaxing to 100-150 km if:")
print("1. You need better spatial coverage")
print("2. You're doing spatial interpolation/mapping")
print("3. You want to capture regional differences")
print("\nCurrent setup (50 km, 5 casts) is probably FINE for most ML applications!")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations/"
drift_df = pd.read_csv(processed_dir + "device_drift_statistics.csv")

# Let's examine the distribution more carefully
print("=== DRIFT DISTRIBUTION ANALYSIS ===\n")

# Percentiles for avg drift
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("Average Drift per Cast (km) - Percentiles:")
for p in percentiles:
    val = np.percentile(drift_df['avg_distance_per_cast_km'], p)
    print(f"  {p}th percentile: {val:.2f} km")

print("\n" + "="*50 + "\n")

# Try different thresholds
thresholds = [25, 50, 75, 100, 150, 200, 300]
min_casts_options = [3, 5, 10]

print("Number of devices at different thresholds:\n")
print(f"{'Max Avg Drift':<15} {'Min Casts=3':<15} {'Min Casts=5':<15} {'Min Casts=10':<15}")
print("-" * 60)

for threshold in thresholds:
    counts = []
    for min_casts in min_casts_options:
        n = len(drift_df[
            (drift_df['avg_distance_per_cast_km'] <= threshold) &
            (drift_df['n_casts'] >= min_casts)
        ])
        counts.append(n)
    print(f"{threshold} km{'':<9} {counts[0]:<15} {counts[1]:<15} {counts[2]:<15}")

print("\n" + "="*50 + "\n")

# Check how many observations you'd get
pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")
total_obs = len(pfl_table)

print("Number of OBSERVATIONS at different thresholds:\n")
print(f"{'Max Avg Drift':<15} {'Min Casts=3':<20} {'Min Casts=5':<20} {'Min Casts=10':<20}")
print("-" * 75)

for threshold in thresholds:
    obs_counts = []
    for min_casts in min_casts_options:
        selected_devices = drift_df[
            (drift_df['avg_distance_per_cast_km'] <= threshold) &
            (drift_df['n_casts'] >= min_casts)
        ]['WMO_ID'].unique()
        
        n_obs = len(pfl_table[pfl_table['WMO_ID'].isin(selected_devices)])
        pct = n_obs / total_obs * 100
        obs_counts.append(f"{n_obs:,} ({pct:.1f}%)")
    
    print(f"{threshold} km{'':<9} {obs_counts[0]:<20} {obs_counts[1]:<20} {obs_counts[2]:<20}")

print("\n" + "="*50 + "\n")

# Visualize the tradeoff
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution of avg drift (zoomed)
ax = axes[0, 0]
ax.hist(drift_df['avg_distance_per_cast_km'], bins=100, edgecolor='black', alpha=0.7)
ax.set_xlabel('Average Drift per Cast (km)', fontsize=11)
ax.set_ylabel('Number of Devices', fontsize=11)
ax.set_title('Distribution of Average Drift (Full Range)', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', linewidth=2, label='Current: 50 km')
ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='Relaxed: 100 km')
ax.axvline(200, color='yellow', linestyle='--', linewidth=2, label='Very relaxed: 200 km')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# 2. Distribution of avg drift (zoomed to low values)
ax = axes[0, 1]
low_drift_vals = drift_df[drift_df['avg_distance_per_cast_km'] <= 300]['avg_distance_per_cast_km']
ax.hist(low_drift_vals, bins=50, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Average Drift per Cast (km)', fontsize=11)
ax.set_ylabel('Number of Devices', fontsize=11)
ax.set_title('Distribution of Average Drift (Zoomed: 0-300 km)', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', linewidth=2, label='Current: 50 km')
ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='Relaxed: 100 km')
ax.axvline(200, color='yellow', linestyle='--', linewidth=2, label='Very relaxed: 200 km')
ax.legend()
ax.grid(alpha=0.3)

# 3. Tradeoff: threshold vs number of devices
ax = axes[1, 0]
thresholds_fine = range(10, 301, 10)
device_counts_5casts = [len(drift_df[(drift_df['avg_distance_per_cast_km'] <= t) & 
                                     (drift_df['n_casts'] >= 5)]) 
                        for t in thresholds_fine]
device_counts_3casts = [len(drift_df[(drift_df['avg_distance_per_cast_km'] <= t) & 
                                     (drift_df['n_casts'] >= 3)]) 
                        for t in thresholds_fine]

ax.plot(thresholds_fine, device_counts_5casts, linewidth=2, label='Min 5 casts', marker='o', markersize=3)
ax.plot(thresholds_fine, device_counts_3casts, linewidth=2, label='Min 3 casts', marker='s', markersize=3)
ax.set_xlabel('Max Average Drift Threshold (km)', fontsize=11)
ax.set_ylabel('Number of Devices', fontsize=11)
ax.set_title('Devices vs Drift Threshold', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Current: 50 km')
ax.axvline(100, color='orange', linestyle='--', alpha=0.5, label='Suggested: 100 km')
ax.legend()
ax.grid(alpha=0.3)

# 4. Tradeoff: threshold vs percentage of observations
ax = axes[1, 1]
obs_percentages_5casts = []
obs_percentages_3casts = []

for t in thresholds_fine:
    # 5 casts
    selected = drift_df[(drift_df['avg_distance_per_cast_km'] <= t) & 
                       (drift_df['n_casts'] >= 5)]['WMO_ID'].unique()
    n_obs = len(pfl_table[pfl_table['WMO_ID'].isin(selected)])
    obs_percentages_5casts.append(n_obs / total_obs * 100)
    
    # 3 casts
    selected = drift_df[(drift_df['avg_distance_per_cast_km'] <= t) & 
                       (drift_df['n_casts'] >= 3)]['WMO_ID'].unique()
    n_obs = len(pfl_table[pfl_table['WMO_ID'].isin(selected)])
    obs_percentages_3casts.append(n_obs / total_obs * 100)

ax.plot(thresholds_fine, obs_percentages_5casts, linewidth=2, label='Min 5 casts', marker='o', markersize=3)
ax.plot(thresholds_fine, obs_percentages_3casts, linewidth=2, label='Min 3 casts', marker='s', markersize=3)
ax.set_xlabel('Max Average Drift Threshold (km)', fontsize=11)
ax.set_ylabel('Percentage of Total Observations (%)', fontsize=11)
ax.set_title('Data Retention vs Drift Threshold', fontsize=12, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Current: 50 km')
ax.axvline(100, color='orange', linestyle='--', alpha=0.5, label='Suggested: 100 km')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(visualization_dir + "drift_threshold_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Recommendations
print("\n=== RECOMMENDATIONS ===\n")

# Find sweet spots
for min_casts in [3, 5]:
    print(f"\nFor min_casts = {min_casts}:")
    for target_pct in [10, 20, 30, 50]:
        # Find threshold that gives you closest to target percentage
        best_threshold = None
        best_diff = float('inf')
        
        for t in range(10, 501, 10):
            selected = drift_df[(drift_df['avg_distance_per_cast_km'] <= t) & 
                              (drift_df['n_casts'] >= min_casts)]['WMO_ID'].unique()
            n_obs = len(pfl_table[pfl_table['WMO_ID'].isin(selected)])
            pct = n_obs / total_obs * 100
            
            if abs(pct - target_pct) < best_diff:
                best_diff = abs(pct - target_pct)
                best_threshold = t
        
        selected = drift_df[(drift_df['avg_distance_per_cast_km'] <= best_threshold) & 
                          (drift_df['n_casts'] >= min_casts)]['WMO_ID'].unique()
        n_obs = len(pfl_table[pfl_table['WMO_ID'].isin(selected)])
        n_dev = len(selected)
        pct = n_obs / total_obs * 100
        
        print(f"  To keep ~{target_pct}% of data: threshold = {best_threshold} km ({n_dev} devices, {pct:.1f}% data)")

print("\n" + "="*50)
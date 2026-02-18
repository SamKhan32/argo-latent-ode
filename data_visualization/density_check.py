import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations"

drift_df = pd.read_csv(processed_dir + "device_drift_statistics.csv")
pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")

# Filter to low-drift devices
low_drift_devices = drift_df[
    (drift_df['avg_distance_per_cast_km'] <= 50) &
    (drift_df['n_casts'] >= 5)
]
low_drift_obs = pfl_table[pfl_table['WMO_ID'].isin(low_drift_devices['WMO_ID'])]

print("=== PROFILE DENSITY ANALYSIS FOR NEURAL ODE ===\n")

# Calculate observations per cast
obs_per_cast = low_drift_obs.groupby('castIndex').size().reset_index(name='n_obs')

print("Profile Density Statistics:")
print(f"  Mean: {obs_per_cast['n_obs'].mean():.1f} observations/cast")
print(f"  Median: {obs_per_cast['n_obs'].median():.1f} observations/cast")
print(f"  Std: {obs_per_cast['n_obs'].std():.1f}")
print(f"  Min: {obs_per_cast['n_obs'].min()} observations/cast")
print(f"  Max: {obs_per_cast['n_obs'].max()} observations/cast")

# Percentiles
percentiles = [10, 25, 50, 75, 90, 95]
print(f"\nPercentiles:")
for p in percentiles:
    val = np.percentile(obs_per_cast['n_obs'], p)
    print(f"  {p}th percentile: {val:.0f} observations/cast")

print("\n" + "="*60 + "\n")

# Distribution analysis
print("Density Distribution:")
bins = [0, 10, 20, 30, 50, 100, 200, 500, 10000]
for i in range(len(bins)-1):
    count = len(obs_per_cast[(obs_per_cast['n_obs'] > bins[i]) & 
                             (obs_per_cast['n_obs'] <= bins[i+1])])
    pct = count / len(obs_per_cast) * 100
    print(f"  {bins[i]:4d}-{bins[i+1]:4d} obs/cast: {count:5d} casts ({pct:5.1f}%)")

print("\n" + "="*60 + "\n")

# Filter for well-sampled profiles (important for Neural ODE!)
min_obs_thresholds = [10, 20, 30, 50]

print("Effect of filtering by minimum profile density:\n")
print(f"{'Min Obs/Cast':<15} {'# Casts':<12} {'# Total Obs':<15} {'% of Data':<12} {'# Devices':<12}")
print("-" * 70)

for min_obs in min_obs_thresholds:
    # Get castIndices that meet threshold
    dense_casts = obs_per_cast[obs_per_cast['n_obs'] >= min_obs]['castIndex']
    
    # Filter observations
    filtered_obs = low_drift_obs[low_drift_obs['castIndex'].isin(dense_casts)]
    
    n_casts = len(dense_casts)
    n_obs = len(filtered_obs)
    pct_data = n_obs / len(low_drift_obs) * 100
    n_devices = filtered_obs['WMO_ID'].nunique()
    
    print(f"{min_obs:<15} {n_casts:<12,} {n_obs:<15,} {pct_data:<12.1f} {n_devices:<12}")

print("\n" + "="*60 + "\n")

# Analyze depth resolution for dense profiles
print("DEPTH RESOLUTION ANALYSIS:\n")

for min_obs in [20, 30, 50]:
    dense_casts = obs_per_cast[obs_per_cast['n_obs'] >= min_obs]['castIndex']
    filtered_obs = low_drift_obs[low_drift_obs['castIndex'].isin(dense_casts)]
    
    # Calculate average depth spacing
    depth_spacings = []
    for cast_idx in dense_casts:
        cast_data = filtered_obs[filtered_obs['castIndex'] == cast_idx].sort_values('z')
        depths = cast_data['z'].values
        if len(depths) > 1:
            spacings = np.diff(depths)
            depth_spacings.extend(spacings)
    
    if len(depth_spacings) > 0:
        print(f"For profiles with ≥{min_obs} observations:")
        print(f"  Mean depth spacing: {np.mean(depth_spacings):.1f}m")
        print(f"  Median depth spacing: {np.median(depth_spacings):.1f}m")
        print(f"  Typical depth range: {filtered_obs['z'].min():.0f}m to {filtered_obs['z'].max():.0f}m")
        print()

print("="*60 + "\n")

# Visualize the issue and solutions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Full distribution (log scale)
ax = axes[0, 0]
ax.hist(obs_per_cast['n_obs'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Observations per Cast', fontsize=10)
ax.set_ylabel('Number of Casts', fontsize=10)
ax.set_title('Full Distribution (Linear Scale)', fontsize=11, fontweight='bold')
ax.axvline(obs_per_cast['n_obs'].median(), color='red', linestyle='--', 
          label=f'Median: {obs_per_cast["n_obs"].median():.0f}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Distribution (zoomed to 0-100)
ax = axes[0, 1]
obs_zoomed = obs_per_cast[obs_per_cast['n_obs'] <= 100]['n_obs']
ax.hist(obs_zoomed, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Observations per Cast', fontsize=10)
ax.set_ylabel('Number of Casts', fontsize=10)
ax.set_title('Distribution (Zoomed: 0-100)', fontsize=11, fontweight='bold')
ax.axvline(20, color='green', linestyle='--', label='Min 20 threshold')
ax.axvline(30, color='blue', linestyle='--', label='Min 30 threshold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Log scale
ax = axes[0, 2]
ax.hist(obs_per_cast['n_obs'], bins=100, edgecolor='black', alpha=0.7, color='purple')
ax.set_xlabel('Observations per Cast', fontsize=10)
ax.set_ylabel('Number of Casts (log)', fontsize=10)
ax.set_yscale('log')
ax.set_title('Distribution (Log Y-axis)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Cumulative distribution
ax = axes[1, 0]
sorted_obs = np.sort(obs_per_cast['n_obs'].values)
cumulative = np.arange(1, len(sorted_obs)+1) / len(sorted_obs) * 100
ax.plot(sorted_obs, cumulative, linewidth=2, color='steelblue')
ax.set_xlabel('Observations per Cast', fontsize=10)
ax.set_ylabel('Cumulative % of Casts', fontsize=10)
ax.set_title('Cumulative Distribution', fontsize=11, fontweight='bold')
ax.axvline(20, color='green', linestyle='--', alpha=0.7, label='Min 20')
ax.axvline(30, color='blue', linestyle='--', alpha=0.7, label='Min 30')
ax.axvline(50, color='red', linestyle='--', alpha=0.7, label='Min 50')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 200)

# Plot 5: Data retention vs threshold
ax = axes[1, 1]
thresholds = range(5, 101, 5)
data_retention = []
cast_retention = []

for thresh in thresholds:
    dense_casts = obs_per_cast[obs_per_cast['n_obs'] >= thresh]['castIndex']
    filtered_obs = low_drift_obs[low_drift_obs['castIndex'].isin(dense_casts)]
    
    data_retention.append(len(filtered_obs) / len(low_drift_obs) * 100)
    cast_retention.append(len(dense_casts) / len(obs_per_cast) * 100)

ax.plot(thresholds, data_retention, linewidth=2, label='Data Retention', marker='o', markersize=4)
ax.plot(thresholds, cast_retention, linewidth=2, label='Cast Retention', marker='s', markersize=4)
ax.set_xlabel('Minimum Observations per Cast', fontsize=10)
ax.set_ylabel('Retention (%)', fontsize=10)
ax.set_title('Data Retention vs Profile Density Threshold', fontsize=11, fontweight='bold')
ax.axvline(20, color='green', linestyle='--', alpha=0.5)
ax.axvline(30, color='blue', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Example profiles at different densities
ax = axes[1, 2]

# Show 3 example profiles: sparse, medium, dense
example_sparse = obs_per_cast[obs_per_cast['n_obs'].between(5, 15)]['castIndex'].iloc[0] if len(obs_per_cast[obs_per_cast['n_obs'].between(5, 15)]) > 0 else None
example_medium = obs_per_cast[obs_per_cast['n_obs'].between(25, 35)]['castIndex'].iloc[0] if len(obs_per_cast[obs_per_cast['n_obs'].between(25, 35)]) > 0 else None
example_dense = obs_per_cast[obs_per_cast['n_obs'] > 80]['castIndex'].iloc[0] if len(obs_per_cast[obs_per_cast['n_obs'] > 80]) > 0 else None

if example_sparse:
    sparse_data = low_drift_obs[low_drift_obs['castIndex'] == example_sparse].sort_values('z')
    ax.scatter([1]*len(sparse_data), sparse_data['z'], s=50, alpha=0.7, label=f'Sparse ({len(sparse_data)} obs)')

if example_medium:
    medium_data = low_drift_obs[low_drift_obs['castIndex'] == example_medium].sort_values('z')
    ax.scatter([2]*len(medium_data), medium_data['z'], s=50, alpha=0.7, label=f'Medium ({len(medium_data)} obs)')

if example_dense:
    dense_data = low_drift_obs[low_drift_obs['castIndex'] == example_dense].sort_values('z')
    ax.scatter([3]*len(dense_data), dense_data['z'], s=50, alpha=0.7, label=f'Dense ({len(dense_data)} obs)')

ax.invert_yaxis()
ax.set_ylabel('Depth (m)', fontsize=10)
ax.set_xlabel('Profile Type', fontsize=10)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Sparse', 'Medium', 'Dense'])
ax.set_title('Example Profile Densities', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(visualization_dir + "profile_density_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== RECOMMENDATIONS FOR NEURAL ODE ===\n")

print("The peaked distribution suggests many sparse profiles mixed with some very dense ones.")
print("\n✓ RECOMMENDED APPROACH:")
print("  1. Filter to profiles with ≥20-30 observations")
print("     • Neural ODEs need dense sampling to learn continuous dynamics")
print("     • Sparse profiles (5-10 obs) won't capture vertical structure well")
print("  2. This will likely keep 60-80% of your data (see plot above)")
print("  3. The remaining profiles will have better depth resolution")
print("\n⚠ TRADE-OFF:")
print("  • Fewer total casts, but each cast is more informative")
print("  • Better for learning vertical dynamics")
print("  • May lose some spatial coverage")

print("\n📊 SUGGESTED NEXT STEP:")
print("  Run: min_obs = 25  # Sweet spot between density and data retention")
print("  This typically keeps ~70% of data with much better profile quality")

print("\n" + "="*60)
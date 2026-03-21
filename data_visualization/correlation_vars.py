import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations/"

drift_df = pd.read_csv(processed_dir + "device_drift_statistics.csv")
pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")

# Filter to low-drift devices
low_drift_devices = drift_df[
    (drift_df['avg_distance_per_cast_km'] <= 50) &
    (drift_df['n_casts'] >= 5)
]
low_drift_obs = pfl_table[pfl_table['WMO_ID'].isin(low_drift_devices['WMO_ID'])]

print("=== VARIABLE vs DEPTH CORRELATION ANALYSIS ===\n")

# Define variables to analyze
variables = ['Temperature', 'Salinity', 'Oxygen', 'Pressure', 'Nitrate', 'pH', 'Chlorophyll']
available_vars = [v for v in variables if v in low_drift_obs.columns]

print(f"Analyzing variables: {', '.join(available_vars)}\n")

# Calculate correlations
correlations = {}
for var in available_vars:
    # Remove NaN values
    valid_data = low_drift_obs[['z', var]].dropna()
    
    if len(valid_data) > 10:  # Need enough data points
        # Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(valid_data['z'], valid_data[var])
        # Spearman correlation (monotonic, handles non-linear)
        spearman_corr, spearman_p = spearmanr(valid_data['z'], valid_data[var])
        
        correlations[var] = {
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'spearman': spearman_corr,
            'spearman_p': spearman_p,
            'n_obs': len(valid_data)
        }

# Print correlation table
print("CORRELATION WITH DEPTH:")
print(f"{'Variable':<15} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'p-value':<12} {'N obs':<10}")
print("-" * 80)
for var, corr in correlations.items():
    print(f"{var:<15} {corr['pearson']:>11.3f} {corr['pearson_p']:>11.2e} "
          f"{corr['spearman']:>11.3f} {corr['spearman_p']:>11.2e} {corr['n_obs']:>10,}")

print("\n" + "="*80 + "\n")

# Interpretation
print("INTERPRETATION:")
print("• Positive correlation: Variable increases with depth")
print("• Negative correlation: Variable decreases with depth")
print("• |r| > 0.7: Strong correlation")
print("• 0.3 < |r| < 0.7: Moderate correlation")
print("• |r| < 0.3: Weak correlation\n")

for var, corr in correlations.items():
    r = corr['pearson']
    if abs(r) > 0.7:
        strength = "STRONG"
    elif abs(r) > 0.3:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    
    direction = "increases" if r > 0 else "decreases"
    print(f"{var:15s}: {strength:8s} correlation - {direction} with depth (r={r:.3f})")

print("\n" + "="*80 + "\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation bar chart
ax1 = fig.add_subplot(gs[0, :])
var_names = list(correlations.keys())
pearson_vals = [correlations[v]['pearson'] for v in var_names]
spearman_vals = [correlations[v]['spearman'] for v in var_names]

x = np.arange(len(var_names))
width = 0.35

bars1 = ax1.bar(x - width/2, pearson_vals, width, label='Pearson (linear)', alpha=0.8)
bars2 = ax1.bar(x + width/2, spearman_vals, width, label='Spearman (monotonic)', alpha=0.8)

# Color bars by strength
for i, (p_val, s_val) in enumerate(zip(pearson_vals, spearman_vals)):
    if abs(p_val) > 0.7:
        bars1[i].set_color('darkred' if p_val > 0 else 'darkblue')
    elif abs(p_val) > 0.3:
        bars1[i].set_color('orange' if p_val > 0 else 'lightblue')
    else:
        bars1[i].set_color('gray')
    
    if abs(s_val) > 0.7:
        bars2[i].set_color('darkred' if s_val > 0 else 'darkblue')
    elif abs(s_val) > 0.3:
        bars2[i].set_color('orange' if s_val > 0 else 'lightblue')
    else:
        bars2[i].set_color('gray')

ax1.set_ylabel('Correlation Coefficient', fontsize=12)
ax1.set_title('Variable Correlation with Depth', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(var_names, rotation=45, ha='right')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.axhline(0.7, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(-0.7, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(0.3, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(-0.3, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis='y')
ax1.set_ylim(-1.05, 1.05)

# Plots 2-7: Scatter plots for each variable vs depth
for idx, var in enumerate(available_vars[:6]):  # First 6 variables
    row = (idx // 3) + 1
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    # Sample data for plotting (too many points otherwise)
    valid_data = low_drift_obs[['z', var]].dropna()
    
    # Downsample for visualization if too many points
    if len(valid_data) > 10000:
        valid_data = valid_data.sample(n=10000, random_state=42)
    
    # Hexbin plot for dense data
    if len(valid_data) > 1000:
        hexbin = ax.hexbin(valid_data[var], valid_data['z'], 
                          gridsize=50, cmap='YlOrRd', mincnt=1, alpha=0.8)
        plt.colorbar(hexbin, ax=ax, label='Count')
    else:
        ax.scatter(valid_data[var], valid_data['z'], 
                  alpha=0.3, s=1, c='steelblue')
    
    # Fit and plot trend line
    if len(valid_data) > 10:
        z_fit = np.linspace(valid_data['z'].min(), valid_data['z'].max(), 100)
        # Use polynomial fit to capture non-linear trends
        coeffs = np.polyfit(valid_data[var], valid_data['z'], deg=2)
        var_fit = np.linspace(valid_data[var].min(), valid_data[var].max(), 100)
        z_pred = np.polyval(coeffs, var_fit)
        ax.plot(var_fit, z_pred, 'r-', linewidth=2, alpha=0.7, label='Trend')
    
    ax.invert_yaxis()
    ax.set_xlabel(f'{var}', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    
    # Add correlation to title
    if var in correlations:
        r = correlations[var]['pearson']
        ax.set_title(f'{var} vs Depth (r={r:.3f})', fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{var} vs Depth', fontsize=11, fontweight='bold')
    
    ax.grid(alpha=0.3)

plt.savefig(visualization_dir + "variable_depth_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved to: " + visualization_dir + "variable_depth_correlations.png")

# Additional analysis: Depth-stratified statistics
print("\n=== DEPTH-STRATIFIED ANALYSIS ===\n")

depth_bins = [(0, 50, 'Surface'), 
              (50, 200, 'Thermocline'),
              (200, 1000, 'Intermediate'),
              (1000, 10000, 'Deep')]

print("Mean values by depth layer:\n")
print(f"{'Depth Layer':<20} " + " ".join([f"{v:>12s}" for v in available_vars]))
print("-" * (20 + 13 * len(available_vars)))

for z_min, z_max, layer_name in depth_bins:
    layer_data = low_drift_obs[(low_drift_obs['z'] >= z_min) & 
                                (low_drift_obs['z'] < z_max)]
    
    if len(layer_data) > 0:
        layer_str = f"{layer_name} ({z_min}-{z_max}m)"
        values = []
        for var in available_vars:
            mean_val = layer_data[var].mean()
            values.append(f"{mean_val:>12.2f}" if not np.isnan(mean_val) else f"{'N/A':>12s}")
        
        print(f"{layer_str:<20} " + " ".join(values))

print("\n" + "="*80 + "\n")

# Vertical gradients
print("VERTICAL GRADIENTS (change per 100m depth):\n")
print(f"{'Variable':<15} {'Gradient':<20} {'Units per 100m'}")
print("-" * 55)

for var in available_vars:
    valid_data = low_drift_obs[['z', var]].dropna()
    
    if len(valid_data) > 10:
        # Linear regression to estimate gradient
        coeffs = np.polyfit(valid_data['z'], valid_data[var], deg=1)
        gradient_per_100m = coeffs[0] * 100  # Convert to per 100m
        
        print(f"{var:<15} {gradient_per_100m:>19.4f}")

print("\n" + "="*80 + "\n")

# Create depth profile plots (mean ± std)
fig, axes = plt.subplots(1, min(len(available_vars), 4), figsize=(16, 6))
if len(available_vars) == 1:
    axes = [axes]

for idx, var in enumerate(available_vars[:4]):
    ax = axes[idx] if idx < len(axes) else None
    if ax is None:
        break
    
    # Bin by depth
    depth_bins_fine = np.arange(0, low_drift_obs['z'].max() + 50, 50)
    
    means = []
    stds = []
    bin_centers = []
    
    for i in range(len(depth_bins_fine) - 1):
        bin_data = low_drift_obs[(low_drift_obs['z'] >= depth_bins_fine[i]) & 
                                  (low_drift_obs['z'] < depth_bins_fine[i+1])][var]
        
        if len(bin_data) > 0:
            means.append(bin_data.mean())
            stds.append(bin_data.std())
            bin_centers.append((depth_bins_fine[i] + depth_bins_fine[i+1]) / 2)
    
    means = np.array(means)
    stds = np.array(stds)
    bin_centers = np.array(bin_centers)
    
    # Remove NaNs
    valid = ~np.isnan(means)
    means = means[valid]
    stds = stds[valid]
    bin_centers = bin_centers[valid]
    
    if len(means) > 0:
        # Plot mean profile with std envelope
        ax.plot(means, bin_centers, 'b-', linewidth=2, label='Mean')
        ax.fill_betweenx(bin_centers, means - stds, means + stds, 
                        alpha=0.3, color='blue', label='±1 std')
        
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)', fontsize=11)
        ax.set_xlabel(var, fontsize=11)
        ax.set_title(f'Vertical Profile: {var}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(visualization_dir + "vertical_profiles.png", dpi=300, bbox_inches='tight')
plt.show()

print("Vertical profiles saved to: " + visualization_dir + "vertical_profiles.png")
print("\n=== Analysis Complete! ===")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations"

drift_df = pd.read_csv(processed_dir + "PFL1_device_drift_statistics.csv")
pfl_table = pd.read_csv(processed_dir + "PFL1_preprocessed.csv")

# Filter to low-drift devices
low_drift_devices = drift_df[
    (drift_df['avg_distance_per_cast_km'] <= 50) &
    (drift_df['n_casts'] >= 5)
]
low_drift_obs = pfl_table[pfl_table['WMO_ID'].isin(low_drift_devices['WMO_ID'])]

print("=== NEURAL ODE DATA QUALITY ASSESSMENT ===\n")

# 1. Check variable completeness (critical for Neural ODE)
print("1. VARIABLE COMPLETENESS (% non-null observations):\n")
variables = ['Temperature', 'Salinity', 'Oxygen', 'Pressure', 'Nitrate', 'pH', 'Chlorophyll']
for var in variables:
    if var in low_drift_obs.columns:
        completeness = (1 - low_drift_obs[var].isna().sum() / len(low_drift_obs)) * 100
        n_valid = low_drift_obs[var].notna().sum()
        print(f"   {var:15s}: {completeness:5.1f}% ({n_valid:,} valid observations)")
    else:
        print(f"   {var:15s}: NOT FOUND")

print("\n" + "="*60 + "\n")

# 2. Check depth coverage (important for ODEs modeling vertical profiles)
print("2. DEPTH COVERAGE:\n")
print(f"   Depth range: {low_drift_obs['z'].min():.1f}m to {low_drift_obs['z'].max():.1f}m")
print(f"   Median depth: {low_drift_obs['z'].median():.1f}m")
print(f"   Mean observations per cast: {len(low_drift_obs) / low_drift_obs['castIndex'].nunique():.1f}")

# Check depth distribution
depth_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
print(f"\n   Depth distribution:")
for i in range(len(depth_bins)-1):
    count = len(low_drift_obs[(low_drift_obs['z'] >= depth_bins[i]) & 
                               (low_drift_obs['z'] < depth_bins[i+1])])
    pct = count / len(low_drift_obs) * 100
    print(f"   {depth_bins[i]:5.0f}-{depth_bins[i+1]:5.0f}m: {count:7,} obs ({pct:5.1f}%)")

print("\n" + "="*60 + "\n")

# 3. Temporal coverage (for Neural ODE dynamics)
print("3. TEMPORAL COVERAGE:\n")
if 'date' in low_drift_obs.columns:
    low_drift_obs['date'] = pd.to_datetime(low_drift_obs['date'], errors='coerce')
    date_range = low_drift_obs['date'].max() - low_drift_obs['date'].min()
    print(f"   Date range: {low_drift_obs['date'].min()} to {low_drift_obs['date'].max()}")
    print(f"   Total span: {date_range.days} days ({date_range.days/365.25:.1f} years)")
    
    # Temporal distribution
    low_drift_obs['year'] = low_drift_obs['date'].dt.year
    yearly_counts = low_drift_obs.groupby('year').size()
    print(f"\n   Observations by year:")
    for year, count in yearly_counts.items():
        print(f"   {year}: {count:,} observations")
else:
    print("   Date information not available")

print("\n" + "="*60 + "\n")

# 4. Variable co-occurrence (critical for your permutation tests!)
print("4. VARIABLE CO-OCCURRENCE MATRIX:\n")
print("   (Shows % of observations where both variables are present)\n")

available_vars = [v for v in variables if v in low_drift_obs.columns]
cooccurrence = pd.DataFrame(index=available_vars, columns=available_vars, dtype=float)

for var1 in available_vars:
    for var2 in available_vars:
        both_present = (low_drift_obs[var1].notna() & low_drift_obs[var2].notna()).sum()
        pct = both_present / len(low_drift_obs) * 100
        cooccurrence.loc[var1, var2] = pct

print(cooccurrence.to_string(float_format=lambda x: f'{x:5.1f}%'))

print("\n" + "="*60 + "\n")

# 5. Check for complete profiles (all variables measured)
print("5. COMPLETE PROFILES ANALYSIS:\n")

# Define different variable sets for your permutation tests
variable_sets = {
    'Core (T,S,P)': ['Temperature', 'Salinity', 'Pressure'],
    'Core + O2': ['Temperature', 'Salinity', 'Pressure', 'Oxygen'],
    'Core + Nutrients': ['Temperature', 'Salinity', 'Pressure', 'Nitrate'],
    'Core + Bio': ['Temperature', 'Salinity', 'Pressure', 'Oxygen', 'Chlorophyll'],
    'Full Suite': ['Temperature', 'Salinity', 'Pressure', 'Oxygen', 'Nitrate', 'pH', 'Chlorophyll']
}

for set_name, var_list in variable_sets.items():
    # Check which variables actually exist
    existing_vars = [v for v in var_list if v in low_drift_obs.columns]
    
    if len(existing_vars) == 0:
        print(f"   {set_name}: No variables available")
        continue
    
    # Find rows where all variables are present
    mask = low_drift_obs[existing_vars[0]].notna()
    for var in existing_vars[1:]:
        mask = mask & low_drift_obs[var].notna()
    
    n_complete = mask.sum()
    pct_complete = n_complete / len(low_drift_obs) * 100
    n_casts_complete = low_drift_obs[mask]['castIndex'].nunique()
    
    print(f"   {set_name}:")
    print(f"      Variables: {', '.join(existing_vars)}")
    print(f"      Complete observations: {n_complete:,} ({pct_complete:.1f}%)")
    print(f"      Complete casts: {n_casts_complete}")

print("\n" + "="*60 + "\n")

# 6. Neural ODE-specific recommendations
print("6. NEURAL ODE RECOMMENDATIONS:\n")

print("✓ STRENGTHS of your low-drift dataset:")
print("  • Low spatial drift = more stable dynamics to learn")
print("  • Multiple profiles per device = good for learning temporal evolution")
print("  • 75% of observations retained = plenty of training data")

print("\n⚠ CONSIDERATIONS:")
print("  • Check variable co-occurrence matrix above - you'll need high co-occurrence")
print("    for input-output permutation tests")
print("  • If key variables (like Oxygen or Nitrate) have low coverage, you may need")
print("    to relax drift threshold to get more of those measurements")
print("  • Neural ODEs work best with dense temporal sampling - check that your")
print("    devices have frequent enough profiles")

print("\n📊 SUGGESTED PERMUTATION TESTS based on your data:")
available_core = [v for v in ['Temperature', 'Salinity', 'Pressure'] if v in low_drift_obs.columns]
available_extended = [v for v in ['Oxygen', 'Nitrate', 'pH', 'Chlorophyll'] if v in low_drift_obs.columns]

if len(available_core) == 3:
    print("\n   High-confidence tests (>70% co-occurrence):")
    for var in available_extended:
        if var in low_drift_obs.columns:
            cooc = (low_drift_obs['Temperature'].notna() & 
                   low_drift_obs['Salinity'].notna() & 
                   low_drift_obs['Pressure'].notna() & 
                   low_drift_obs[var].notna()).sum() / len(low_drift_obs) * 100
            if cooc > 70:
                print(f"      • Predict {var} from T,S,P ({cooc:.1f}% co-occurrence)")

print("\n" + "="*60 + "\n")

# 7. Create visualization of data completeness
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variable completeness bar chart
ax = axes[0, 0]
var_completeness = {}
for var in available_vars:
    var_completeness[var] = (1 - low_drift_obs[var].isna().sum() / len(low_drift_obs)) * 100

bars = ax.barh(list(var_completeness.keys()), list(var_completeness.values()), 
               color='steelblue', edgecolor='black')
ax.set_xlabel('Completeness (%)', fontsize=11)
ax.set_title('Variable Completeness', fontsize=12, fontweight='bold')
ax.axvline(70, color='red', linestyle='--', alpha=0.5, label='70% threshold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Plot 2: Depth distribution
ax = axes[0, 1]
ax.hist(low_drift_obs['z'], bins=50, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Depth (m)', fontsize=11)
ax.set_ylabel('Number of Observations', fontsize=11)
ax.set_title('Depth Distribution', fontsize=12, fontweight='bold')
ax.invert_xaxis()  # Deeper is more to the right
ax.grid(alpha=0.3)

# Plot 3: Co-occurrence heatmap
ax = axes[1, 0]
cooccurrence_numeric = cooccurrence.astype(float)
im = ax.imshow(cooccurrence_numeric, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(available_vars)))
ax.set_yticks(range(len(available_vars)))
ax.set_xticklabels(available_vars, rotation=45, ha='right')
ax.set_yticklabels(available_vars)
ax.set_title('Variable Co-occurrence (%)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(available_vars)):
    for j in range(len(available_vars)):
        text = ax.text(j, i, f'{cooccurrence_numeric.iloc[i, j]:.0f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label='Co-occurrence %')

# Plot 4: Observations per cast
ax = axes[1, 1]
obs_per_cast = low_drift_obs.groupby('castIndex').size()
ax.hist(obs_per_cast, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Observations per Cast', fontsize=11)
ax.set_ylabel('Number of Casts', fontsize=11)
ax.set_title('Profile Density (Neural ODE needs dense profiles)', fontsize=12, fontweight='bold')
ax.axvline(obs_per_cast.median(), color='red', linestyle='--', 
          label=f'Median: {obs_per_cast.median():.0f}')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(visualization_dir + "neural_ode_data_quality.png", dpi=300, bbox_inches='tight')
plt.show()

print("=== Analysis complete! ===")
print(f"Visualization saved to: {visualization_dir}neural_ode_data_quality.png")
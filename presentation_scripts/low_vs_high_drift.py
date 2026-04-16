import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

df = pd.read_csv('data/processed/PFL1_preprocessed.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

high_drift_id = 1901467.0
low_drift_id = 5903890.0

high = df[df['WMO_ID'] == high_drift_id].sort_values('date')
low = df[df['WMO_ID'] == low_drift_id].sort_values('date')

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.OCEAN, facecolor='#0a1628')
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#555555', zorder=3)
ax.set_facecolor('#0a1628')
fig.patch.set_facecolor('#0a1628')

ax.plot(high['lon'].values, high['lat'].values,
        color='#E84545', linewidth=1.2, alpha=0.85,
        transform=ccrs.Geodetic(), zorder=5, label='High Drift (1901467)')

ax.plot(low['lon'].values, low['lat'].values,
        color='#4A90D9', linewidth=1.2, alpha=0.85,
        transform=ccrs.Geodetic(), zorder=5, label='Low Drift (5903890)')

# Mark start and end points
for track, color in [(high, '#E84545'), (low, '#4A90D9')]:
    ax.scatter(track['lon'].iloc[0], track['lat'].iloc[0],
               color=color, marker='o', s=60, zorder=6,
               edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree())
    ax.scatter(track['lon'].iloc[-1], track['lat'].iloc[-1],
               color=color, marker='X', s=80, zorder=6,
               edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree())

legend_elements = [
    plt.Line2D([0],[0], color='#E84545', linewidth=2, label='High Drift'),
    plt.Line2D([0],[0], color='#4A90D9', linewidth=2, label='Low Drift'),
    plt.scatter([],[],  marker='o', color='white', s=40, label='Start'),
    plt.scatter([],[],  marker='X', color='white', s=40, label='End'),
]

ax.legend(handles=legend_elements, loc='lower center',
          bbox_to_anchor=(0.5, 0.05),
          facecolor='#1a1a2e', edgecolor='#555', labelcolor='white',
          fontsize=11, framealpha=0.8)

plt.tight_layout()
plt.savefig('drift_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0a1628')
plt.close()
print('saved drift_comparison.png')
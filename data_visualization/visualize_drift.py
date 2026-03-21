import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial.distance import euclidean
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects

# Load the processed data
processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations/"

pfl_table = pd.read_csv(processed_dir + "PFL1_preprocessed.csv")


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth (in km).
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    return c * r


def calculate_device_drift(df):
    """
    Calculate drift statistics for each WMO_ID device.
    Returns a DataFrame with drift metrics per device.
    """
    # Get unique cast locations per device
    cast_locations = df.groupby('castIndex').first().reset_index()
    cast_locations = cast_locations.dropna(subset=['lat', 'lon', 'WMO_ID'])
    
    # Group by WMO_ID
    devices = cast_locations.groupby('WMO_ID')
    
    drift_data = []
    
    for wmo_id, device_casts in devices:
        if len(device_casts) < 2:
            continue  # Need at least 2 casts to calculate drift
        
        # Sort by date/time if available, otherwise by castIndex
        if 'date' in device_casts.columns:
            device_casts = device_casts.sort_values('date')
        else:
            device_casts = device_casts.sort_values('castIndex')
        
        # Calculate distances between consecutive casts
        lats = device_casts['lat'].values
        lons = device_casts['lon'].values
        
        distances = []
        for i in range(len(lats) - 1):
            dist = calculate_haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
            distances.append(dist)
        
        # Calculate total path length
        total_distance = np.sum(distances)
        
        # Calculate straight-line distance (start to end)
        straight_line_dist = calculate_haversine_distance(lats[0], lons[0], lats[-1], lons[-1])
        
        # Calculate statistics
        drift_data.append({
            'WMO_ID': wmo_id,
            'n_casts': len(device_casts),
            'total_distance_km': total_distance,
            'straight_line_distance_km': straight_line_dist,
            'avg_distance_per_cast_km': np.mean(distances),
            'max_distance_per_cast_km': np.max(distances),
            'min_distance_per_cast_km': np.min(distances),
            'std_distance_per_cast_km': np.std(distances),
            'tortuosity': total_distance / straight_line_dist if straight_line_dist > 0 else np.nan,
            'start_lat': lats[0],
            'start_lon': lons[0],
            'end_lat': lats[-1],
            'end_lon': lons[-1],
            'lat_range': lats.max() - lats.min(),
            'lon_range': lons.max() - lons.min()
        })
    
    drift_df = pd.DataFrame(drift_data)
    return drift_df


def filter_low_drift_devices(drift_df, max_avg_drift_km=50, min_casts=5):
    """
    Filter devices with low drift.
    
    Parameters:
    -----------
    drift_df : pd.DataFrame
        Output from calculate_device_drift
    max_avg_drift_km : float
        Maximum average drift per cast in km
    min_casts : int
        Minimum number of casts required
    """
    low_drift = drift_df[
        (drift_df['avg_distance_per_cast_km'] <= max_avg_drift_km) &
        (drift_df['n_casts'] >= min_casts)
    ]
    return low_drift


def plot_drift_histogram(drift_df, figsize=(14, 10)):
    """
    Plot histogram of drift statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Average distance per cast
    ax = axes[0, 0]
    ax.hist(drift_df['avg_distance_per_cast_km'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Average Distance per Cast (km)', fontsize=11)
    ax.set_ylabel('Number of Devices', fontsize=11)
    ax.set_title('Distribution of Average Drift per Cast', fontsize=12, fontweight='bold')
    ax.axvline(50, color='red', linestyle='--', label='50 km threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Total distance
    ax = axes[0, 1]
    ax.hist(drift_df['total_distance_km'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Total Path Distance (km)', fontsize=11)
    ax.set_ylabel('Number of Devices', fontsize=11)
    ax.set_title('Distribution of Total Path Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Number of casts per device
    ax = axes[1, 0]
    ax.hist(drift_df['n_casts'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Number of Casts', fontsize=11)
    ax.set_ylabel('Number of Devices', fontsize=11)
    ax.set_title('Distribution of Casts per Device', fontsize=12, fontweight='bold')
    ax.axvline(5, color='red', linestyle='--', label='5 cast threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Tortuosity (path length / straight line)
    ax = axes[1, 1]
    tortuosity_clean = drift_df['tortuosity'].dropna()
    tortuosity_clean = tortuosity_clean[tortuosity_clean < 10]  # Remove outliers
    ax.hist(tortuosity_clean, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Tortuosity (path/straight-line)', fontsize=11)
    ax.set_ylabel('Number of Devices', fontsize=11)
    ax.set_title('Distribution of Path Tortuosity', fontsize=12, fontweight='bold')
    ax.axvline(1, color='red', linestyle='--', label='Straight line (1.0)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_device_trajectories(df, drift_df, n_devices=20, color_by='avg_distance_per_cast_km', 
                            figsize=(16, 10), use_geodesic=True):
    """
    Plot trajectories of selected devices on a map using geodesic (great circle) paths.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original PFL data
    drift_df : pd.DataFrame
        Drift statistics
    n_devices : int
        Number of devices to plot (will select lowest drift)
    color_by : str
        Column to use for coloring trajectories
    use_geodesic : bool
        If True, use Geodetic transform for curved lines on sphere
    """
    # Get cast locations
    cast_locations = df.groupby('castIndex').first().reset_index()
    cast_locations = cast_locations.dropna(subset=['lat', 'lon', 'WMO_ID'])
    
    # Select devices (lowest drift first)
    selected_devices = drift_df.nsmallest(n_devices, 'avg_distance_per_cast_km')
    
    # Create map
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot trajectories
    cmap = plt.cm.RdYlGn_r  # Red = high drift, Green = low drift
    norm = plt.Normalize(vmin=selected_devices[color_by].min(), 
                        vmax=selected_devices[color_by].max())
    
    for idx, row in selected_devices.iterrows():
        wmo_id = row['WMO_ID']
        device_casts = cast_locations[cast_locations['WMO_ID'] == wmo_id].sort_values('castIndex')
        
        if len(device_casts) < 2:
            continue
        
        lons = device_casts['lon'].values
        lats = device_casts['lat'].values
        
        color = cmap(norm(row[color_by]))
        
        # Plot trajectory using geodesic transform for proper great circle paths
        if use_geodesic:
            # Use Geodetic transform - this will draw great circles
            ax.plot(lons, lats, '-', linewidth=1.5, alpha=0.7, color=color, 
                   transform=ccrs.Geodetic(), zorder=3)
        else:
            # Regular PlateCarree (straight lines on the projection)
            ax.plot(lons, lats, '-', linewidth=1.5, alpha=0.7, color=color, 
                   transform=ccrs.PlateCarree(), zorder=3)
        
        # Mark start (green circle) and end (red triangle)
        ax.plot(lons[0], lats[0], 'o', color='darkgreen', markersize=8, 
               markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        ax.plot(lons[-1], lats[-1], '^', color='darkred', markersize=8,
               markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        
        # Mark intermediate points
        ax.scatter(lons[1:-1], lats[1:-1], s=15, c=[color], alpha=0.5, 
                  edgecolors='white', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(f'{color_by.replace("_", " ").title()}', fontsize=11)
    
    transform_type = "Geodesic (Great Circle)" if use_geodesic else "PlateCarree (Straight)"
    ax.set_title(f'Trajectories of {n_devices} Lowest-Drift Devices [{transform_type}]\n' + 
                f'(●=Start, ▲=End, •=Intermediate)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_drift_vs_casts_scatter(drift_df, figsize=(12, 8)):
    """
    Scatter plot: drift vs number of casts.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        drift_df['n_casts'],
        drift_df['avg_distance_per_cast_km'],
        c=drift_df['total_distance_km'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Number of Casts', fontsize=12)
    ax.set_ylabel('Average Drift per Cast (km)', fontsize=12)
    ax.set_title('Drift vs Number of Casts per Device', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add threshold lines
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50 km drift threshold')
    ax.axvline(5, color='orange', linestyle='--', alpha=0.7, label='5 cast threshold')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, label='Total Path Distance (km)')
    
    plt.tight_layout()
    return fig


# ===== RUN ANALYSIS =====

print("Calculating drift for all devices...")
drift_df = calculate_device_drift(pfl_table)
print(f"Total devices with 2+ casts: {len(drift_df)}")

# Save drift statistics
drift_df.to_csv(processed_dir + "PFL1_device_drift_statistics.csv", index=False)
print(f"Saved drift statistics to {processed_dir}device_drift_statistics.csv")

# Print summary statistics
print("\n=== Drift Statistics ===")
print(f"Mean avg drift per cast: {drift_df['avg_distance_per_cast_km'].mean():.2f} km")
print(f"Median avg drift per cast: {drift_df['avg_distance_per_cast_km'].median():.2f} km")
print(f"Max avg drift per cast: {drift_df['avg_distance_per_cast_km'].max():.2f} km")
print(f"Mean number of casts: {drift_df['n_casts'].mean():.1f}")

# Filter low drift devices
low_drift_devices = filter_low_drift_devices(drift_df, max_avg_drift_km=50, min_casts=5)
print(f"\n=== Low Drift Devices (avg drift ≤ 50 km, ≥ 5 casts) ===")
print(f"Number of low-drift devices: {len(low_drift_devices)}")
print(f"Percentage of total: {len(low_drift_devices)/len(drift_df)*100:.1f}%")

# Save low drift device list
low_drift_devices.to_csv(processed_dir + "PFL1_low_drift_devices.csv", index=False)

# Create visualizations
print("\nCreating drift histograms...")
fig1 = plot_drift_histogram(drift_df)
plt.savefig(visualization_dir + "drift_histograms.png", dpi=300, bbox_inches='tight')
plt.show()

print("Creating drift vs casts scatter plot...")
fig2 = plot_drift_vs_casts_scatter(drift_df)
plt.savefig(visualization_dir + "drift_vs_casts.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plotting trajectories of 20 lowest-drift devices (GEODESIC)...")
fig3 = plot_device_trajectories(pfl_table, drift_df, n_devices=20, use_geodesic=True)
plt.savefig(visualization_dir + "low_drift_trajectories_geodesic.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plotting trajectories of 20 highest-drift devices (GEODESIC)...")
fig4 = plot_device_trajectories(pfl_table, drift_df.nlargest(20, 'avg_distance_per_cast_km'), 
                                n_devices=20, use_geodesic=True)
plt.savefig(visualization_dir + "high_drift_trajectories_geodesic.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Done! ===")
print(f"Low-drift device IDs saved to: {processed_dir}low_drift_devices.csv")
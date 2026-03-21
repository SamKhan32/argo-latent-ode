import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the processed data
processed_dir = "data/processed/"
visualization_dir = "data_visualization/new_visualizations"

pfl_table = pd.read_csv(processed_dir + "PFL_preprocessed.csv")


def plot_observation_heatmap_map(df, title="Observation Density", figsize=(16, 10), bins=100, use_log=True):
    """
    Create a heatmap overlaid on a geographic map.
    """
    # Remove NaN values
    df_clean = df.dropna(subset=['lat', 'lon'])
    
    # Get lat/lon ranges
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    
    # Create 2D histogram
    counts, lon_edges, lat_edges, _ = binned_statistic_2d(
        df_clean['lon'], 
        df_clean['lat'], 
        None, 
        statistic='count',
        bins=bins,
        range=[lon_range, lat_range]
    )
    
    # Replace zeros with NaN
    counts[counts == 0] = np.nan
    
    # Create figure with map projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to your data range
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3, alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Choose normalization
    if use_log:
        norm = LogNorm(vmin=np.nanmin(counts), vmax=np.nanmax(counts))
        cmap = 'hot'
    else:
        vmax = np.nanpercentile(counts, 95)
        norm = None
        cmap = 'hot'
    
    # Plot heatmap
    im = ax.imshow(
        counts.T, 
        origin='lower',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        interpolation='bilinear',
        norm=norm if use_log else None,
        vmax=None if use_log else vmax,
        alpha=0.7  # Make semi-transparent so we can see the map underneath
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Number of Observations (log scale)' if use_log else 'Number of Observations', fontsize=11)
    
    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics text
    stats_text = f"Total observations: {len(df_clean):,}\n"
    stats_text += f"Unique casts: {df_clean['castIndex'].nunique():,}\n"
    stats_text += f"Max density: {np.nanmax(counts):.0f}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10,
            zorder=10)
    
    plt.tight_layout()
    return fig, ax, counts


def plot_cast_heatmap_map(df, title="Cast Density", figsize=(16, 10), bins=100, use_log=True):
    """
    Create a cast density heatmap overlaid on a geographic map.
    """
    # Get one row per cast
    df_casts = df.groupby('castIndex').first().reset_index()
    df_casts = df_casts.dropna(subset=['lat', 'lon'])
    
    # Get lat/lon ranges
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    
    # Create 2D histogram
    counts, lon_edges, lat_edges, _ = binned_statistic_2d(
        df_casts['lon'], 
        df_casts['lat'], 
        None, 
        statistic='count',
        bins=bins,
        range=[lon_range, lat_range]
    )
    
    # Replace zeros with NaN
    counts[counts == 0] = np.nan
    
    # Create figure with map projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='wheat', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', linewidth=0.3, alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Choose normalization
    if use_log:
        norm = LogNorm(vmin=np.nanmin(counts), vmax=np.nanmax(counts))
        cmap = 'YlOrRd'
    else:
        vmax = np.nanpercentile(counts, 95)
        norm = None
        cmap = 'YlOrRd'
    
    # Plot heatmap
    im = ax.imshow(
        counts.T, 
        origin='lower',
        extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        interpolation='bilinear',
        norm=norm if use_log else None,
        vmax=None if use_log else vmax,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Number of Casts (log scale)' if use_log else 'Number of Casts', fontsize=11)
    
    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics
    stats_text = f"Total casts: {len(df_casts):,}\n"
    stats_text += f"Max density: {np.nanmax(counts):.0f}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10,
            zorder=10)
    
    plt.tight_layout()
    return fig, ax, counts


def plot_scatter_map(df, title="Observation Locations", figsize=(16, 10), alpha=0.3, s=1):
    """
    Scatter plot on map with coastlines.
    """
    df_clean = df.dropna(subset=['lat', 'lon'])
    
    # Get lat/lon ranges
    lon_range = [-97.5480, -4.7344]
    lat_range = [-14.0801, 54.1231]
    
    # Create figure with map projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Scatter plot
    scatter = ax.scatter(
        df_clean['lon'], 
        df_clean['lat'], 
        s=s, 
        alpha=alpha, 
        c='red',
        edgecolors='none',
        transform=ccrs.PlateCarree()
    )
    
    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Statistics
    stats_text = f"Total observations: {len(df_clean):,}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10,
            zorder=10)
    
    plt.tight_layout()
    return fig, ax


# Create map-based visualizations
print("Creating PFL observation heatmap on map...")
fig1, ax1, counts1 = plot_observation_heatmap_map(
    pfl_table, 
    title="PFL: Observation Density on Map (Log Scale)", 
    bins=100,
    use_log=True
)
plt.savefig(processed_dir + "PFL_observation_map.png", dpi=300, bbox_inches='tight')
plt.show()

print("Creating PFL cast heatmap on map...")
fig2, ax2, counts2 = plot_cast_heatmap_map(
    pfl_table, 
    title="PFL: Cast Density on Map (Log Scale)", 
    bins=100,
    use_log=True
)
plt.savefig(processed_dir + "PFL_cast_map.png", dpi=300, bbox_inches='tight')
plt.show()

print("Creating PFL scatter plot on map...")
fig3, ax3 = plot_scatter_map(
    pfl_table, 
    title="PFL: All Observation Locations",
    alpha=0.1,
    s=0.5
)
plt.savefig(processed_dir + "PFL_scatter_map.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Done! ===")
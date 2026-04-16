import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('data/processed/PFL1_preprocessed.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

variables = ['Chlorophyll', 'Nitrate', 'pH']
colors = {'Chlorophyll': '#7ED321', 'Nitrate': '#BD10E0', 'pH': '#E84545'}

float_coverage = df.groupby('WMO_ID')[variables].apply(lambda g: g.notna().any())
trajectories = df.sort_values('date').groupby('WMO_ID')[['lat','lon']].apply(lambda g: g.values)

def draw_trajectories(ax, wmo_ids, color, alpha, lw):
    for wmo_id in wmo_ids:
        if wmo_id not in trajectories.index:
            continue
        traj = trajectories[wmo_id]
        if len(traj) < 2:
            continue
        ax.plot(traj[:,1], traj[:,0], color=color, linewidth=lw,
                alpha=alpha, transform=ccrs.Geodetic(), zorder=5)

def make_plot(highlight_var, highlight_color, filename, title):
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor='#0a1628')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#555555', zorder=3)
    ax.set_facecolor('#0a1628')
    fig.patch.set_facecolor('#0a1628')

    all_floats = list(trajectories.index)

    if highlight_var is None:
        draw_trajectories(ax, all_floats, '#4A90D9', alpha=0.7, lw=1.0)
        legend_elements = [plt.Line2D([0],[0], color='#4A90D9', linewidth=2, label='Temperature, Salinity, Oxygen (227 floats)')]
    else:
        highlight_floats = float_coverage[float_coverage[highlight_var]].index.tolist()
        gray_floats = [f for f in all_floats if f not in highlight_floats]
        draw_trajectories(ax, gray_floats, '#1e3a5f', alpha=0.5, lw=0.6)

        draw_trajectories(ax, highlight_floats, highlight_color, alpha=0.9, lw=1.4)
        n = len(highlight_floats)
        legend_elements = [
            plt.Line2D([0],[0], color='#1e3a5f', linewidth=2, label='No data'),
            plt.Line2D([0],[0], color=highlight_color, linewidth=2, label=f'{highlight_var} ({n} floats)'),
        ]

    legend = ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.12),
              facecolor='#1a1a2e', edgecolor='#555', labelcolor='white',
              fontsize=11, framealpha=0.8)

    ax.set_title(title, color='white', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0a1628')
    plt.close()
    print(f'saved {filename}')

make_plot(None, None, 'frame1_all.png', 'All Argo Floats — Temperature, Salinity, Oxygen')
make_plot('Chlorophyll', colors['Chlorophyll'], 'frame2_chlorophyll.png', 'Chlorophyll Floats')
make_plot('Nitrate', colors['Nitrate'], 'frame3_nitrate.png', 'Nitrate Floats')
make_plot('pH', colors['pH'], 'frame4_ph.png', 'pH Floats')

print('done')
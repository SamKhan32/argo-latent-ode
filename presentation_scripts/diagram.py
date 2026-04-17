import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

def box(ax, x, y, w, h, color, label, sublabel=None):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#555',
                                    linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.3, sublabel, ha='center', va='center',
                fontsize=8, color='white', alpha=0.85, zorder=4)

def arrow(ax, x1, x2, y=2.5, color='#555'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8))

# boxes: x positions
positions = {
    'input':   1.2,
    'encoder': 3.0,
    'latent':  5.0,
    'ode':     7.0,
    'decoder': 9.0,
    'output':  10.8,
}

colors = {
    'input':   '#4A90D9',
    'encoder': '#2c3e50',
    'latent':  '#9B59B6',
    'ode':     '#E84545',
    'decoder': '#2c3e50',
    'output':  '#4A90D9',
}

box(ax, positions['input'],   2.5, 1.6, 1.4, colors['input'],   'Profile Input', 'T / S / O\nvs. Depth')
box(ax, positions['encoder'], 2.5, 1.6, 1.4, colors['encoder'], 'Encoder')
box(ax, positions['latent'],  2.5, 1.6, 1.4, colors['latent'],  'Latent State', 'z(t)')
box(ax, positions['ode'],     2.5, 1.6, 1.4, colors['ode'],     'Neural ODE', 'dz/dt = f(z,t)')
box(ax, positions['decoder'], 2.5, 1.6, 1.4, colors['decoder'], 'Decoder')
box(ax, positions['output'],  2.5, 1.6, 1.4, colors['input'],   'Reconstruction', 'T / S / O')

# main arrows
for a, b in [('input','encoder'), ('encoder','latent'),
             ('latent','ode'), ('ode','decoder'), ('decoder','output')]:
    arrow(ax, positions[a] + 0.8, positions[b] - 0.8)

# probe head branching off latent
probe_x = positions['latent']
probe_y = 0.9
box(ax, probe_x, probe_y, 1.6, 0.8, '#7ED321', 'Probe Head', 'Chlorophyll')

ax.annotate('', xy=(probe_x, probe_y + 0.4), xytext=(probe_x, 2.5 - 0.7),
            arrowprops=dict(arrowstyle='->', color='#7ED321', lw=1.8, linestyle='dashed'))

ax.text(probe_x + 0.95, 1.65, 'never trained\non this →', fontsize=8,
        color='#7ED321', ha='left', va='center', style='italic')

plt.tight_layout()
plt.savefig('pipeline_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved pipeline_diagram.png')
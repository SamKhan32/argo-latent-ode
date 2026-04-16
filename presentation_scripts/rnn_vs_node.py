import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

np.random.seed(42)
t_known = np.array([0, 1.5, 3, 4, 6, 7.5, 9, 10])
y_known = np.array([2, 2.8, 2.3, 3.5, 3.1, 4.2, 3.8, 4.5])

cs = CubicSpline(t_known, y_known)
t_smooth = np.linspace(0, 10, 300)
y_smooth = cs(t_smooth)

dot_style = dict(color='#2c3e50', s=100, zorder=5)
legend_style = dict(fontsize=10, frameon=True, facecolor='white',
                    edgecolor='#cccccc', framealpha=1.0)

def base_ax(title):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('', fontsize=1)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(1, 5.5)
    ax.set_xticks(t_known)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

def add_shading(ax):
    for i in range(len(t_known) - 1):
        ax.axvspan(t_known[i], t_known[i+1], alpha=0.25, color='#E84545', zorder=1)
    for t in t_known:
        ax.axvspan(t - 0.15, t + 0.15, alpha=0.5, color='#7ED321', zorder=2)

def add_rnn_and_dots(ax):
    ax.plot(t_known, y_known, color='#4A90D9', linewidth=2, zorder=3)
    return ax.scatter(t_known, y_known, **dot_style, label='Observations')

def make_wiggle(rng, i):
    t_seg = np.linspace(t_known[i], t_known[i+1], 40)
    y_start, y_end = y_known[i], y_known[i+1]
    noise = rng.normal(0, 0.3, len(t_seg))
    noise[0] = 0
    noise[-1] = 0
    taper = np.sin(np.linspace(0, np.pi, len(t_seg)))
    y_alt = np.linspace(y_start, y_end, len(t_seg)) + noise * taper
    return t_seg, y_alt

# pre-generate all wiggly paths so they're consistent across frames
rng = np.random.default_rng(7)
all_paths = []
for path_idx in range(3):
    path = []
    for i in range(len(t_known) - 1):
        t_seg, y_alt = make_wiggle(rng, i)
        path.append((t_seg, y_alt))
    all_paths.append(path)

def base_legend_handles(sc):
    rnn_line = plt.Line2D([0],[0], color='#4A90D9', linewidth=2, label='RNN path')
    shading_patch = plt.Rectangle((0,0), 1, 1, fc='#E84545', alpha=0.4, label='Undefined region')
    green_patch = plt.Rectangle((0,0), 1, 1, fc='#7ED321', alpha=0.5, label='Defined (observation)')
    return [sc, rnn_line, shading_patch, green_patch]

# --- Frame 0: just points ---
fig, ax = base_ax('Observations Over Time')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
ax.legend(handles=[sc], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame0.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame0.png')

# --- Frame 1: RNN line ---
fig, ax = base_ax('RNN: Discrete Timesteps')
line, = ax.plot(t_known, y_known, color='#4A90D9', linewidth=2, zorder=3, label='RNN path')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
ax.legend(handles=[sc, line], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame1.png')

# --- Frame 2: shading + green bands ---
fig, ax = base_ax('RNN: Gaps Between Timesteps')
add_shading(ax)
sc = add_rnn_and_dots(ax)
ax.legend(handles=base_legend_handles(sc), **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame2.png')

# --- Frames 3-5: one wiggle path added per frame ---
for path_idx in range(3):
    fig, ax = base_ax(f'RNN: Possible Path {path_idx + 1}')
    add_shading(ax)
    sc = add_rnn_and_dots(ax)
    for p in range(path_idx + 1):
        for t_seg, y_alt in all_paths[p]:
            ax.plot(t_seg, y_alt, color='#9B59B6', linewidth=1.5,
                    alpha=0.8, zorder=4, linestyle='--')
    alt_line = plt.Line2D([0],[0], color='#9B59B6', linewidth=1.5,
                          linestyle='--', label='Possible path')
    ax.legend(handles=base_legend_handles(sc) + [alt_line], **legend_style)
    plt.tight_layout()
    plt.savefig(f'rnn_frame{path_idx + 3}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'saved rnn_frame{path_idx + 3}.png')

# --- Frame 6: Neural ODE ---
fig, ax = base_ax('Neural ODE: Continuous Dynamics')
line_ode, = ax.plot(t_smooth, y_smooth, color='#7ED321', linewidth=2.5,
                    zorder=3, label='Neural ODE trajectory')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
t_query = np.array([0.8, 2.2, 5.0, 8.3])
y_query = cs(t_query)
sc_q = ax.scatter(t_query, y_query, color='#E84545', s=80, zorder=6,
                   marker='D', label='Query at any time')
for tq in t_query:
    ax.axvline(tq, color='#E84545', linewidth=0.8, linestyle=':', alpha=0.4)
ax.legend(handles=[sc, line_ode, sc_q], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame6.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame6.png')
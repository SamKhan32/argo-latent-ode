import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

np.random.seed(42)
t_known = np.array([0, 1.5, 3, 4, 6, 7.5, 9, 10])
y_known = np.array([2, 2.8, 2.3, 3.5, 3.1, 4.2, 3.8, 4.5])

cs = CubicSpline(t_known, y_known)
t_smooth = np.linspace(0, 10, 300)
y_smooth = cs(t_smooth)

t_predict = 5.0
y_predict = cs(t_predict)

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

def add_predict_point(ax):
    ax.scatter(t_predict, 3.0, color='#F5A623', s=120, zorder=7,
               marker='D', label='Point to predict')
    ax.axvline(t_predict, color='#F5A623', linewidth=0.8, linestyle=':', alpha=0.5, zorder=6)

def add_predict_point_solved(ax):
    ax.scatter(t_predict, y_predict, color='#F5A623', s=120, zorder=7,
               marker='D', label='Point to predict')
    ax.axvline(t_predict, color='#F5A623', linewidth=0.8, linestyle=':', alpha=0.5, zorder=6)

def make_smooth_path(i, offset):
    t_seg = np.linspace(t_known[i], t_known[i+1], 40)
    y_start, y_end = y_known[i], y_known[i+1]
    baseline = np.linspace(y_start, y_end, len(t_seg))
    bump = offset * np.sin(np.linspace(0, np.pi, len(t_seg)))
    return t_seg, baseline + bump

offsets_per_path = [
    [0.6, 0.5, 0.7, 0.5, 0.6, 0.5, 0.6],
    [-0.6, -0.5, -0.7, -0.5, -0.6, -0.5, -0.6],
    [0.7, -0.6, 0.5, -0.7, 0.6, -0.5, 0.7],
]

all_paths = []
for offsets in offsets_per_path:
    path = []
    for i in range(len(t_known) - 1):
        t_seg, y_alt = make_smooth_path(i, offsets[i])
        path.append((t_seg, y_alt))
    all_paths.append(path)

def base_legend_handles(sc):
    rnn_line = plt.Line2D([0],[0], color='#4A90D9', linewidth=2, label='RNN path')
    shading_patch = plt.Rectangle((0,0), 1, 1, fc='#E84545', alpha=0.4, label='Undefined region')
    green_patch = plt.Rectangle((0,0), 1, 1, fc='#7ED321', alpha=0.5, label='Defined (observation)')
    predict_handle = plt.scatter([],[], color='#F5A623', s=120, marker='D', label='Point to predict')
    return [sc, rnn_line, shading_patch, green_patch, predict_handle]

# --- Frame 0: just points, no prediction ---
fig, ax = base_ax('Observations Over Time')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
ax.legend(handles=[sc], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame0.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame0.png')

# --- Frame 1: points + prediction target ---
fig, ax = base_ax('Can We Predict Between Observations?')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
add_predict_point(ax)
predict_handle = plt.scatter([],[], color='#F5A623', s=120, marker='D', label='Point to predict')
ax.legend(handles=[sc, predict_handle], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame1.png')

# --- Frame 2: RNN line + prediction target ---
fig, ax = base_ax('RNN: Discrete Timesteps')
line, = ax.plot(t_known, y_known, color='#4A90D9', linewidth=2, zorder=3, label='RNN path')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
add_predict_point(ax)
predict_handle = plt.scatter([],[], color='#F5A623', s=120, marker='D', label='Point to predict')
ax.legend(handles=[sc, line, predict_handle], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame2.png')

# --- Frame 3: shading + green bands + prediction target ---
fig, ax = base_ax('RNN: Gaps Between Timesteps')
add_shading(ax)
sc = add_rnn_and_dots(ax)
add_predict_point(ax)
ax.legend(handles=base_legend_handles(sc), **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame3.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame3.png')

# --- Frames 4-6: one smooth path per frame + prediction target ---
for path_idx in range(3):
    fig, ax = base_ax(f'RNN: Possible Path {path_idx + 1}')
    add_shading(ax)
    sc = add_rnn_and_dots(ax)
    add_predict_point(ax)
    for t_seg, y_alt in all_paths[path_idx]:
        ax.plot(t_seg, y_alt, color='#9B59B6', linewidth=1.8,
                alpha=0.85, zorder=4, linestyle='--')
    alt_line = plt.Line2D([0],[0], color='#9B59B6', linewidth=1.8,
                          linestyle='--', label=f'Possible path {path_idx + 1}')
    ax.legend(handles=base_legend_handles(sc) + [alt_line], **legend_style)
    plt.tight_layout()
    plt.savefig(f'rnn_frame{path_idx + 4}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'saved rnn_frame{path_idx + 4}.png')

# --- Frame 7: Neural ODE + prediction solved ---
fig, ax = base_ax('Neural ODE: Continuous Dynamics')
line_ode, = ax.plot(t_smooth, y_smooth, color='#7ED321', linewidth=2.5,
                    zorder=3, label='Neural ODE trajectory')
sc = ax.scatter(t_known, y_known, **dot_style, label='Observations')
add_predict_point_solved(ax)
predict_handle = plt.scatter([],[], color='#F5A623', s=120, marker='D', label='Point to predict')
ax.legend(handles=[sc, line_ode, predict_handle], **legend_style)
plt.tight_layout()
plt.savefig('rnn_frame7.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved rnn_frame7.png')
import numpy as np
import matplotlib.pyplot as plt

def make_field(ax, title):
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)

    U = Y
    V = -X

    speed = np.sqrt(U**2 + V**2)
    speed[speed == 0] = 1
    U_n = U / speed
    V_n = V / speed

    ax.quiver(X, Y, U_n, V_n, speed, cmap='Blues', alpha=0.75,
              scale=22, width=0.004)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Velocity', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_facecolor('white')

def integrate_trajectory(x0, y0, steps=500, dt=0.02):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(steps):
        u = y
        v = -x
        x += u * dt
        y += v * dt
        if abs(x) > 2.5 or abs(y) > 2.5:
            break
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Frame 1: vector field only ---
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
make_field(ax, 'Neural ODE: A Vector Field Over State Space')
plt.tight_layout()
plt.savefig('node_field_frame1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved node_field_frame1.png')

# --- Frame 2: vector field + integrated trajectories ---
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
make_field(ax, 'Neural ODE: Integrate From Any Starting Point')

start_points = [(-1.5, 0.0), (1.0, 0.0), (0.0, 1.8)]
colors = ['#E84545', '#9B59B6', '#F5A623']

for (x0, y0), color in zip(start_points, colors):
    xs, ys = integrate_trajectory(x0, y0)
    ax.plot(xs, ys, color=color, linewidth=2, zorder=5)
    ax.scatter(x0, y0, color=color, s=80, zorder=6,
               marker='o', edgecolors='white', linewidths=0.8)

plt.tight_layout()
plt.savefig('node_field_frame2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('saved node_field_frame2.png')
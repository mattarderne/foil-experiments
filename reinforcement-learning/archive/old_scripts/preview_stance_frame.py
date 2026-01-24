"""Quick preview of stance learning animation frame."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from animate_stance_learning import draw_foil_and_rider, compute_body_positions
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Stance Learning - Mid-Episode Comparison', fontsize=14)

# Sample states at different training stages with crash status
# (name, z, vx, theta, stance, color, is_crashed, duration)
stages = [
    ('Baseline', -0.15, 3.5, -0.15, 0.02, 'gray', True, 1.2),      # Crashed - nose down
    ('5k steps', -0.20, 3.0, -0.20, 0.03, '#e74c3c', True, 1.0),   # Crashed - wrong direction
    ('30k steps', 0.14, 4.4, 0.03, -0.033, '#f39c12', False, 5.0), # Flying well
    ('50k steps', 0.15, 4.5, 0.02, -0.035, '#27ae60', False, 5.0), # Optimal
]

# Create fake trail history for non-crashed states
def make_trail(z, theta, stance, vx, n=10):
    trail = []
    for i in range(n):
        # Slight variations for visual effect
        trail.append({
            'z': z - 0.01 * (n - i),
            'theta': theta,
            'stance': stance,
            'vx': vx
        })
    return trail

for ax, (name, z, vx, theta, stance, color, is_crashed, duration) in zip(axes, stages):
    # Water
    ax.axhspan(-1.5, 0.0, alpha=0.3, color='dodgerblue')
    ax.axhline(y=0.0, color='blue', linewidth=2, alpha=0.5)

    # Water streaks (only if not crashed)
    if not is_crashed:
        np.random.seed(42)
        for j in range(6):
            streak_y = np.random.uniform(-0.6, -0.1)
            streak_x = np.random.uniform(-1.5, 1.5)
            ax.plot([streak_x, streak_x - 0.15], [streak_y, streak_y],
                    'white', linewidth=1.5, alpha=0.5)

    # Trail history for non-crashed
    trail = make_trail(z, theta, stance, vx) if not is_crashed else None

    draw_foil_and_rider(ax, z, theta, vx, 1.0, stance, color, alpha=0.9,
                        trail_history=trail, is_crashed=is_crashed)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.0, 2.5)  # Taller to fit realistic body proportions
    ax.set_aspect('equal')
    ax.set_facecolor('lightcyan')

    status = f"CRASHED @ {duration:.1f}s" if is_crashed else f"({duration:.1f}s)"
    title_color = 'red' if is_crashed else color
    ax.set_title(f'{name}\nstance: {stance*100:+.1f}cm {status}', fontsize=10, color=title_color)

plt.tight_layout()
plt.savefig('stance_frame_preview.png', dpi=150)
print('Saved: stance_frame_preview.png')

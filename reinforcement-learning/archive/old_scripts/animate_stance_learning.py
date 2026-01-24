"""
Animate stance learning progression with overlaid trajectories.

Shows how the agent's stance strategy evolves during training,
with multiple training stages overlaid on one animation.
Uses the full foil + body visualization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO

from foil_env.stance_env import StanceEnv


def run_episode(model_path: str, seed: int = 42):
    """Run episode with given model and return trajectory."""
    model = PPO.load(model_path)
    env = StanceEnv()

    obs, _ = env.reset(seed=seed)
    done = False

    trajectory = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            't': info['step_count'] * 0.01,
            'z': info['z'],
            'vx': info['vx'],
            'theta': np.radians(info['theta_deg']),
            'stance': info['stance_offset'],
            'is_stalled': info['is_stalled'],
        })
        done = terminated or truncated

    return trajectory


def run_random_episode(seed: int = 42):
    """Run episode with random/no action for baseline."""
    env = StanceEnv()
    obs, _ = env.reset(seed=seed)
    done = False

    trajectory = []
    while not done:
        # No stance adjustment (stay at initial random position)
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            't': info['step_count'] * 0.01,
            'z': info['z'],
            'vx': info['vx'],
            'theta': np.radians(info['theta_deg']),
            'stance': info['stance_offset'],
            'is_stalled': info['is_stalled'],
        })
        done = terminated or truncated

    return trajectory


def compute_body_positions(z, theta, stance_offset):
    """Compute key body positions for motion trails."""
    water_y = 0.0
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z
    x_center = 0.0

    # Human proportions for ~1.75m tall person (must match draw_foil_and_rider)
    board_thickness = 0.05
    shin_len = 0.43
    thigh_len = 0.45
    torso_len = 0.55
    head_radius = 0.11
    stance_width = 0.30
    shoulder_half_width = 0.18

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Feet on board
    l_foot = transform((-stance_width/2 + stance_offset, board_thickness))
    r_foot = transform((stance_width/2 + stance_offset, board_thickness))

    # Body upright
    leg_height = shin_len + thigh_len * 0.85
    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)
    shoulder_world = (hip_world[0], hip_world[1] + torso_len)
    head_world = (shoulder_world[0], shoulder_world[1] + head_radius + 0.02)

    # Hands (static pose) - must match draw_foil_and_rider
    upper_arm_len = 0.32
    forearm_len = 0.27
    l_shoulder = (shoulder_world[0] - shoulder_half_width, shoulder_world[1] - 0.05)
    r_shoulder = (shoulder_world[0] + shoulder_half_width, shoulder_world[1] - 0.05)
    l_hand = (l_shoulder[0] - upper_arm_len * 0.6 - forearm_len * 0.5,
              l_shoulder[1] - upper_arm_len * 0.5 - forearm_len * 0.4)
    r_hand = (r_shoulder[0] + upper_arm_len * 0.6 + forearm_len * 0.5,
              r_shoulder[1] - upper_arm_len * 0.5 - forearm_len * 0.4)

    return {
        'l_foot': l_foot, 'r_foot': r_foot,
        'l_hand': l_hand, 'r_hand': r_hand,
        'head': head_world
    }


def draw_foil_and_rider(ax, z, theta, vx, t, stance_offset, color, alpha=1.0, label=None, x_offset=0.0,
                        trail_history=None, is_crashed=False):
    """Draw full foil assembly and rider at given state with optional motion trails."""
    water_y = 0.0
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z
    x_center = x_offset

    # Foil dimensions
    board_length = 0.6
    board_thickness = 0.05
    mast_length = 0.7
    fuse_length = 0.7

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # === RIGID FOIL ASSEMBLY ===
    # Board
    board_local = [
        (-board_length/2, 0), (board_length/2, 0),
        (board_length/2, board_thickness), (-board_length/2, board_thickness),
    ]
    board_world = [transform(p) for p in board_local]
    ax.add_patch(Polygon(board_world, facecolor='saddlebrown', edgecolor='black',
                         linewidth=2, alpha=alpha, label=label))

    # Mast
    mast_top = transform((0, 0))
    mast_bottom = transform((0, -mast_length))
    ax.plot([mast_top[0], mast_bottom[0]], [mast_top[1], mast_bottom[1]],
            'dimgray', linewidth=5, alpha=alpha, solid_capstyle='round')

    # Fuselage
    fuse_front = transform((fuse_length * 0.4, -mast_length))
    fuse_rear = transform((-fuse_length * 0.6, -mast_length))
    ax.plot([fuse_front[0], fuse_rear[0]], [fuse_front[1], fuse_rear[1]],
            'dimgray', linewidth=4, alpha=alpha, solid_capstyle='round')

    # Front wing
    wing_local = [
        (fuse_length * 0.4, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length + 0.02),
        (fuse_length * 0.4, -mast_length + 0.02),
    ]
    wing_world = [transform(p) for p in wing_local]
    ax.add_patch(Polygon(wing_world, facecolor='silver', edgecolor='black',
                         linewidth=1.5, alpha=alpha))

    # Rear stabilizer
    stab_local = [
        (-fuse_length * 0.6, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length + 0.015),
        (-fuse_length * 0.6, -mast_length + 0.015),
    ]
    stab_world = [transform(p) for p in stab_local]
    ax.add_patch(Polygon(stab_world, facecolor='silver', edgecolor='black',
                         linewidth=1, alpha=alpha))

    # === RIDER (static pose, positioned by stance_offset) ===
    # Human proportions for ~1.75m tall person (70kg)
    shin_len = 0.43       # Tibia length
    thigh_len = 0.45      # Femur length
    torso_len = 0.55      # Hip to shoulder
    head_radius = 0.11    # ~22cm head height
    stance_width = 0.30   # Feet ~shoulder width apart
    shoulder_half_width = 0.18  # Total shoulder width ~36cm

    # Feet on board (offset by stance position)
    l_foot_local = (-stance_width/2 + stance_offset, board_thickness)
    r_foot_local = (stance_width/2 + stance_offset, board_thickness)
    l_foot = transform(l_foot_local)
    r_foot = transform(r_foot_local)

    # Body stays upright in world coords
    leg_height = shin_len + thigh_len * 0.85  # Standing height
    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)
    shoulder_world = (hip_world[0], hip_world[1] + torso_len)
    head_world = (shoulder_world[0], shoulder_world[1] + head_radius + 0.02)

    # Solve knee positions (IK)
    def solve_knee(foot, hip, shin_l, thigh_l):
        dx = hip[0] - foot[0]
        dy = hip[1] - foot[1]
        dist = np.sqrt(dx*dx + dy*dy)
        max_reach = shin_l + thigh_l
        min_reach = abs(shin_l - thigh_l)
        dist = np.clip(dist, min_reach + 0.01, max_reach - 0.01)
        cos_foot = (shin_l**2 + dist**2 - thigh_l**2) / (2 * shin_l * dist)
        cos_foot = np.clip(cos_foot, -1, 1)
        angle_to_hip = np.arctan2(dy, dx)
        angle_at_foot = np.arccos(cos_foot)
        knee_angle = angle_to_hip + angle_at_foot
        return (foot[0] + shin_l * np.cos(knee_angle),
                foot[1] + shin_l * np.sin(knee_angle))

    l_knee = solve_knee(l_foot, hip_world, shin_len, thigh_len)
    r_knee = solve_knee(r_foot, hip_world, shin_len, thigh_len)

    # Draw legs
    for foot, knee, leg_color in [(l_foot, l_knee, 'royalblue'), (r_foot, r_knee, 'crimson')]:
        ax.plot([foot[0], knee[0]], [foot[1], knee[1]], leg_color,
                linewidth=6, alpha=alpha, solid_capstyle='round')
        ax.plot([knee[0], hip_world[0]], [knee[1], hip_world[1]], leg_color,
                linewidth=8, alpha=alpha, solid_capstyle='round')
        ax.plot(knee[0], knee[1], 'ko', markersize=5, alpha=alpha)

    # Torso
    ax.plot([hip_world[0], shoulder_world[0]], [hip_world[1], shoulder_world[1]],
            'darkgreen', linewidth=10, alpha=alpha, solid_capstyle='round')

    # Arms (static, slightly out for balance)
    upper_arm_len = 0.32   # Humerus
    forearm_len = 0.27     # Radius/ulna
    for side, arm_color in [(-1, 'royalblue'), (1, 'crimson')]:
        shoulder = (shoulder_world[0] + side * shoulder_half_width, shoulder_world[1] - 0.05)
        elbow = (shoulder[0] + side * upper_arm_len * 0.6,
                 shoulder[1] - upper_arm_len * 0.5)
        wrist = (elbow[0] + side * forearm_len * 0.5,
                 elbow[1] - forearm_len * 0.4)
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], arm_color,
                linewidth=5, alpha=alpha*0.8)
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], arm_color,
                linewidth=4, alpha=alpha*0.8)

    # Head
    head = Circle(head_world, head_radius, facecolor='peachpuff',
                  edgecolor='black', linewidth=2, alpha=alpha)
    ax.add_patch(head)
    # Eye
    ax.plot(head_world[0] + 0.03, head_world[1] + 0.02, 'ko',
            markersize=3, alpha=alpha)

    # Stance indicator line (from mast to rider center)
    mast_board_world = transform((0, board_thickness))
    rider_center = ((l_foot[0] + r_foot[0])/2, mast_board_world[1])
    ax.plot([mast_board_world[0], rider_center[0]],
            [mast_board_world[1] - 0.02, rider_center[1] - 0.02],
            color=color, linewidth=3, alpha=alpha*0.8)

    # === MOTION TRAILS ===
    if trail_history and len(trail_history) > 1:
        trail_colors = {
            'l_foot': 'royalblue', 'r_foot': 'crimson',
            'l_hand': 'royalblue', 'r_hand': 'crimson',
            'head': 'orange'
        }
        dt = 0.01 * 3  # frame_skip = 3

        for key in ['l_foot', 'r_foot', 'l_hand', 'r_hand', 'head']:
            points = []
            for j, hist_frame in enumerate(trail_history):
                pos = compute_body_positions(hist_frame['z'], hist_frame['theta'], hist_frame['stance'])
                # Offset by movement
                frames_ago = len(trail_history) - 1 - j
                x_trail_offset = frames_ago * hist_frame['vx'] * dt * 2
                points.append((pos[key][0] - x_trail_offset, pos[key][1]))

            # Draw trail
            trail_color = trail_colors[key]
            for j in range(len(points) - 1):
                trail_alpha = (j + 1) / len(points) * 0.5 * alpha
                linewidth = 2 if 'foot' in key else (1.5 if 'hand' in key else 2.5)
                ax.plot([points[j][0], points[j+1][0]],
                        [points[j][1], points[j+1][1]],
                        color=trail_color, linewidth=linewidth, alpha=trail_alpha)

    # === CRASH INDICATOR ===
    if is_crashed:
        # Red X over the crashed foil
        cx, cy = x_center, board_center_y + 0.5
        size = 0.3
        ax.plot([cx - size, cx + size], [cy - size, cy + size], 'r-', linewidth=4, alpha=0.8)
        ax.plot([cx - size, cx + size], [cy + size, cy - size], 'r-', linewidth=4, alpha=0.8)
        ax.text(cx, cy + size + 0.1, 'CRASHED', ha='center', fontsize=10,
                color='red', fontweight='bold', alpha=0.9)


def create_overlaid_animation(checkpoint_dir: str, output_path: str = 'stance_learning.mp4'):
    """Create animation with multiple training stages shown side by side."""

    # Find checkpoints
    checkpoints = []
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith('.zip'):
            path = os.path.join(checkpoint_dir, f)
            if 'stance_' in f:
                # Extract step number
                if 'final' in f:
                    steps = 50000
                else:
                    try:
                        steps = int(f.split('_')[1].split('_')[0])
                    except:
                        steps = int(f.replace('stance_', '').replace('_steps.zip', ''))
                checkpoints.append((steps, path, f))

    checkpoints.sort()
    print(f"Found {len(checkpoints)} checkpoints")

    # Select key stages (early, mid, final)
    if len(checkpoints) >= 3:
        selected = [checkpoints[0], checkpoints[len(checkpoints)//2], checkpoints[-1]]
    else:
        selected = checkpoints

    # Add random baseline
    print("Running baseline (no learning)...")
    baseline_traj = run_random_episode(seed=42)
    print(f"  Baseline: {len(baseline_traj)} steps, {baseline_traj[-1]['t']:.2f}s")

    # Run episodes for each checkpoint
    trajectories = [('Baseline', baseline_traj, 'gray', len(baseline_traj))]
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

    for i, (steps, path, filename) in enumerate(selected):
        print(f"Running {filename}...")
        traj = run_episode(path, seed=42)
        label = f"{steps//1000}k steps"
        original_len = len(traj)
        trajectories.append((label, traj, colors[i % len(colors)], original_len))
        print(f"  {label}: {len(traj)} steps, {traj[-1]['t']:.2f}s, stance={traj[-1]['stance']:+.3f}m")

    # Find max frames
    max_frames = max(len(t[1]) for t in trajectories)

    # Pad shorter trajectories with "crashed" state
    for name, traj, color, orig_len in trajectories:
        last_frame = traj[-1].copy()
        while len(traj) < max_frames:
            traj.append(last_frame)

    # Create animation - side by side layout
    n_stages = len(trajectories)
    print(f"\nCreating animation with {max_frames} frames, {n_stages} stages...")
    fig = plt.figure(figsize=(16, 8))

    # Top row: foil visualizations side by side
    axes_foil = [fig.add_subplot(2, n_stages, i+1) for i in range(n_stages)]
    # Bottom: combined time series plot
    ax_plot = fig.add_subplot(2, 1, 2)

    fig.suptitle('Stance Learning Progression', fontsize=14)

    frame_skip = 3
    frames_to_render = list(range(0, max_frames, frame_skip))

    # Trail length in frames
    trail_len = 15

    def animate(frame_num):
        i = frames_to_render[frame_num]
        t = trajectories[0][1][min(i, len(trajectories[0][1])-1)]['t']

        # Draw each stage in its own subplot
        for idx, (name, traj, color, orig_len) in enumerate(trajectories):
            ax = axes_foil[idx]
            ax.clear()

            frame = traj[min(i, len(traj)-1)]

            # Check if crashed (past original trajectory length)
            is_crashed = i >= orig_len

            # Water
            ax.axhspan(-1.5, 0.0, alpha=0.3, color='dodgerblue')
            ax.axhline(y=0.0, color='blue', linewidth=2, alpha=0.5)

            # Water streaks
            if not is_crashed:
                np.random.seed(int(frame['t'] * 100) % 1000)
                water_offset = (frame['t'] * frame['vx'] * 2) % 6 - 3
                for j in range(8):
                    streak_y = np.random.uniform(-0.8, -0.1)
                    streak_x = ((j * 0.5 - water_offset) % 4) - 2
                    ax.plot([streak_x, streak_x - 0.15], [streak_y, streak_y],
                            'white', linewidth=1.5, alpha=0.5)

            # Build trail history
            trail_start = max(0, i - trail_len)
            trail_history = traj[trail_start:i+1] if not is_crashed else None

            # Draw foil and rider
            draw_foil_and_rider(ax, frame['z'], frame['theta'], frame['vx'],
                               frame['t'], frame['stance'], color, alpha=0.9,
                               trail_history=trail_history, is_crashed=is_crashed)

            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.2, 2.5)  # Taller to fit realistic body proportions
            ax.set_aspect('equal')
            ax.set_facecolor('lightcyan')

            # Title with stats
            stance_cm = frame['stance'] * 100
            duration = orig_len * 0.01
            status = f"({duration:.1f}s)" if not is_crashed else f"CRASHED @ {duration:.1f}s"
            title_color = color if not is_crashed else 'red'
            ax.set_title(f"{name}\nstance: {stance_cm:+.1f}cm {status}",
                        fontsize=10, color=title_color)

            if idx == 0:
                ax.set_ylabel('Height (m)')

        # Time series plot
        ax_plot.clear()

        for name, traj, color, orig_len in trajectories:
            times = [f['t'] for f in traj[:i+1]]
            stances = [f['stance'] * 100 for f in traj[:i+1]]

            linestyle = '--' if 'Baseline' in name else '-'
            ax_plot.plot(times, stances, color=color, linestyle=linestyle,
                        linewidth=2, label=f'{name}')

        ax_plot.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax_plot.axvline(x=t, color='black', linestyle='-', alpha=0.3, linewidth=2)
        ax_plot.set_xlim(0, 5.5)
        ax_plot.set_ylim(-25, 25)
        ax_plot.set_xlabel('Time (s)')
        ax_plot.set_ylabel('Stance Offset (cm)')
        ax_plot.set_title(f'Stance Position Over Time (t={t:.2f}s)')
        ax_plot.legend(loc='upper right', fontsize=9, ncol=4)
        ax_plot.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render),
                        interval=33, blit=False)

    try:
        writer = FFMpegWriter(fps=30, metadata=dict(artist='PumpFoil RL'))
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error ({e}), saving as GIF...")
        from matplotlib.animation import PillowWriter
        gif_path = output_path.replace('.mp4', '.gif')
        writer = PillowWriter(fps=15)
        anim.save(gif_path, writer=writer)
        print(f"Saved: {gif_path}")

    plt.close()

    # Summary plot
    print("\nCreating summary plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Stance Learning Summary', fontsize=14)

    for name, traj, color, orig_len in trajectories:
        # Only plot up to original trajectory length (before crash padding)
        traj_actual = traj[:orig_len]
        times = [f['t'] for f in traj_actual]
        stances = [f['stance'] * 100 for f in traj_actual]
        altitudes = [f['z'] * 100 for f in traj_actual]
        velocities = [f['vx'] for f in traj_actual]
        pitches = [np.degrees(f['theta']) for f in traj_actual]

        linestyle = '--' if 'Baseline' in name else '-'
        label_with_duration = f"{name} ({orig_len*0.01:.1f}s)"

        axes[0, 0].plot(times, stances, color=color, linestyle=linestyle, label=label_with_duration)
        axes[0, 1].plot(times, altitudes, color=color, linestyle=linestyle, label=label_with_duration)
        axes[1, 0].plot(times, velocities, color=color, linestyle=linestyle, label=label_with_duration)
        axes[1, 1].plot(times, pitches, color=color, linestyle=linestyle, label=label_with_duration)

    axes[0, 0].set_ylabel('Stance (cm)')
    axes[0, 0].set_title('Stance Position')
    axes[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel('Altitude (cm)')
    axes[0, 1].set_title('Foil Altitude')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title('Forward Velocity')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Pitch (deg)')
    axes[1, 1].set_title('Pitch Angle')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stance_learning_summary.png', dpi=150)
    print("Saved: stance_learning_summary.png")
    plt.close()


if __name__ == '__main__':
    import sys

    # Find most recent checkpoint dir
    checkpoint_dirs = [d for d in os.listdir('checkpoints') if d.startswith('stance_')]
    if not checkpoint_dirs:
        print("No stance checkpoints found!")
        sys.exit(1)

    checkpoint_dir = os.path.join('checkpoints', sorted(checkpoint_dirs)[-1])
    print(f"Using checkpoint dir: {checkpoint_dir}")

    create_overlaid_animation(checkpoint_dir)

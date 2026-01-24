#!/usr/bin/env python3
"""
Create standard 4-panel comparison video for Phase 1 training progression.
Shows: Baseline → Early → Mid → Final training stages.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb_phase1 import PumpFoilEnv4LimbPhase1


def draw_foil_and_rider(ax, z, theta, vx, t, left_leg, right_leg, left_arm, right_arm, waist, color, alpha=1.0, is_crashed=False):
    """Draw full foil assembly and rider with arms and bendable knees."""
    water_y = 0.0
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z
    x_center = 0.0

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

    # Board
    board_local = [(-board_length/2, 0), (board_length/2, 0),
                   (board_length/2, board_thickness), (-board_length/2, board_thickness)]
    board_world = [transform(p) for p in board_local]
    ax.add_patch(Polygon(board_world, facecolor='saddlebrown', edgecolor='black', linewidth=2, alpha=alpha))

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

    # Wings
    wing_local = [(fuse_length * 0.4, -mast_length - 0.02), (fuse_length * 0.4 + 0.06, -mast_length - 0.02),
                  (fuse_length * 0.4 + 0.06, -mast_length + 0.02), (fuse_length * 0.4, -mast_length + 0.02)]
    ax.add_patch(Polygon([transform(p) for p in wing_local], facecolor='silver', edgecolor='black', linewidth=1.5, alpha=alpha))

    stab_local = [(-fuse_length * 0.6, -mast_length - 0.015), (-fuse_length * 0.6 + 0.04, -mast_length - 0.015),
                  (-fuse_length * 0.6 + 0.04, -mast_length + 0.015), (-fuse_length * 0.6, -mast_length + 0.015)]
    ax.add_patch(Polygon([transform(p) for p in stab_local], facecolor='silver', edgecolor='black', linewidth=1, alpha=alpha))

    # === RIDER with proper articulation ===
    shin_len = 0.40
    thigh_len = 0.42
    torso_len = 0.50
    upper_arm_len = 0.30
    forearm_len = 0.25
    head_radius = 0.10
    stance_width = 0.25

    # Feet on board
    l_foot = transform((-stance_width/2, board_thickness))
    r_foot = transform((stance_width/2, board_thickness))

    # Hip height varies with leg extension (more extension = higher hip)
    # left_leg/right_leg are in meters, normalize to -1 to 1 range
    MAX_LEG_EXT = 0.15
    l_leg_norm = left_leg / MAX_LEG_EXT  # -1 to 1
    r_leg_norm = right_leg / MAX_LEG_EXT
    avg_leg_ext = (l_leg_norm + r_leg_norm) / 2

    # Leg height: crouched (ext=-1) to extended (ext=+1)
    leg_height = shin_len + thigh_len * (0.6 + 0.35 * avg_leg_ext)  # More range for visible bending

    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)

    # Torso with waist lean
    shoulder_world = (hip_world[0] + torso_len * np.sin(waist),
                      hip_world[1] + torso_len * np.cos(waist))
    head_world = (shoulder_world[0] + (head_radius + 0.02) * np.sin(waist),
                  shoulder_world[1] + (head_radius + 0.02) * np.cos(waist))

    def solve_knee(foot, hip, shin_l, thigh_l):
        dx, dy = hip[0] - foot[0], hip[1] - foot[1]
        dist = np.clip(np.sqrt(dx*dx + dy*dy), abs(shin_l - thigh_l) + 0.01, shin_l + thigh_l - 0.01)
        cos_foot = np.clip((shin_l**2 + dist**2 - thigh_l**2) / (2 * shin_l * dist), -1, 1)
        angle = np.arctan2(dy, dx) + np.arccos(cos_foot)
        return (foot[0] + shin_l * np.cos(angle), foot[1] + shin_l * np.sin(angle))

    l_knee = solve_knee(l_foot, hip_world, shin_len, thigh_len)
    r_knee = solve_knee(r_foot, hip_world, shin_len, thigh_len)

    for foot, knee, c in [(l_foot, l_knee, 'royalblue'), (r_foot, r_knee, 'crimson')]:
        ax.plot([foot[0], knee[0]], [foot[1], knee[1]], c, linewidth=6, alpha=alpha, solid_capstyle='round')
        ax.plot([knee[0], hip_world[0]], [knee[1], hip_world[1]], c, linewidth=8, alpha=alpha, solid_capstyle='round')
        ax.plot(knee[0], knee[1], 'ko', markersize=6, alpha=alpha)

    # Torso
    ax.plot([hip_world[0], shoulder_world[0]], [hip_world[1], shoulder_world[1]], 'darkgreen', linewidth=10, alpha=alpha, solid_capstyle='round')

    # === ARMS (swing with arm position) ===
    # left_arm/right_arm are in radians (max ±1.5 rad)
    MAX_ARM_SWING = 1.5
    l_arm_norm = left_arm / MAX_ARM_SWING  # -1 to 1
    r_arm_norm = right_arm / MAX_ARM_SWING

    # Shoulders offset from torso center
    shoulder_offset = 0.12
    l_shoulder = (shoulder_world[0] - shoulder_offset * np.cos(waist),
                  shoulder_world[1] - shoulder_offset * np.sin(waist) - 0.05)
    r_shoulder = (shoulder_world[0] + shoulder_offset * np.cos(waist),
                  shoulder_world[1] + shoulder_offset * np.sin(waist) - 0.05)

    def get_arm_points(shoulder, arm_swing, side):
        """Get elbow and wrist positions based on arm swing."""
        # Base arm angle (hanging down is pi/2, forward swing decreases, back increases)
        base_angle = np.pi / 2 + waist  # Align with torso
        swing_angle = base_angle - arm_swing * 0.8  # Swing forward/back

        elbow = (shoulder[0] + side * 0.02 + upper_arm_len * np.sin(swing_angle),
                 shoulder[1] - upper_arm_len * np.cos(swing_angle))

        # Forearm bends slightly
        forearm_angle = swing_angle + 0.3  # Slight bend
        wrist = (elbow[0] + forearm_len * np.sin(forearm_angle) * 0.8,
                 elbow[1] - forearm_len * np.cos(forearm_angle) * 0.8)

        return elbow, wrist

    l_elbow, l_wrist = get_arm_points(l_shoulder, l_arm_norm, -1)
    r_elbow, r_wrist = get_arm_points(r_shoulder, r_arm_norm, 1)

    # Draw arms
    for shoulder, elbow, wrist, c in [(l_shoulder, l_elbow, l_wrist, 'royalblue'),
                                       (r_shoulder, r_elbow, r_wrist, 'crimson')]:
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], c, linewidth=5, alpha=alpha*0.9, solid_capstyle='round')
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], c, linewidth=4, alpha=alpha*0.9, solid_capstyle='round')
        ax.plot(elbow[0], elbow[1], 'ko', markersize=4, alpha=alpha)

    # Head
    ax.add_patch(Circle(head_world, head_radius, facecolor='peachpuff', edgecolor='black', linewidth=2, alpha=alpha))
    ax.plot(head_world[0] + 0.03, head_world[1] + 0.02, 'ko', markersize=3, alpha=alpha)

    if is_crashed:
        cx, cy = x_center, board_center_y + 0.5
        ax.plot([cx - 0.3, cx + 0.3], [cy - 0.3, cy + 0.3], 'r-', linewidth=4, alpha=0.8)
        ax.plot([cx - 0.3, cx + 0.3], [cy + 0.3, cy - 0.3], 'r-', linewidth=4, alpha=0.8)


def run_episode(model_path=None, max_steps=1000):
    """Run episode and collect data."""
    env = PumpFoilEnv4LimbPhase1()
    obs, _ = env.reset(seed=42)

    if model_path:
        model = PPO.load(model_path)
    else:
        model = None

    frames = []
    done = False
    step = 0

    while not done and step < max_steps:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(5)

        frames.append({
            't': step * env.dt,
            'z': env.state.z,
            'vx': env.state.vx,
            'theta': env.state.theta,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
            'waist': env.waist_pos,
        })

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    return frames, info.get('termination_reason', 'max_steps')


def create_progression_video(output_path='phase1_progression.mp4'):
    """Create 4-panel video showing training progression."""

    checkpoints = [
        ('Baseline\n(no training)', None),
        ('100k steps', 'checkpoints/phase1/phase1_100000_steps.zip'),
        ('500k steps', 'checkpoints/phase1/phase1_500000_steps.zip'),
        ('1M steps\n(final)', 'checkpoints/phase1/phase1_final.zip'),
    ]

    colors = ['gray', '#e74c3c', '#f39c12', '#27ae60']

    # Run episodes
    scenarios = []
    for (name, path), color in zip(checkpoints, colors):
        print(f"Running: {name}...")
        frames, term = run_episode(path, max_steps=1500)
        scenarios.append((name, frames, color, term, len(frames)))
        print(f"  Duration: {len(frames)*0.01:.2f}s, term: {term}")

    # Pad to same length
    max_frames = max(len(s[1]) for s in scenarios)
    for name, frames, color, term, orig_len in scenarios:
        last = frames[-1].copy()
        while len(frames) < max_frames:
            frames.append(last)
        frames[0]['orig_len'] = orig_len

    # Create animation
    print(f"\nCreating video with {max_frames} frames...")
    fig = plt.figure(figsize=(20, 10))

    # Use GridSpec for flexible layout: 4 foil panels on top, 5 time series on bottom
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.2, 1], hspace=0.25, wspace=0.3)

    # Top row: 4 foil panels (spanning columns to center them)
    axes_foil = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
    ]
    # Hide the 5th top cell
    ax_info = fig.add_subplot(gs[0, 4])
    ax_info.axis('off')

    # Bottom row: 5 time series plots
    ax_z = fig.add_subplot(gs[1, 0])
    ax_vx = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_arm = fig.add_subplot(gs[1, 3])
    ax_waist = fig.add_subplot(gs[1, 4])

    frame_skip = 3
    frames_to_render = list(range(0, max_frames, frame_skip))

    def animate(frame_num):
        i = frames_to_render[frame_num]

        for idx, (name, frames, color, term, orig_len) in enumerate(scenarios):
            ax = axes_foil[idx]
            ax.clear()

            orig_len = frames[0].get('orig_len', len(frames))
            is_crashed = i >= orig_len
            frame = frames[min(i, len(frames)-1)]

            ax.axhspan(-1.5, 0.0, alpha=0.3, color='dodgerblue')
            ax.axhline(y=0.0, color='blue', linewidth=2, alpha=0.5)

            if not is_crashed:
                np.random.seed(int(frame['t'] * 100) % 1000)
                for j in range(6):
                    streak_y = np.random.uniform(-0.7, -0.1)
                    streak_x = np.random.uniform(-1.5, 1.5)
                    ax.plot([streak_x, streak_x - 0.1], [streak_y, streak_y], 'white', linewidth=1.5, alpha=0.4)

            draw_foil_and_rider(ax, frame['z'], frame['theta'], frame['vx'], frame['t'],
                               frame['left_leg'], frame['right_leg'],
                               frame['left_arm'], frame['right_arm'], frame['waist'],
                               color, alpha=0.9, is_crashed=is_crashed)

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 2.0)
            ax.set_aspect('equal')
            ax.set_facecolor('lightcyan')

            duration = orig_len * 0.01
            status = f"{duration:.1f}s" if not is_crashed else f"CRASHED @ {duration:.1f}s"
            title_color = color if not is_crashed else 'red'
            ax.set_title(f"{name}\n{status}", fontsize=9, color=title_color)

        t_now = scenarios[0][1][min(i, len(scenarios[0][1])-1)]['t']

        for ax, key, ylabel, limits in [
            (ax_z, 'z', 'Altitude (cm)', (-60, 30)),
            (ax_vx, 'vx', 'Velocity (m/s)', (0, 6)),
            (ax_leg, 'left_leg', 'Leg pos (cm)', (-20, 20)),
            (ax_arm, 'left_arm', 'Arm pos (deg)', (-100, 100)),
            (ax_waist, 'waist', 'Waist (deg)', (-25, 25)),
        ]:
            ax.clear()
            for name, frames, color, term, orig_len in scenarios:
                orig_len = frames[0].get('orig_len', len(frames))
                plot_len = min(i + 1, orig_len)
                times = [f['t'] for f in frames[:plot_len]]
                if key == 'z':
                    values = [f[key] * 100 for f in frames[:plot_len]]
                elif key == 'waist':
                    values = [np.degrees(f[key]) for f in frames[:plot_len]]
                elif key == 'left_leg':
                    values = [(f['left_leg'] + f['right_leg']) / 2 * 100 for f in frames[:plot_len]]
                elif key == 'left_arm':
                    values = [np.degrees((f['left_arm'] + f['right_arm']) / 2) for f in frames[:plot_len]]
                else:
                    values = [f[key] for f in frames[:plot_len]]
                linestyle = '--' if 'Baseline' in name else '-'
                ax.plot(times, values, color=color, linestyle=linestyle, linewidth=1.5, label=name.split('\n')[0])

            ax.axvline(x=t_now, color='black', linestyle='-', alpha=0.2)
            ax.set_xlim(0, max_frames * 0.01)
            ax.set_ylim(limits)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=8)
            if ax == ax_z:
                ax.legend(loc='upper right', fontsize=7)

        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render), interval=33, blit=False)

    writer = FFMpegWriter(fps=30, bitrate=3000)
    anim.save(output_path, writer=writer)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    create_progression_video()

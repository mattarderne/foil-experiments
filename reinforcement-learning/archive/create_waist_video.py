#!/usr/bin/env python3
"""
Create 4-panel video of trained 5-action waist model.
Standard format: foil viz | altitude | velocity | controls
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_episode(model_path):
    """Run episode and collect all data."""
    model = PPO.load(model_path)
    env = PumpFoilEnv4Limb()

    obs, _ = env.reset()
    done = False

    data = {
        't': [], 'z': [], 'vx': [], 'vz': [], 'theta': [],
        'left_leg': [], 'right_leg': [], 'left_arm': [], 'right_arm': [],
        'waist': [], 'thrust': [], 'positions': [], 'actions': [],
    }

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        positions = env.get_body_positions()

        data['t'].append(env.step_count * env.dt)
        data['z'].append(env.state.z)
        data['vx'].append(env.state.vx)
        data['vz'].append(env.state.vz)
        data['theta'].append(env.state.theta)
        data['left_leg'].append(env.left_leg_pos * 100)  # cm
        data['right_leg'].append(env.right_leg_pos * 100)
        data['left_arm'].append(np.degrees(env.left_arm_pos))
        data['right_arm'].append(np.degrees(env.right_arm_pos))
        data['waist'].append(np.degrees(env.waist_pos))
        data['positions'].append(positions)
        data['actions'].append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        data['thrust'].append(info.get('thrust', 0))
        done = terminated or truncated

    data['termination'] = info.get('termination_reason', 'unknown')

    for k in data:
        if k not in ['positions', 'termination', 'actions']:
            data[k] = np.array(data[k])
        elif k == 'actions':
            data[k] = np.array(data[k])

    return data


def draw_foil_and_rider(ax, positions, z, theta, z_min, z_max):
    """Draw the foil board with rider."""
    ax.clear()

    # Water
    ax.axhline(y=0.2, color='deepskyblue', alpha=0.8, linewidth=3)
    ax.axhspan(-1.5, 0.2, alpha=0.15, color='deepskyblue')

    # Mast (from board down to foil)
    board_y = positions['left_ankle'][1]
    mast_bottom = board_y - 0.8  # approximate mast length
    ax.plot([0, 0], [board_y - 0.05, mast_bottom], 'gray', linewidth=4, alpha=0.8)

    # Foil wings
    foil_span = 0.7
    ax.plot([-foil_span/2, foil_span/2], [mast_bottom, mast_bottom],
            'darkblue', linewidth=8, solid_capstyle='round')

    # Board
    board_len = 0.5
    c, s = np.cos(theta), np.sin(theta)
    board_pts = np.array([
        [-board_len/2, 0], [board_len/2, 0],
        [board_len/2 + 0.05, -0.03], [-board_len/2 - 0.05, -0.03]
    ])
    rotated = np.array([[c*p[0] - s*p[1], s*p[0] + c*p[1] + board_y] for p in board_pts])
    board = Polygon(rotated, facecolor='saddlebrown', edgecolor='black', linewidth=2)
    ax.add_patch(board)

    # Legs
    for side, color in [('left', 'royalblue'), ('right', 'crimson')]:
        ankle = positions[f'{side}_ankle']
        knee = positions[f'{side}_knee']
        hip = positions[f'{side}_hip']

        ax.plot([ankle[0], knee[0]], [ankle[1], knee[1]], color, linewidth=6, solid_capstyle='round')
        ax.plot([knee[0], hip[0]], [knee[1], hip[1]], color, linewidth=8, solid_capstyle='round')
        ax.plot(knee[0], knee[1], 'ko', markersize=6)

    # Torso
    ax.plot([positions['torso_base'][0], positions['torso_top'][0]],
            [positions['torso_base'][1], positions['torso_top'][1]],
            'darkgreen', linewidth=14, solid_capstyle='round')

    # Arms
    for side, color in [('left', 'royalblue'), ('right', 'crimson')]:
        shoulder = positions[f'{side}_shoulder']
        elbow = positions[f'{side}_elbow']
        wrist = positions[f'{side}_wrist']

        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], color, linewidth=5, alpha=0.9)
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color, linewidth=4, alpha=0.9)
        ax.plot(elbow[0], elbow[1], 'ko', markersize=4)

    # Head
    head = positions['head']
    ax.add_patch(Circle(head, 0.10, facecolor='peachpuff', edgecolor='black', linewidth=2))
    ax.plot(head[0] + 0.04, head[1] + 0.02, 'ko', markersize=4)  # eye

    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(z_min - 0.3, z_max + 0.8)
    ax.set_aspect('equal')
    ax.set_ylabel('Z (m)')
    ax.set_title('Foil & Rider')
    ax.grid(True, alpha=0.3)


def create_video(data, output_path='waist_model.mp4', fps=30):
    """Create 4-panel video."""
    n_frames = len(data['t'])

    # Subsample for reasonable video length
    step = max(1, n_frames // (fps * 10))  # max 10 seconds
    indices = list(range(0, n_frames, step))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_foil = axes[0, 0]
    ax_alt = axes[0, 1]
    ax_vel = axes[1, 0]
    ax_ctrl = axes[1, 1]

    z_min, z_max = np.min(data['z']) - 0.1, np.max(data['z']) + 0.1

    def animate(frame_idx):
        i = indices[frame_idx]

        # Panel 1: Foil visualization
        draw_foil_and_rider(ax_foil, data['positions'][i], data['z'][i],
                           data['theta'][i], z_min, z_max)

        # Panel 2: Altitude
        ax_alt.clear()
        t_hist = data['t'][:i+1]
        ax_alt.plot(t_hist, data['z'][:i+1] * 100, 'b-', linewidth=2)
        ax_alt.axhline(15, color='g', linestyle='--', alpha=0.5, label='Target')
        ax_alt.axhline(20, color='r', linestyle=':', alpha=0.3, label='Ceiling')
        ax_alt.axhline(-50, color='r', linestyle=':', alpha=0.3, label='Floor')
        ax_alt.set_xlim(0, data['t'][-1])
        ax_alt.set_ylim(-60, 30)
        ax_alt.set_ylabel('Altitude (cm)')
        ax_alt.set_title(f't={data["t"][i]:.2f}s | z={data["z"][i]*100:.1f}cm')
        ax_alt.legend(loc='upper right', fontsize=8)
        ax_alt.grid(True, alpha=0.3)

        # Panel 3: Velocity
        ax_vel.clear()
        ax_vel.plot(t_hist, data['vx'][:i+1], 'g-', linewidth=2, label='vx')
        ax_vel.plot(t_hist, data['vz'][:i+1], 'b-', linewidth=1.5, alpha=0.7, label='vz')
        ax_vel.axhline(4.5, color='g', linestyle='--', alpha=0.5, label='Target vx')
        ax_vel.set_xlim(0, data['t'][-1])
        ax_vel.set_ylim(-2, 6)
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.set_title(f'vx={data["vx"][i]:.2f} m/s | thrust={data["thrust"][i]:.0f}N')
        ax_vel.legend(loc='upper right', fontsize=8)
        ax_vel.grid(True, alpha=0.3)

        # Panel 4: Controls (legs, waist, arms)
        ax_ctrl.clear()
        avg_leg = (data['left_leg'][:i+1] + data['right_leg'][:i+1]) / 2
        ax_ctrl.plot(t_hist, avg_leg, 'b-', linewidth=2, label='Legs (cm)')
        ax_ctrl.plot(t_hist, data['waist'][:i+1], 'purple', linewidth=2, label='Waist (°)')
        avg_arm = (data['left_arm'][:i+1] + data['right_arm'][:i+1]) / 2
        ax_ctrl.plot(t_hist, avg_arm, 'orange', linewidth=1.5, alpha=0.7, label='Arms (°)')

        ax_ctrl.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax_ctrl.set_xlim(0, data['t'][-1])
        ax_ctrl.set_ylim(-20, 20)
        ax_ctrl.set_xlabel('Time (s)')
        ax_ctrl.set_ylabel('Position')
        ax_ctrl.set_title('Controls: Legs, Waist, Arms')
        ax_ctrl.legend(loc='upper right', fontsize=8)
        ax_ctrl.grid(True, alpha=0.3)

        return []

    print(f"Creating video with {len(indices)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(indices), interval=1000/fps, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading trained 5-action waist model...")
    data = run_episode("checkpoints/model_4limb_modal.zip")

    print(f"Episode: {len(data['t'])} frames, {data['t'][-1]:.2f}s")
    print(f"Termination: {data['termination']}")
    print(f"Final vx: {data['vx'][-1]:.2f} m/s")
    print(f"Velocity loss: {data['vx'][0] - data['vx'][-1]:.2f} m/s")
    print(f"Mean thrust: {np.mean(data['thrust']):.1f} N")

    create_video(data, 'waist_model_video.mp4', fps=30)


if __name__ == "__main__":
    main()

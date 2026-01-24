"""
Create animation of 4-limb pump foil agent.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb
import glob
import os


def find_latest_model():
    """Find the most recent 4-limb model."""
    checkpoints = glob.glob("./checkpoints/pump_foil_4limb_*/final_model.zip")
    if not checkpoints:
        checkpoints = glob.glob("./checkpoints/pump_foil_4limb_*/best_model.zip")
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None


def run_episode(model_path=None):
    """Run an episode and collect data for animation."""
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No model found!")
            return None

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    env = PumpFoilEnv4Limb()

    obs, _ = env.reset()
    done = False

    frames = []
    t = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Get body positions for this frame
        positions = env.get_body_positions()

        frame_data = {
            't': t,
            'z': env.state.z,
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,
            'positions': positions,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
        }
        frames.append(frame_data)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += env.dt

    print(f"Episode: {len(frames)} frames, {t:.2f}s")
    print(f"Termination: {info.get('termination_reason', 'unknown')}")
    return frames


def create_animation(frames, output_path='4limb_pumping.gif', fps=30):
    """Create animation from episode frames."""
    if frames is None or len(frames) == 0:
        print("No frames to animate!")
        return

    # Subsample for reasonable gif size
    step = max(1, len(frames) // (fps * 5))  # ~5 seconds max
    frames = frames[::step]

    fig, (ax_body, ax_side) = plt.subplots(1, 2, figsize=(14, 6))

    # Find bounds
    all_z = [f['z'] for f in frames]
    z_min, z_max = min(all_z) - 0.2, max(all_z) + 0.5

    def draw_body(ax, positions, z_offset, title_extra=""):
        ax.clear()

        # Water surface
        ax.axhline(y=0.2, color='blue', alpha=0.3, linewidth=2, label='Water surface')
        ax.axhspan(-1, 0.2, alpha=0.1, color='blue')

        # Board
        board_y = positions['left_ankle'][1]
        ax.plot([-0.3, 0.3], [board_y, board_y], 'saddlebrown', linewidth=10, solid_capstyle='round')

        # Legs
        for side, color in [('left', 'royalblue'), ('right', 'crimson')]:
            ankle = positions[f'{side}_ankle']
            knee = positions[f'{side}_knee']
            hip = positions[f'{side}_hip']

            # Shin
            ax.plot([ankle[0], knee[0]], [ankle[1], knee[1]], color, linewidth=6, solid_capstyle='round')
            # Thigh
            ax.plot([knee[0], hip[0]], [knee[1], hip[1]], color, linewidth=8, solid_capstyle='round')
            # Knee joint
            ax.plot(knee[0], knee[1], 'ko', markersize=8)

        # Torso
        ax.plot([positions['torso_base'][0], positions['torso_top'][0]],
                [positions['torso_base'][1], positions['torso_top'][1]],
                'darkgreen', linewidth=12, solid_capstyle='round')

        # Arms
        for side, color in [('left', 'royalblue'), ('right', 'crimson')]:
            shoulder = positions[f'{side}_shoulder']
            elbow = positions[f'{side}_elbow']
            wrist = positions[f'{side}_wrist']

            ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], color, linewidth=5, alpha=0.8)
            ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color, linewidth=4, alpha=0.8)
            ax.plot(elbow[0], elbow[1], 'ko', markersize=5)

        # Head
        head = positions['head']
        ax.add_patch(Circle(head, 0.10, facecolor='peachpuff', edgecolor='black', linewidth=2))

        # Eyes (to show direction)
        ax.plot(head[0] + 0.03, head[1] + 0.02, 'ko', markersize=3)

        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'4-Limb Pump Foil Agent{title_extra}')

    def draw_metrics(ax, frame, frame_idx):
        ax.clear()

        t = frame['t']
        z = frame['z']
        vx = frame['vx']
        vz = frame['vz']
        theta = np.degrees(frame['theta'])

        # Time series up to current frame
        times = [f['t'] for f in frames[:frame_idx+1]]
        z_hist = [f['z'] for f in frames[:frame_idx+1]]
        vx_hist = [f['vx'] for f in frames[:frame_idx+1]]
        vz_hist = [f['vz'] for f in frames[:frame_idx+1]]

        ax2 = ax.twinx()

        ax.plot(times, z_hist, 'b-', linewidth=2, label='Altitude (m)')
        ax.axhline(y=0.15, color='b', linestyle='--', alpha=0.5, label='Target')
        ax.set_ylabel('Altitude (m)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(-0.6, 0.3)

        ax2.plot(times, vx_hist, 'g-', linewidth=2, label='Vx (m/s)')
        ax2.plot(times, vz_hist, 'r-', linewidth=1, alpha=0.7, label='Vz (m/s)')
        ax2.set_ylabel('Velocity (m/s)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(-2, 6)

        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, frames[-1]['t'])
        ax.set_title(f't={t:.2f}s | z={z:.2f}m | vx={vx:.1f}m/s | pitch={theta:.1f}°')
        ax.grid(True, alpha=0.3)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    def animate(frame_idx):
        frame = frames[frame_idx]
        draw_body(ax_body, frame['positions'], frame['z'])
        draw_metrics(ax_side, frame, frame_idx)
        return []

    print(f"Creating animation with {len(frames)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    frames = run_episode(model_path)
    create_animation(frames, '4limb_pumping.gif', fps=20)

#!/usr/bin/env python
"""
Animate the learning progression of the pump foil agent.

Shows body mechanics at different training iterations (10, 50, 100, 200, etc.)
to visualize how the agent learns to pump.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foil_env import PumpFoilEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


def train_with_checkpoints(checkpoints=[10, 50, 100, 200, 500, 1000], total_steps=50000):
    """Train and save models at specific iteration checkpoints."""
    print("Training with checkpoints...")

    # Create environment
    def make_env():
        return PumpFoilEnv()

    env = DummyVecEnv([make_env for _ in range(4)])

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
    )

    # Save initial (untrained) model
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/model_0")
    print("Saved checkpoint: 0 iterations")

    # Train in increments
    current_steps = 0
    for checkpoint in checkpoints:
        steps_to_train = (checkpoint * 1000) - current_steps
        if steps_to_train > 0:
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
            current_steps = checkpoint * 1000
            model.save(f"checkpoints/model_{checkpoint}k")
            print(f"Saved checkpoint: {checkpoint}k steps")

    env.close()
    return checkpoints


def record_episode(model_path, max_steps=300):
    """Record an episode with the given model."""
    env = PumpFoilEnv()

    if model_path == "random":
        model = None
    else:
        model = PPO.load(model_path)

    obs, info = env.reset(seed=42)

    trajectory = []

    for step in range(max_steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            'step': step,
            'z': info['z'],
            'theta': np.radians(info['theta_deg']),
            'vx': info['vx'],
            'leg_action': action[0],
            'pitch_action': action[1],
        })

        if terminated or truncated:
            break

    env.close()
    return trajectory


def draw_rider(ax, z, theta, leg_action, pitch_action, alpha=1.0):
    """Draw the rider and foil with clear body mechanics."""
    ax.clear()

    # Set up plot
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.6, 0.8)
    ax.set_aspect('equal')
    ax.set_facecolor('#e6f3ff')

    # Water surface
    water_x = np.linspace(-1, 1, 100)
    water_y = np.sin(water_x * 10) * 0.01
    ax.fill_between(water_x, water_y, 0.8, color='#87CEEB', alpha=0.3)
    ax.plot(water_x, water_y, 'b-', linewidth=2, alpha=0.7)

    # Target altitude line
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.4, linewidth=1)

    # Transformation
    cx, cz = 0, z
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def transform(local_x, local_y):
        return (cx + local_x * cos_t - local_y * sin_t,
                cz + local_x * sin_t + local_y * cos_t)

    # BOARD - larger and more visible
    board_len, board_h = 0.5, 0.04
    board_pts = [
        transform(-board_len/2, board_h),
        transform(board_len/2, board_h),
        transform(board_len/2, 0),
        transform(-board_len/2, 0),
    ]
    ax.add_patch(Polygon(board_pts, closed=True, facecolor='#8B4513',
                         edgecolor='#5D3A1A', linewidth=2, alpha=alpha))

    # MAST
    mast_top = transform(0.05, 0)
    mast_bot = transform(0.05, -0.2)
    ax.plot([mast_top[0], mast_bot[0]], [mast_top[1], mast_bot[1]],
            color='#333333', linewidth=4, alpha=alpha, solid_capstyle='round')

    # FRONT WING
    wing_cx, wing_cy = transform(0.05, -0.2)
    wing_w, wing_h = 0.25, 0.025
    wing_pts = [
        (wing_cx - wing_w/2 * cos_t, wing_cy - wing_w/2 * sin_t),
        (wing_cx + wing_w/2 * cos_t, wing_cy + wing_w/2 * sin_t),
        (wing_cx + wing_w/2 * cos_t - wing_h * sin_t, wing_cy + wing_w/2 * sin_t + wing_h * cos_t),
        (wing_cx - wing_w/2 * cos_t - wing_h * sin_t, wing_cy - wing_w/2 * sin_t + wing_h * cos_t),
    ]
    ax.add_patch(Polygon(wing_pts, closed=True, facecolor='#1a5276',
                         edgecolor='#0d3b5a', linewidth=2, alpha=alpha))

    # STABILIZER
    stab_cx, stab_cy = transform(-0.25, -0.18)
    stab_w, stab_h = 0.1, 0.015
    stab_pts = [
        (stab_cx - stab_w/2 * cos_t, stab_cy - stab_w/2 * sin_t),
        (stab_cx + stab_w/2 * cos_t, stab_cy + stab_w/2 * sin_t),
        (stab_cx + stab_w/2 * cos_t - stab_h * sin_t, stab_cy + stab_w/2 * sin_t + stab_h * cos_t),
        (stab_cx - stab_w/2 * cos_t - stab_h * sin_t, stab_cy - stab_w/2 * sin_t + stab_h * cos_t),
    ]
    ax.add_patch(Polygon(stab_pts, closed=True, facecolor='#666666',
                         edgecolor='#444444', linewidth=1, alpha=alpha))

    # === RIDER ===
    # Leg extension based on action: -1 = compressed, +1 = extended
    leg_extension = 0.08 + leg_action * 0.06  # Range: 0.02 to 0.14

    # Feet position on board
    feet_x, feet_y = transform(0, board_h)

    # Knee position (bends based on leg action)
    knee_bend = (1 - leg_action) * 0.04  # More bend when compressed
    knee_x = feet_x - 0.02 * cos_t - knee_bend * sin_t
    knee_y = feet_y + leg_extension * 0.5 - 0.02 * sin_t + knee_bend * cos_t

    # Hip position
    hip_x = feet_x - 0.03 * cos_t
    hip_y = feet_y + leg_extension

    # Torso lean based on pitch action
    torso_lean = pitch_action * 0.08

    # Shoulder position
    shoulder_x = hip_x + torso_lean * cos_t
    shoulder_y = hip_y + 0.12

    # Head position
    head_x = shoulder_x + torso_lean * 0.3 * cos_t
    head_y = shoulder_y + 0.06

    # Draw legs (thick lines)
    leg_color = '#2c3e50'
    ax.plot([feet_x, knee_x], [feet_y, knee_y], color=leg_color,
            linewidth=6, alpha=alpha, solid_capstyle='round')
    ax.plot([knee_x, hip_x], [knee_y, hip_y], color=leg_color,
            linewidth=6, alpha=alpha, solid_capstyle='round')

    # Draw torso
    ax.plot([hip_x, shoulder_x], [hip_y, shoulder_y], color=leg_color,
            linewidth=8, alpha=alpha, solid_capstyle='round')

    # Draw arms (extended for balance)
    arm_len = 0.1
    # Left arm
    ax.plot([shoulder_x, shoulder_x - arm_len * cos_t + 0.02 * sin_t],
            [shoulder_y, shoulder_y - arm_len * sin_t - 0.02 * cos_t],
            color=leg_color, linewidth=4, alpha=alpha, solid_capstyle='round')
    # Right arm
    ax.plot([shoulder_x, shoulder_x + arm_len * cos_t + 0.02 * sin_t],
            [shoulder_y, shoulder_y + arm_len * sin_t - 0.02 * cos_t],
            color=leg_color, linewidth=4, alpha=alpha, solid_capstyle='round')

    # Draw head
    ax.add_patch(Circle((head_x, head_y), 0.04, facecolor='#f5cba7',
                         edgecolor='#d4a574', linewidth=2, alpha=alpha))

    # Draw joints
    joint_color = '#1a252f'
    for jx, jy in [(feet_x, feet_y), (knee_x, knee_y), (hip_x, hip_y), (shoulder_x, shoulder_y)]:
        ax.add_patch(Circle((jx, jy), 0.012, facecolor=joint_color, alpha=alpha))

    ax.axis('off')


def create_learning_animation(checkpoints=[0, 10, 50, 100, 200, 500, 1000]):
    """Create animation showing learning progression."""
    print("\nCreating learning progression animation...")

    # Record trajectories at each checkpoint
    trajectories = {}
    labels = {}

    for cp in checkpoints:
        if cp == 0:
            path = "checkpoints/model_0"
            label = "Untrained (0 steps)"
        else:
            path = f"checkpoints/model_{cp}k"
            label = f"{cp}k steps"

        if os.path.exists(path + ".zip"):
            print(f"Recording episode for {label}...")
            trajectories[cp] = record_episode(path)
            labels[cp] = label
            print(f"  -> {len(trajectories[cp])} steps, final vx={trajectories[cp][-1]['vx']:.2f}")
        else:
            print(f"Checkpoint not found: {path}")

    if not trajectories:
        print("No checkpoints found. Training first...")
        train_with_checkpoints(checkpoints[1:])  # Skip 0
        return create_learning_animation(checkpoints)

    # Create figure with subplots for each checkpoint
    n_checkpoints = len(trajectories)
    fig, axes = plt.subplots(2, (n_checkpoints + 1) // 2, figsize=(4 * ((n_checkpoints + 1) // 2), 8))
    axes = axes.flatten()

    # Hide unused axes
    for i in range(n_checkpoints, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Learning to Pump: Training Progression', fontsize=16, fontweight='bold')

    # Animation function
    def update(frame):
        for idx, (cp, traj) in enumerate(trajectories.items()):
            ax = axes[idx]

            # Get frame data (loop if needed)
            frame_idx = frame % len(traj)
            data = traj[frame_idx]

            draw_rider(ax, data['z'], data['theta'], data['leg_action'], data['pitch_action'])

            # Add title with stats
            flight_time = len(traj) * 0.01
            ax.set_title(f"{labels[cp]}\n{flight_time:.1f}s flight", fontsize=11)

        return axes

    # Create animation
    max_frames = max(len(t) for t in trajectories.values())
    ani = FuncAnimation(fig, update, frames=min(max_frames, 300), interval=33, blit=False)

    # Save as GIF
    print("Saving animation...")
    ani.save('learning_progression.gif', writer=PillowWriter(fps=30))
    print("Saved: learning_progression.gif")

    plt.close()


def create_comparison_animation():
    """Create side-by-side comparison of untrained vs trained."""
    print("\nCreating comparison animation...")

    # Check for models
    if not os.path.exists("checkpoints/model_0.zip"):
        print("No checkpoints found. Training first...")
        train_with_checkpoints([10, 50, 100, 200, 500, 1000])

    # Find best trained model
    trained_path = None
    for cp in [1000, 500, 200, 100, 50, 10]:
        path = f"checkpoints/model_{cp}k"
        if os.path.exists(path + ".zip"):
            trained_path = path
            trained_label = f"{cp}k steps"
            break

    if not trained_path:
        print("No trained model found!")
        return

    # Record episodes
    print("Recording untrained episode...")
    untrained_traj = record_episode("checkpoints/model_0")
    print(f"  -> {len(untrained_traj)} steps")

    print(f"Recording trained episode ({trained_label})...")
    trained_traj = record_episode(trained_path)
    print(f"  -> {len(trained_traj)} steps")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Learning to Pump: Before vs After Training', fontsize=16, fontweight='bold')

    def update(frame):
        # Untrained (left)
        if frame < len(untrained_traj):
            data = untrained_traj[frame]
            draw_rider(ax1, data['z'], data['theta'], data['leg_action'], data['pitch_action'])
            ax1.set_title(f"Untrained\nStep {frame}, {frame*0.01:.2f}s", fontsize=12)
        else:
            ax1.clear()
            ax1.set_xlim(-0.8, 0.8)
            ax1.set_ylim(-0.6, 0.8)
            ax1.text(0, 0.1, "CRASHED!", fontsize=24, ha='center', va='center',
                    color='red', fontweight='bold')
            ax1.text(0, -0.1, f"Lasted {len(untrained_traj)*0.01:.2f}s", fontsize=14,
                    ha='center', va='center')
            ax1.axis('off')

        # Trained (right)
        if frame < len(trained_traj):
            data = trained_traj[frame]
            draw_rider(ax2, data['z'], data['theta'], data['leg_action'], data['pitch_action'])
            ax2.set_title(f"Trained ({trained_label})\nStep {frame}, {frame*0.01:.2f}s", fontsize=12)
        else:
            ax2.clear()
            ax2.set_xlim(-0.8, 0.8)
            ax2.set_ylim(-0.6, 0.8)
            ax2.text(0, 0.1, "Still flying!", fontsize=20, ha='center', va='center',
                    color='green', fontweight='bold')
            ax2.text(0, -0.1, f">{len(trained_traj)*0.01:.1f}s", fontsize=14,
                    ha='center', va='center')
            ax2.axis('off')

        return ax1, ax2

    # Create animation
    max_frames = max(len(untrained_traj), len(trained_traj))
    ani = FuncAnimation(fig, update, frames=min(max_frames + 30, 350), interval=33, blit=False)

    # Save
    print("Saving comparison animation...")
    ani.save('before_after.gif', writer=PillowWriter(fps=30))
    print("Saved: before_after.gif")

    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train models at checkpoints')
    parser.add_argument('--compare', action='store_true', help='Create before/after comparison')
    args = parser.parse_args()

    if args.train:
        train_with_checkpoints([10, 50, 100, 200, 500, 1000])

    if args.compare:
        create_comparison_animation()
    else:
        create_learning_animation()

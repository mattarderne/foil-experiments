#!/usr/bin/env python3
"""
Create side-by-side comparison video: Baseline vs Stylized motion.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '.')

from stable_baselines3 import PPO
from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
from foil_env.config import LimbConfig
from foil_env.foil_visualizer import draw_foil_and_rider, draw_water


def collect_episode(checkpoint_path: str, max_steps: int = 3000, seed: int = 0):
    """Collect frames from an episode."""
    foil_dict = {
        'S': 0.10, 'S_stab': 0.02, 'stab_angle': -4.0, 'AR': 8,
        'mast_length': 0.7, 'riding_depth': 0.2,
        'Cd0': 0.013, 'Cd_mast': 0.12, 'pump_thrust_efficiency': 0.20,
    }
    limbs = LimbConfig(arm_start_position=np.pi/2, arm_centering_spring=2.0)

    class Env(PumpFoilEnvCurriculum):
        MAX_STEPS = 6000
        MAX_ENERGY = 50000
        TARGET_ALTITUDE = -0.15
        MIN_VELOCITY = 1.5

    env = Env(config=foil_dict, limb_config=limbs, curriculum_phase=2)
    model = PPO.load(checkpoint_path)

    obs, _ = env.reset(seed=seed)

    frames = []
    actions = []
    leg_vels = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect data
        frames.append({
            'x': env.state.x,
            'z': env.state.z,
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,
            'left_leg_pos': env.left_leg_pos,
            'right_leg_pos': env.right_leg_pos,
            'left_leg_vel': env.left_leg_vel,
            'right_leg_vel': env.right_leg_vel,
            'body_state': env.body.state,
        })
        actions.append(action.copy())
        leg_vels.append((env.left_leg_vel + env.right_leg_vel) / 2)

        if terminated or truncated:
            break

    return frames, actions, leg_vels


def create_comparison_video(
    baseline_path: str,
    stylized_path: str,
    output_path: str = "style_comparison.mp4",
    duration: float = 20.0,  # seconds
    fps: int = 30,
):
    """Create side-by-side comparison video."""

    max_steps = int(duration * 100)  # 100 Hz sim
    frame_skip = int(100 / fps)  # Downsample for video

    print(f"Collecting baseline episode...")
    baseline_frames, baseline_actions, baseline_vels = collect_episode(baseline_path, max_steps)

    print(f"Collecting stylized episode...")
    stylized_frames, stylized_actions, stylized_vels = collect_episode(stylized_path, max_steps)

    # Use minimum length
    n_frames = min(len(baseline_frames), len(stylized_frames))
    print(f"Creating video with {n_frames} frames ({n_frames/100:.1f}s)...")

    # Compute jitter metrics for display
    baseline_vel_jitter = np.mean(np.diff(baseline_vels[:n_frames])**2)
    stylized_vel_jitter = np.mean(np.diff(stylized_vels[:n_frames])**2)

    if len(baseline_actions) > 1:
        baseline_act_jitter = np.mean(np.sum(np.diff(baseline_actions[:n_frames], axis=0)**2, axis=1))
    else:
        baseline_act_jitter = 0
    if len(stylized_actions) > 1:
        stylized_act_jitter = np.mean(np.sum(np.diff(stylized_actions[:n_frames], axis=0)**2, axis=1))
    else:
        stylized_act_jitter = 0

    # Create figure - taller to fit everything
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, height_ratios=[3, 1, 1], hspace=0.25, wspace=0.15)

    # Top row: Side-by-side foil visualizations
    ax_base = fig.add_subplot(gs[0, 0])
    ax_style = fig.add_subplot(gs[0, 1])

    # Middle row: Leg velocity comparison
    ax_vel = fig.add_subplot(gs[1, :])

    # Bottom row: Action comparison
    ax_act = fig.add_subplot(gs[2, :])

    def init():
        return []

    def animate(frame_idx):
        # Clear axes
        ax_base.clear()
        ax_style.clear()
        ax_vel.clear()
        ax_act.clear()

        sim_idx = frame_idx * frame_skip
        if sim_idx >= n_frames:
            sim_idx = n_frames - 1

        t = sim_idx * 0.01

        # Draw baseline - wider vertical range to show full rider
        bf = baseline_frames[sim_idx]
        ax_base.set_xlim(-2.5, 2.5)
        ax_base.set_ylim(-1.2, 2.5)  # More headroom for rider
        draw_foil_and_rider(
            ax_base, bf['z'], bf['theta'],
            left_leg=bf['left_leg_pos'],
            right_leg=bf['right_leg_pos'],
            left_arm=getattr(bf['body_state'], 'left_arm', 0.0),
            right_arm=getattr(bf['body_state'], 'right_arm', 0.0),
            waist=getattr(bf['body_state'], 'waist_lean', 0.0),
            t=t, vx=bf['vx'], draw_water_bg=True
        )
        ax_base.set_title(f"BASELINE\nVel jitter: {baseline_vel_jitter:.4f}", fontsize=14, fontweight='bold')
        ax_base.set_aspect('equal')
        ax_base.axis('off')

        # Draw stylized
        sf = stylized_frames[sim_idx]
        ax_style.set_xlim(-2.5, 2.5)
        ax_style.set_ylim(-1.2, 2.5)  # More headroom for rider
        draw_foil_and_rider(
            ax_style, sf['z'], sf['theta'],
            left_leg=sf['left_leg_pos'],
            right_leg=sf['right_leg_pos'],
            left_arm=getattr(sf['body_state'], 'left_arm', 0.0),
            right_arm=getattr(sf['body_state'], 'right_arm', 0.0),
            waist=getattr(sf['body_state'], 'waist_lean', 0.0),
            t=t, vx=sf['vx'], draw_water_bg=True
        )
        jitter_change = ((stylized_vel_jitter - baseline_vel_jitter) / baseline_vel_jitter) * 100
        ax_style.set_title(f"STYLIZED\nVel jitter: {stylized_vel_jitter:.4f} ({jitter_change:+.1f}%)",
                          fontsize=14, fontweight='bold', color='green' if jitter_change < 0 else 'red')
        ax_style.set_aspect('equal')
        ax_style.axis('off')

        # Plot leg velocities
        window = 500  # 5 seconds of history
        start = max(0, sim_idx - window)
        times = np.arange(start, sim_idx + 1) * 0.01

        ax_vel.plot(times, baseline_vels[start:sim_idx+1], 'b-', alpha=0.7, label='Baseline', linewidth=1.5)
        ax_vel.plot(times, stylized_vels[start:sim_idx+1], 'g-', alpha=0.7, label='Stylized', linewidth=1.5)
        ax_vel.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_vel.set_xlim(times[0], times[-1] + 0.1)
        ax_vel.set_ylim(-2.5, 2.5)
        ax_vel.set_ylabel('Leg Velocity (m/s)')
        ax_vel.set_title(f'Leg Velocity Comparison (t={t:.1f}s)', fontsize=12)
        ax_vel.legend(loc='upper right')
        ax_vel.grid(True, alpha=0.3)

        # Plot actions (first dimension - leg command)
        base_acts = [a[0] for a in baseline_actions[start:sim_idx+1]]
        style_acts = [a[0] for a in stylized_actions[start:sim_idx+1]]

        ax_act.plot(times, base_acts, 'b-', alpha=0.7, label='Baseline', linewidth=1.5)
        ax_act.plot(times, style_acts, 'g-', alpha=0.7, label='Stylized', linewidth=1.5)
        ax_act.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_act.set_xlim(times[0], times[-1] + 0.1)
        ax_act.set_ylim(-1.5, 1.5)
        ax_act.set_xlabel('Time (s)')
        ax_act.set_ylabel('Leg Action Command')
        ax_act.set_title('Action Commands (notice smoother transitions in stylized)', fontsize=12)
        ax_act.legend(loc='upper right')
        ax_act.grid(True, alpha=0.3)

        return []

    n_video_frames = n_frames // frame_skip
    print(f"Rendering {n_video_frames} video frames...")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_video_frames, interval=1000/fps, blit=False
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"\nSaved to {output_path}")
    jitter_change = ((stylized_vel_jitter - baseline_vel_jitter) / baseline_vel_jitter) * 100
    act_jitter_change = ((stylized_act_jitter - baseline_act_jitter) / baseline_act_jitter) * 100 if baseline_act_jitter > 0 else 0
    print(f"\nMotion Quality Summary:")
    print(f"  Baseline vel jitter:  {baseline_vel_jitter:.4f}")
    print(f"  Stylized vel jitter:  {stylized_vel_jitter:.4f} ({jitter_change:+.1f}%)")
    print(f"  Baseline act jitter:  {baseline_act_jitter:.4f}")
    print(f"  Stylized act jitter:  {stylized_act_jitter:.4f} ({act_jitter_change:+.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "-b", type=str, required=True)
    parser.add_argument("--stylized", "-s", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="style_comparison.mp4")
    parser.add_argument("--duration", "-d", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    create_comparison_video(
        args.baseline, args.stylized, args.output,
        args.duration, args.fps
    )

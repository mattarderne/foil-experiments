#!/usr/bin/env python3
"""
Standard 4-panel video format as per CLAUDE.md:
- Top-left: Full foil assembly with rider, motion trails, water streaks
- Top-right: Altitude time series
- Bottom-left: Velocity time series
- Bottom-right: Control/state metrics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb
import sys

# Import the proper foil drawing from animate_foil_proper
from animate_foil_proper import draw_frame as draw_foil_frame, compute_body_positions


def run_episode(model_path):
    """Run episode and collect all data."""
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    env = PumpFoilEnv4Limb()

    obs, _ = env.reset(seed=42)
    done = False

    frames = []
    t = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        frame = {
            't': t,
            'z': env.state.z,
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,
            'x': t * env.state.vx,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
            'waist': env.waist_pos if hasattr(env, 'waist_pos') else 0,
            'action': action.copy(),
        }
        frames.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        frame['thrust'] = info.get('thrust', 0)
        done = terminated or truncated
        t += env.dt

    print(f"Episode: {len(frames)} frames, {t:.2f}s")
    print(f"Termination: {info.get('termination_reason', 'unknown')}")
    return frames, env.dt, info


def create_4panel_video(frames, dt, output_path='standard_video.mp4'):
    """Create standard 4-panel video."""
    # Subsample for reasonable fps
    target_fps = 30
    fps = int(1.0 / dt)
    frame_skip = max(1, fps // target_fps)
    actual_fps = fps // frame_skip

    frames_to_render = frames[::frame_skip]
    n_frames = len(frames_to_render)

    print(f"Rendering {n_frames} frames at {actual_fps} fps")

    # Extract time series
    all_t = np.array([f['t'] for f in frames])
    all_z = np.array([f['z'] for f in frames]) * 100  # cm
    all_vx = np.array([f['vx'] for f in frames])
    all_vz = np.array([f['vz'] for f in frames])
    all_thrust = np.array([f.get('thrust', 0) for f in frames])
    all_left_leg = np.array([f['left_leg'] for f in frames]) * 100  # cm
    all_right_leg = np.array([f['right_leg'] for f in frames]) * 100
    all_waist = np.array([np.degrees(f['waist']) for f in frames])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_foil = axes[0, 0]
    ax_alt = axes[0, 1]
    ax_vel = axes[1, 0]
    ax_ctrl = axes[1, 1]

    def animate(frame_idx):
        i = frame_idx * frame_skip  # Index into full data
        frame = frames_to_render[frame_idx]

        # Panel 1: Foil visualization (reuse from animate_foil_proper)
        draw_foil_frame(ax_foil, frames_to_render, frame_idx)

        # Panel 2: Altitude
        ax_alt.clear()
        ax_alt.plot(all_t[:i+1], all_z[:i+1], 'b-', linewidth=2)
        ax_alt.axhline(15, color='green', linestyle='--', alpha=0.7, label='Target (15cm)')
        ax_alt.axhline(20, color='red', linestyle=':', alpha=0.5, label='Surface (20cm)')
        ax_alt.axhline(-50, color='red', linestyle=':', alpha=0.5, label='Floor (-50cm)')
        ax_alt.fill_between([0, all_t[-1]], -50, 20, alpha=0.1, color='green')
        ax_alt.set_xlim(0, all_t[-1])
        ax_alt.set_ylim(-60, 30)
        ax_alt.set_ylabel('Altitude (cm)')
        ax_alt.set_title(f'Altitude | z = {all_z[i]:.1f} cm')
        ax_alt.legend(loc='upper right', fontsize=8)
        ax_alt.grid(True, alpha=0.3)

        # Panel 3: Velocity
        ax_vel.clear()
        ax_vel.plot(all_t[:i+1], all_vx[:i+1], 'g-', linewidth=2, label='vx (forward)')
        ax_vel.plot(all_t[:i+1], all_vz[:i+1], 'b-', linewidth=1.5, alpha=0.7, label='vz (vertical)')
        ax_vel.axhline(4.5, color='green', linestyle='--', alpha=0.5, label='Target vx')
        ax_vel.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax_vel.set_xlim(0, all_t[-1])
        ax_vel.set_ylim(-2, 6)
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.set_title(f'Velocity | vx = {all_vx[i]:.2f} m/s | thrust = {all_thrust[i]:.0f} N')
        ax_vel.legend(loc='upper right', fontsize=8)
        ax_vel.grid(True, alpha=0.3)

        # Panel 4: Controls
        ax_ctrl.clear()
        avg_leg = (all_left_leg[:i+1] + all_right_leg[:i+1]) / 2
        ax_ctrl.plot(all_t[:i+1], avg_leg, 'b-', linewidth=2, label='Legs (cm)')
        ax_ctrl.plot(all_t[:i+1], all_waist[:i+1], 'purple', linewidth=2, label='Waist (°)')
        ax_ctrl.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax_ctrl.axhline(15, color='blue', linestyle='--', alpha=0.3)  # leg limit
        ax_ctrl.axhline(-15, color='blue', linestyle='--', alpha=0.3)
        ax_ctrl.axhline(17, color='purple', linestyle='--', alpha=0.3)  # waist limit
        ax_ctrl.axhline(-17, color='purple', linestyle='--', alpha=0.3)
        ax_ctrl.set_xlim(0, all_t[-1])
        ax_ctrl.set_ylim(-25, 25)
        ax_ctrl.set_xlabel('Time (s)')
        ax_ctrl.set_ylabel('Position')
        ax_ctrl.set_title(f'Controls | legs = {avg_leg[-1]:.1f} cm | waist = {all_waist[i]:.1f}°')
        ax_ctrl.legend(loc='upper right', fontsize=8)
        ax_ctrl.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/actual_fps, blit=False)

    try:
        writer = FFMpegWriter(fps=actual_fps, bitrate=3000)
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error ({e}), trying GIF...")
        from matplotlib.animation import PillowWriter
        gif_path = output_path.replace('.mp4', '.gif')
        writer = PillowWriter(fps=min(actual_fps, 20))
        anim.save(gif_path, writer=writer)
        print(f"Saved: {gif_path}")

    plt.close()


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/model_4limb_modal.zip'
    output = sys.argv[2] if len(sys.argv) > 2 else 'waist_standard_video.mp4'

    frames, dt, info = run_episode(model_path)

    # Print summary
    vx_start = frames[0]['vx']
    vx_end = frames[-1]['vx']
    mean_thrust = np.mean([f.get('thrust', 0) for f in frames])
    print(f"Velocity: {vx_start:.2f} → {vx_end:.2f} m/s (loss: {vx_start - vx_end:.2f})")
    print(f"Mean thrust: {mean_thrust:.1f} N")

    create_4panel_video(frames, dt, output)

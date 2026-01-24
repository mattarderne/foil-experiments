#!/usr/bin/env python3
"""
Training evolution video for curriculum pump foil model.

Shows 4 panels:
1. Passive glide (baseline - no pumping)
2. Manual 2Hz pump (target behavior reference)
3. Untrained agent (random actions)
4. Curriculum trained model (learned pumping)

Bottom row: 5 time series showing all methods overlaid.
"""

import os
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
from foil_env.foil_visualizer import draw_foil_and_rider


# Training foil config (larger, more stable)
FOIL_CONFIG = {'S_stab': 0.035, 'stab_angle': -4.0, 'S': 0.18, 'AR': 8}


def run_episode(env, model=None, action_fn=None, max_steps=1000, name=""):
    """Run episode and collect frame data."""
    obs, _ = env.reset(seed=42)
    done = False

    frames = []
    step = 0

    while not done and step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif action_fn is not None:
            action = action_fn(step, env)
        else:
            action = np.zeros(5)

        frame = {
            't': step * env.dt,
            'z': env.state.z,
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
            'waist': env.waist_pos,
        }
        frames.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    term_reason = info.get('termination_reason', 'max_steps')
    return frames, term_reason


def create_curriculum_video(output_path='curriculum_evolution.mp4'):
    """Create 4-panel training evolution video."""

    scenarios = []

    # 1. Passive glide (baseline)
    print("Running: Passive glide...")
    env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
    frames, term = run_episode(env, action_fn=lambda s, e: np.zeros(5), name="Passive")
    scenarios.append(('Passive Glide\n(no pumping)', frames, '#7f8c8d', term))
    print(f"  Duration: {len(frames)*0.01:.2f}s, term: {term}")

    # 2. Manual 2Hz pump (target behavior)
    print("Running: Manual 2Hz pump...")
    env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
    def manual_pump(step, env):
        t = step * 0.01
        leg = 0.9 * np.sin(2 * np.pi * 2.0 * t)  # 2Hz, full amplitude
        return np.array([leg, leg, 0, 0, 0])
    frames, term = run_episode(env, action_fn=manual_pump, name="Manual")
    scenarios.append(('Manual 2Hz\n(target)', frames, '#3498db', term))
    print(f"  Duration: {len(frames)*0.01:.2f}s, term: {term}")

    # 3. Untrained/random agent
    print("Running: Untrained agent...")
    env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
    np.random.seed(123)
    def random_action(step, env):
        # Smoothed random noise (simulates untrained policy)
        t = step * 0.01
        noise = np.random.randn(5) * 0.3
        # Add some slow oscillation to make it look like attempted learning
        base = 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Very slow, wrong frequency
        return np.clip(np.array([base + noise[0], base + noise[1],
                                  noise[2], noise[3], noise[4]]), -1, 1)
    frames, term = run_episode(env, action_fn=random_action, name="Untrained")
    scenarios.append(('Untrained\n(random)', frames, '#e74c3c', term))
    print(f"  Duration: {len(frames)*0.01:.2f}s, term: {term}")

    # 4. Trained curriculum model
    print("Running: Trained model...")
    env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
    model = PPO.load("checkpoints/model_curriculum_pump.zip")
    frames, term = run_episode(env, model=model, name="Trained")
    scenarios.append(('Curriculum\nTrained', frames, '#27ae60', term))
    print(f"  Duration: {len(frames)*0.01:.2f}s, term: {term}")

    # Pad to same length
    max_frames = max(len(s[1]) for s in scenarios)
    for name, frames, color, term in scenarios:
        orig_len = len(frames)
        frames[0]['orig_len'] = orig_len
        last = frames[-1].copy()
        while len(frames) < max_frames:
            frames.append(last)

    # Create figure
    print(f"\nCreating video with {max_frames} frames...")
    fig = plt.figure(figsize=(20, 10))

    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.2, 1], hspace=0.25, wspace=0.3)

    # Top row: 4 foil panels
    axes_foil = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
    ]
    # Info panel (5th cell)
    ax_info = fig.add_subplot(gs[0, 4])
    ax_info.axis('off')

    # Bottom row: 5 time series
    ax_z = fig.add_subplot(gs[1, 0])
    ax_vx = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_arm = fig.add_subplot(gs[1, 3])
    ax_waist = fig.add_subplot(gs[1, 4])

    frame_skip = 3
    frames_to_render = list(range(0, max_frames, frame_skip))

    def animate(frame_num):
        i = frames_to_render[frame_num]

        # Draw each scenario
        for idx, (name, frames, color, term) in enumerate(scenarios):
            ax = axes_foil[idx]
            ax.clear()

            orig_len = frames[0].get('orig_len', len(frames))
            is_crashed = i >= orig_len
            frame = frames[min(i, len(frames)-1)]

            # Draw foil and rider using the reusable module
            draw_foil_and_rider(
                ax,
                z=frame['z'],
                theta=frame['theta'],
                left_leg=frame['left_leg'],
                right_leg=frame['right_leg'],
                left_arm=frame['left_arm'],
                right_arm=frame['right_arm'],
                waist=frame['waist'],
                t=frame['t'],
                foil_config=FOIL_CONFIG,
                is_crashed=is_crashed,
                max_leg_ext=0.15,
                max_arm_swing=1.5,
            )

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 2.0)
            ax.set_aspect('equal')
            ax.set_facecolor('lightcyan')

            # Title with status
            duration = orig_len * 0.01
            if is_crashed:
                status = f"CRASHED @ {duration:.1f}s"
                title_color = 'red'
            else:
                status = f"{frame['t']:.1f}s / {duration:.1f}s"
                title_color = color
            ax.set_title(f"{name}\n{status}", fontsize=10, fontweight='bold', color=title_color)

        # Info panel
        ax_info.clear()
        ax_info.axis('off')
        ax_info.text(0.5, 0.9, "Training Evolution", fontsize=12, fontweight='bold',
                    ha='center', va='top', transform=ax_info.transAxes)
        ax_info.text(0.5, 0.75, f"Foil: Training (stable)", fontsize=9,
                    ha='center', va='top', transform=ax_info.transAxes)
        ax_info.text(0.5, 0.6, f"S={FOIL_CONFIG['S']}m², AR={FOIL_CONFIG['AR']}", fontsize=8,
                    ha='center', va='top', transform=ax_info.transAxes, color='gray')

        # Legend
        for j, (name, _, color, _) in enumerate(scenarios):
            ax_info.scatter([0.15], [0.4 - j*0.1], c=[color], s=50)
            ax_info.text(0.25, 0.4 - j*0.1, name.split('\n')[0], fontsize=8,
                        va='center', transform=ax_info.transAxes)

        # Time series plots
        t_now = scenarios[0][1][min(i, len(scenarios[0][1])-1)]['t']

        plot_configs = [
            (ax_z, 'z', 'Altitude (cm)', (-60, 30), lambda f: f['z'] * 100),
            (ax_vx, 'vx', 'Velocity (m/s)', (0, 6), lambda f: f['vx']),
            (ax_leg, 'leg', 'Leg pos (cm)', (-20, 20),
             lambda f: (f['left_leg'] + f['right_leg']) / 2 * 100),
            (ax_arm, 'arm', 'Arm pos (deg)', (-100, 100),
             lambda f: np.degrees((f['left_arm'] + f['right_arm']) / 2)),
            (ax_waist, 'waist', 'Waist (deg)', (-25, 25),
             lambda f: np.degrees(f['waist'])),
        ]

        for ax, key, ylabel, limits, value_fn in plot_configs:
            ax.clear()
            for name, frames, color, term in scenarios:
                orig_len = frames[0].get('orig_len', len(frames))
                plot_len = min(i + 1, orig_len)
                times = [f['t'] for f in frames[:plot_len]]
                values = [value_fn(f) for f in frames[:plot_len]]
                linestyle = '--' if 'Passive' in name else '-'
                linewidth = 2 if 'Trained' in name else 1.5
                ax.plot(times, values, color=color, linestyle=linestyle,
                       linewidth=linewidth, label=name.split('\n')[0])

            ax.axvline(x=t_now, color='black', linestyle='-', alpha=0.2)
            ax.set_xlim(0, max_frames * 0.01)
            ax.set_ylim(limits)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)', fontsize=8)
            if ax == ax_z:
                ax.legend(loc='upper right', fontsize=7)

        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render),
                         interval=33, blit=False)

    try:
        writer = FFMpegWriter(fps=30, bitrate=3000)
        anim.save(output_path, writer=writer)
        print(f"\nSaved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")
        # Try saving as gif
        try:
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=15)
            print(f"Saved as GIF: {gif_path}")
        except Exception as e2:
            print(f"GIF error: {e2}")

    plt.close()

    # Print summary
    print("\n=== SUMMARY ===")
    for name, frames, color, term in scenarios:
        orig_len = frames[0].get('orig_len', len(frames))

        # Analyze pumping behavior
        leg_positions = [(f['left_leg'] + f['right_leg'])/2 for f in frames[:orig_len]]
        leg_range = max(leg_positions) - min(leg_positions)
        leg_range_pct = leg_range / (2 * 0.15) * 100

        # Frequency estimate via zero crossings
        leg_arr = np.array(leg_positions)
        zero_crossings = np.where(np.diff(np.signbit(leg_arr)))[0]
        if len(zero_crossings) >= 2:
            avg_half_period = np.mean(np.diff(zero_crossings)) * 0.01
            freq = 0.5 / avg_half_period if avg_half_period > 0 else 0
        else:
            freq = 0

        print(f"{name.split(chr(10))[0]:15s}: {orig_len*0.01:.1f}s | "
              f"leg_range={leg_range_pct:.0f}% | freq={freq:.1f}Hz | {term}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='curriculum_evolution.mp4',
                       help='Output video path')
    args = parser.parse_args()

    create_curriculum_video(args.output)

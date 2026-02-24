#!/usr/bin/env python3
"""
Standard video creation for pump foil training evolution.

Creates 4-panel comparison video showing training progression:
- Top row: 4 foil visualizations (1%, 25%, 50%, 100% or specified checkpoints)
- Bottom row: 5 time series plots (altitude, velocity, leg, arm, waist)

Now loads config from checkpoint files for reproducibility.

Usage:
    python create_video.py --checkpoint-dir checkpoints/run
    python create_video.py --checkpoint-dir checkpoints/run -o evolution.mp4
"""

import os
import sys
import argparse
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
from foil_env.foil_visualizer import draw_foil_and_rider, compute_body_positions, draw_motion_trails, draw_water
from foil_env.checkpoint_utils import load_checkpoint, has_config, create_env_from_config
from foil_env.config import TrainingConfig, TRAINING_FOIL, DEFAULT_LIMBS
from foil_env.metrics import analyze_episode, format_metrics

# Fallback foil config for legacy checkpoints (no config file)
LEGACY_FOIL_CONFIG = {
    'S_stab': 0.035,
    'stab_angle': -4.0,
    'S': 0.18,
    'AR': 8,
}

# Physics constants for limit lines
RIDER_MASS = 70.0
BOARD_MASS = 6.0
RIDING_DEPTH = 0.20  # m
MAST_LENGTH = 0.70   # m
RHO = 1000.0         # water density kg/m³
G = 9.81
STALL_ANGLE_DEG = 13.0


def compute_flight_limits(foil_config):
    """Compute physical flight limits based on foil config."""
    # Height limits (in cm for plotting)
    # Operating range = mast_length (70cm)
    max_height_cm = RIDING_DEPTH * 100  # foil breach (z >= riding_depth)
    min_height_cm = (RIDING_DEPTH - MAST_LENGTH) * 100  # board touchdown (z <= riding_depth - mast_length)

    # Stall velocity calculation
    # At stall angle, Cl_max = lift_slope * stall_angle
    AR = foil_config.get('AR', 8)
    S = foil_config.get('S', 0.18)
    lift_slope = (2 * np.pi) / (1 + 2 / AR)
    stall_rad = np.radians(STALL_ANGLE_DEG)
    Cl_max = lift_slope * stall_rad

    # L = 0.5 * rho * v^2 * S * Cl = W
    # v_stall = sqrt(W / (0.5 * rho * S * Cl_max))
    W = (RIDER_MASS + BOARD_MASS) * G
    v_stall = np.sqrt(W / (0.5 * RHO * S * Cl_max))

    return {
        'max_height_cm': max_height_cm,
        'min_height_cm': min_height_cm,
        'stall_velocity': v_stall,
        'weight': W,
    }


FLIGHT_LIMITS = compute_flight_limits(LEGACY_FOIL_CONFIG)

# Colors for each panel
COLORS = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']  # red, orange, blue, green


def run_episode(env, model=None, max_steps=1000):
    """Run episode and collect frame data."""
    obs, _ = env.reset(seed=42)
    frames = []
    step = 0
    done = False

    while not done and step < max_steps:
        if model is not None:
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

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    return frames, info.get('termination_reason', 'max_steps')


def analyze_pumping_simple(frames, dt=0.01, max_leg_ext=0.15):
    """Simple pumping analysis (legacy, for compatibility)."""
    leg_positions = [(f['left_leg'] + f['right_leg'])/2 for f in frames]
    leg_range = max(leg_positions) - min(leg_positions)
    leg_range_pct = leg_range / (2 * max_leg_ext) * 100

    # Frequency via zero crossings
    leg_arr = np.array(leg_positions)
    crossings = np.where(np.diff(np.signbit(leg_arr)))[0]
    if len(crossings) >= 2:
        avg_half_period = np.mean(np.diff(crossings)) * dt
        freq = 0.5 / avg_half_period if avg_half_period > 0 else 0
    else:
        freq = 0

    return {'leg_range_pct': leg_range_pct, 'freq': freq, 'duration': len(frames) * dt}


def analyze_pumping(frames, dt=0.01, max_leg_ext=0.15):
    """Analyze pumping using new metrics system (detects true oscillation)."""
    # Use the new metrics system
    metrics = analyze_episode(frames, dt=dt, max_leg_ext=max_leg_ext)

    # Return compatible format with additional metrics
    return {
        'leg_range_pct': metrics.get('leg_range_pct', 0),
        'freq': metrics.get('leg_frequency', 0),
        'duration': metrics.get('duration', len(frames) * dt),
        'leg_oscillating': metrics.get('leg_is_oscillating', False),
        'arm_oscillating': metrics.get('arm_is_oscillating', False),
        'arm_range_deg': metrics.get('arm_range_deg', 0),
    }


def create_evolution_video(checkpoint_dir: str, output_path: str = 'training_evolution.mp4', dark_mode: bool = True):
    """Create training evolution video from checkpoints."""

    # Find checkpoints
    checkpoints = []
    for pct in [1, 25, 50, 75, 100]:
        path = os.path.join(checkpoint_dir, f"model_{pct}pct.zip")
        if os.path.exists(path):
            checkpoints.append((pct, path))

    if len(checkpoints) < 2:
        print(f"Error: Need at least 2 checkpoints in {checkpoint_dir}/")
        print(f"Expected: model_1pct.zip, model_25pct.zip, model_50pct.zip, model_75pct.zip, model_100pct.zip")
        return None

    # Use up to 4 checkpoints for the 4 panels
    if len(checkpoints) > 4:
        # Pick 1%, 25%, 75%, 100%
        checkpoints = [c for c in checkpoints if c[0] in [1, 25, 75, 100]]

    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}%")

    # Run episodes for each checkpoint
    scenarios = []
    foil_config_used = None  # Track which config we're using

    for pct, path in checkpoints:
        print(f"Running {pct}% model...")

        # Try to load config from checkpoint
        if has_config(path):
            model, config = load_checkpoint(path)
            env = create_env_from_config(config)
            foil_config_used = config.foil.to_dict()
            print(f"  (loaded config: S={config.foil.S}, arm_start={np.degrees(config.limbs.arm_start_position):.0f}°)")
        else:
            # Legacy checkpoint - use fallback config
            model = PPO.load(path)
            env = PumpFoilEnvCurriculum(config=LEGACY_FOIL_CONFIG, curriculum_phase=2)
            foil_config_used = LEGACY_FOIL_CONFIG
            print(f"  (legacy checkpoint, using fallback config)")

        frames, term = run_episode(env, model=model)
        stats = analyze_pumping(frames)
        scenarios.append({
            'name': f'{pct}%',
            'frames': frames,
            'color': COLORS[len(scenarios) % len(COLORS)],
            'term': term,
            'stats': stats,
            'foil_config': foil_config_used,  # Store config for visualization
        })

        # Show oscillation status for new metrics
        leg_status = "✓" if stats.get('leg_oscillating', False) else "✗"
        arm_status = "✓" if stats.get('arm_oscillating', False) else "✗"
        print(f"  {stats['duration']:.1f}s | leg: {stats['leg_range_pct']:.0f}% {leg_status} {stats['freq']:.1f}Hz | arm: {stats.get('arm_range_deg', 0):.0f}° {arm_status} | {term}")

    # Pad frames to same length
    max_frames = max(len(s['frames']) for s in scenarios)
    for s in scenarios:
        orig_len = len(s['frames'])
        s['orig_len'] = orig_len
        last = s['frames'][-1].copy()
        while len(s['frames']) < max_frames:
            s['frames'].append(last)

    # Create video
    print(f"\nCreating video ({max_frames} frames)...")
    fig = plt.figure(figsize=(20, 10))

    # Dark mode styling
    if dark_mode:
        fig.patch.set_facecolor('#1a1a1a')  # Dark figure background
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'

    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.2, 1], hspace=0.25, wspace=0.3)

    # Top row: foil panels (up to 4)
    n_panels = min(4, len(scenarios))
    axes_foil = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]

    # Info panel
    ax_info = fig.add_subplot(gs[0, 4])
    ax_info.axis('off')

    # Bottom row: time series
    ax_z = fig.add_subplot(gs[1, 0])
    ax_vx = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_arm = fig.add_subplot(gs[1, 3])
    ax_waist = fig.add_subplot(gs[1, 4])

    frame_skip = 3
    render_frames = list(range(0, max_frames, frame_skip))

    # Pre-compute body positions for all frames (for motion trails)
    TRAIL_LENGTH = 15
    for s in scenarios:
        s['body_positions'] = []
        for frame in s['frames']:
            pos = compute_body_positions(
                z=frame['z'], theta=frame['theta'],
                left_leg=frame['left_leg'], right_leg=frame['right_leg'],
                left_arm=frame['left_arm'], right_arm=frame['right_arm'],
                waist=frame['waist']
            )
            s['body_positions'].append(pos)

    def animate(frame_num):
        i = render_frames[frame_num]

        # Draw each scenario
        for idx, s in enumerate(scenarios[:n_panels]):
            ax = axes_foil[idx]
            ax.clear()

            is_crashed = i >= s['orig_len']
            frame_idx = min(i, len(s['frames'])-1)
            frame = s['frames'][frame_idx]

            # Draw water (or skip for dark mode)
            if not dark_mode:
                draw_water(ax, t=frame['t'], is_crashed=is_crashed, vx=frame['vx'])

            # Draw motion trails (before foil so trails are behind)
            if not is_crashed and frame_idx > 0:
                trail_start = max(0, frame_idx - TRAIL_LENGTH)
                trail_positions = s['body_positions'][trail_start:frame_idx + 1]
                draw_motion_trails(ax, trail_positions, current_vx=frame['vx'], dt=0.01)

            # Draw foil and rider (skip water since we drew it above)
            draw_foil_and_rider(
                ax, z=frame['z'], theta=frame['theta'],
                left_leg=frame['left_leg'], right_leg=frame['right_leg'],
                left_arm=frame['left_arm'], right_arm=frame['right_arm'],
                waist=frame['waist'], t=frame['t'],
                foil_config=s.get('foil_config', LEGACY_FOIL_CONFIG), is_crashed=is_crashed,
                vx=frame['vx'], draw_water_bg=False,
            )

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 2.0)
            ax.set_aspect('equal')
            ax.set_facecolor('black' if dark_mode else 'lightcyan')

            # Draw water line in dark mode
            if dark_mode:
                ax.axhline(y=0.0, color='#3366cc', linewidth=2, alpha=0.6, zorder=2)
                ax.axhspan(-1.5, 0.0, alpha=0.15, color='#3366cc', zorder=1)

            duration = s['orig_len'] * 0.01
            status = f"CRASHED @ {duration:.1f}s" if is_crashed else f"{frame['t']:.1f}s"
            title_color = '#ff6666' if is_crashed else s['color']
            ax.set_title(f"Training {s['name']}\n{status}", fontsize=10, fontweight='bold', color=title_color)

        # Info panel
        ax_info.clear()
        ax_info.axis('off')
        if dark_mode:
            ax_info.set_facecolor('#1a1a1a')
        info_text_color = 'white' if dark_mode else 'black'
        dim_text_color = '#888888'

        ax_info.text(0.5, 0.98, "Training Evolution", fontsize=12, fontweight='bold',
                    ha='center', transform=ax_info.transAxes, color=info_text_color)

        # Foil config section
        fc = scenarios[0].get('foil_config', LEGACY_FOIL_CONFIG)
        foil_area_cm2 = fc.get('S', 0.18) * 10000
        stab_area_cm2 = fc.get('S_stab', 0.035) * 10000
        ar = fc.get('AR', 8)
        ax_info.text(0.5, 0.88, "─── Foil Config ───", fontsize=9, ha='center',
                    transform=ax_info.transAxes, color=dim_text_color)
        ax_info.text(0.5, 0.80, f"Wing: {foil_area_cm2:.0f} cm²  |  Stab: {stab_area_cm2:.0f} cm²", fontsize=9,
                    ha='center', transform=ax_info.transAxes, color=info_text_color)
        ax_info.text(0.5, 0.73, f"AR: {ar}  |  Stab angle: {fc.get('stab_angle', -4)}°", fontsize=8,
                    ha='center', transform=ax_info.transAxes, color=dim_text_color)

        # Checkpoint results
        ax_info.text(0.5, 0.62, "─── Results ───", fontsize=9, ha='center',
                    transform=ax_info.transAxes, color=dim_text_color)

        for j, s in enumerate(scenarios):
            y = 0.54 - j * 0.12
            ax_info.scatter([0.08], [y], c=[s['color']], s=50, transform=ax_info.transAxes)
            stats = s['stats']
            ax_info.text(0.14, y, f"{s['name']}: {stats['duration']:.1f}s | {stats['freq']:.1f}Hz | {stats['leg_range_pct']:.0f}%",
                        fontsize=8, va='center', transform=ax_info.transAxes, color=info_text_color)

        # Time series
        t_now = scenarios[0]['frames'][min(i, len(scenarios[0]['frames'])-1)]['t']

        plot_configs = [
            (ax_z, 'Altitude (cm)', (-60, 30), lambda f: f['z'] * 100),
            (ax_vx, 'Velocity (m/s)', (0, 6), lambda f: f['vx']),
            (ax_leg, 'Leg pos (cm)', (-20, 20), lambda f: (f['left_leg'] + f['right_leg']) / 2 * 100),
            (ax_arm, 'Arm pos (deg)', (-100, 100), lambda f: np.degrees((f['left_arm'] + f['right_arm']) / 2)),
            (ax_waist, 'Waist (deg)', (-25, 25), lambda f: np.degrees(f['waist'])),
        ]

        for ax, ylabel, limits, value_fn in plot_configs:
            ax.clear()

            # Soft grey background for charts in dark mode
            if dark_mode:
                ax.set_facecolor('#2d2d2d')

            for s in scenarios:
                plot_len = min(i + 1, s['orig_len'])
                times = [f['t'] for f in s['frames'][:plot_len]]
                values = [value_fn(f) for f in s['frames'][:plot_len]]
                lw = 2.5 if '100' in s['name'] else 1.5
                ax.plot(times, values, color=s['color'], linewidth=lw, label=s['name'])

            vline_color = '#666666' if dark_mode else 'black'
            ax.axvline(x=t_now, color=vline_color, linestyle='-', alpha=0.3)
            ax.set_xlim(0, max_frames * 0.01)
            ax.set_ylim(limits)
            ax.set_ylabel(ylabel, fontsize=8, color='white' if dark_mode else 'black')
            ax.set_xlabel('Time (s)', fontsize=8, color='white' if dark_mode else 'black')
            ax.tick_params(colors='white' if dark_mode else 'black')
            ax.grid(True, alpha=0.2 if dark_mode else 0.3, color='#555555' if dark_mode else 'gray')

            # Spine colors for dark mode
            if dark_mode:
                for spine in ax.spines.values():
                    spine.set_color('#555555')

            # Add physical limit lines
            if ax == ax_z:
                # Max height (foil breach) and min height (board touchdown)
                ax.axhline(y=FLIGHT_LIMITS['max_height_cm'], color='#ff6666', linestyle='--',
                          linewidth=1, alpha=0.7, label='breach')
                ax.axhline(y=FLIGHT_LIMITS['min_height_cm'], color='#ff6666', linestyle='--',
                          linewidth=1, alpha=0.7, label='touchdown')
                water_color = '#3366cc' if dark_mode else 'blue'
                ax.axhline(y=0, color=water_color, linestyle='-', linewidth=1, alpha=0.5)  # water surface reference
                leg = ax.legend(loc='upper right', fontsize=6)
                if dark_mode:
                    leg.get_frame().set_facecolor('#2d2d2d')
                    for text in leg.get_texts():
                        text.set_color('white')
            elif ax == ax_vx:
                # Stall velocity line
                ax.axhline(y=FLIGHT_LIMITS['stall_velocity'], color='#ff6666', linestyle='--',
                          linewidth=1.5, alpha=0.7)
                ax.text(0.5, FLIGHT_LIMITS['stall_velocity'] + 0.15, f"stall ({FLIGHT_LIMITS['stall_velocity']:.1f} m/s)",
                       fontsize=7, color='#ff6666', alpha=0.8)

        return []

    anim = FuncAnimation(fig, animate, frames=len(render_frames), interval=33, blit=False)

    try:
        writer = FFMpegWriter(fps=30, bitrate=3000)
        anim.save(output_path, writer=writer)
        print(f"\nSaved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=15)
            print(f"Saved as GIF: {gif_path}")
        except Exception as e2:
            print(f"GIF error: {e2}")

    plt.close()

    # Print summary
    print("\n=== TRAINING EVOLUTION SUMMARY ===")
    for s in scenarios:
        stats = s['stats']
        print(f"{s['name']:>5}: {stats['duration']:.1f}s | leg_range={stats['leg_range_pct']:.0f}% | freq={stats['freq']:.1f}Hz | {s['term']}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training evolution video")
    parser.add_argument("--checkpoint-dir", "-c", required=True,
                       help="Directory containing model_Npct.zip checkpoints")
    parser.add_argument("--output", "-o", default="training_evolution.mp4",
                       help="Output video path")
    parser.add_argument("--light", action="store_true",
                       help="Use light mode (default is dark mode)")
    args = parser.parse_args()

    create_evolution_video(args.checkpoint_dir, args.output, dark_mode=not args.light)

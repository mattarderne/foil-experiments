"""
Analyze waist articulation training results.

Creates visualizations showing:
1. Training progression (baseline vs trained)
2. Waist action analysis (phase, amplitude, coordination)
3. Performance comparison

Usage: python analyze_waist_training.py [model_path]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO

from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_episode(model_path=None, seed=42, max_frames=1500):
    """Run episode and collect detailed trajectory data."""
    env = PumpFoilEnv4Limb()

    if model_path:
        model = PPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    done = False

    trajectory = []
    while not done and len(trajectory) < max_frames:
        if model_path:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(5, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            't': len(trajectory) * 0.01,
            'z': env.state.z,
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,
            'waist_pos': env.waist_pos,
            'waist_vel': env.waist_vel,
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
            'action': action.copy(),
            'reward': reward,
            'positions': env.get_body_positions(),
        })
        done = terminated or truncated

    return trajectory, info.get('termination_reason', 'max_steps')


def analyze_waist_usage(trajectory):
    """Analyze how the agent uses the waist action."""
    waist_actions = [f['action'][4] for f in trajectory]
    leg_actions = [(f['action'][0] + f['action'][1]) / 2 for f in trajectory]
    waist_pos = [f['waist_pos'] for f in trajectory]

    # Compute phase relationship using cross-correlation
    if len(trajectory) > 100:
        waist_arr = np.array(waist_actions)
        leg_arr = np.array(leg_actions)

        # Normalize
        waist_norm = (waist_arr - np.mean(waist_arr)) / (np.std(waist_arr) + 1e-8)
        leg_norm = (leg_arr - np.mean(leg_arr)) / (np.std(leg_arr) + 1e-8)

        # Cross-correlation
        correlation = np.correlate(waist_norm, leg_norm, mode='full')
        lags = np.arange(-len(waist_norm) + 1, len(waist_norm))

        # Find peak (phase relationship)
        peak_idx = np.argmax(np.abs(correlation))
        phase_lag = lags[peak_idx] * 0.01  # Convert to seconds

        # Compute frequency via FFT
        if len(waist_arr) > 50:
            fft = np.fft.fft(waist_norm)
            freqs = np.fft.fftfreq(len(waist_arr), 0.01)
            pos_mask = freqs > 0.5  # Above 0.5 Hz
            if np.any(pos_mask):
                peak_freq_idx = np.argmax(np.abs(fft[pos_mask]))
                peak_freq = freqs[pos_mask][peak_freq_idx]
            else:
                peak_freq = 0
        else:
            peak_freq = 0
    else:
        phase_lag = 0
        peak_freq = 0

    return {
        'waist_mean': np.mean(waist_actions),
        'waist_std': np.std(waist_actions),
        'waist_range': np.max(waist_actions) - np.min(waist_actions),
        'phase_lag_sec': phase_lag,
        'peak_freq_hz': abs(peak_freq),
        'waist_pos_mean': np.mean(waist_pos),
        'waist_pos_std': np.std(waist_pos),
    }


def create_comparison_plots(model_path, output_prefix='waist_analysis'):
    """Create comparison plots for baseline vs trained."""

    print("Running baseline episode...")
    baseline_traj, baseline_reason = run_episode(None, seed=42)
    print(f"  Baseline: {len(baseline_traj)} steps ({baseline_traj[-1]['t']:.2f}s), {baseline_reason}")

    print("Running trained episode...")
    trained_traj, trained_reason = run_episode(model_path, seed=42)
    print(f"  Trained: {len(trained_traj)} steps ({trained_traj[-1]['t']:.2f}s), {trained_reason}")

    # Analyze waist usage
    baseline_analysis = analyze_waist_usage(baseline_traj)
    trained_analysis = analyze_waist_usage(trained_traj)

    print("\nWaist Usage Analysis:")
    print(f"  Baseline waist action: mean={baseline_analysis['waist_mean']:.3f}, std={baseline_analysis['waist_std']:.3f}")
    print(f"  Trained waist action:  mean={trained_analysis['waist_mean']:.3f}, std={trained_analysis['waist_std']:.3f}")
    print(f"  Trained waist range:   {trained_analysis['waist_range']:.3f}")
    print(f"  Trained frequency:     {trained_analysis['peak_freq_hz']:.2f} Hz")
    print(f"  Leg-waist phase lag:   {trained_analysis['phase_lag_sec']:.3f} s")

    # Create 4-panel analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Waist Articulation Training Analysis\nBaseline: {len(baseline_traj)*0.01:.1f}s | Trained: {len(trained_traj)*0.01:.1f}s', fontsize=14)

    # Panel 1: Episode comparison - altitude
    ax = axes[0, 0]
    t_base = [f['t'] for f in baseline_traj]
    t_train = [f['t'] for f in trained_traj]
    ax.plot(t_base, [f['z'] for f in baseline_traj], 'gray', linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(t_train, [f['z'] for f in trained_traj], 'green', linewidth=2, label='Trained')
    ax.axhline(y=0.15, color='blue', linestyle='--', alpha=0.5, label='Target')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Floor')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(t_train[-1], 5) if trained_traj else 5)

    # Panel 2: Waist action over time (trained only)
    ax = axes[0, 1]
    ax.plot(t_train, [f['action'][4] for f in trained_traj], 'orange', linewidth=1.5, label='Waist action')
    ax.plot(t_train, [(f['action'][0] + f['action'][1])/2 for f in trained_traj], 'purple', linewidth=1.5, alpha=0.7, label='Avg leg action')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action')
    ax.set_title(f'Actions Over Time (Trained)\nFreq: {trained_analysis["peak_freq_hz"]:.1f} Hz, Phase lag: {trained_analysis["phase_lag_sec"]*1000:.0f} ms')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Waist position vs leg position (phase plot)
    ax = axes[1, 0]
    if len(trained_traj) > 100:
        # Take middle portion for cleaner phase plot
        start = len(trained_traj) // 4
        end = 3 * len(trained_traj) // 4
        waist_pos = [np.degrees(f['waist_pos']) for f in trained_traj[start:end]]
        leg_pos = [(f['left_leg'] + f['right_leg']) / 2 * 100 for f in trained_traj[start:end]]

        # Color by time
        colors = np.linspace(0, 1, len(waist_pos))
        scatter = ax.scatter(leg_pos, waist_pos, c=colors, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('Avg Leg Position (cm)')
        ax.set_ylabel('Waist Lean (deg)')
        ax.set_title('Waist vs Leg Phase Plot (middle 50%)')
        plt.colorbar(scatter, ax=ax, label='Time progression')
    ax.grid(True, alpha=0.3)

    # Panel 4: Velocity comparison
    ax = axes[1, 1]
    ax.plot(t_base, [f['vx'] for f in baseline_traj], 'gray', linewidth=2, label='Baseline Vx', alpha=0.7)
    ax.plot(t_train, [f['vx'] for f in trained_traj], 'green', linewidth=2, label='Trained Vx')
    ax.axhline(y=4.5, color='blue', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_plots.png', dpi=150)
    print(f"\nSaved: {output_prefix}_plots.png")
    plt.close()

    return baseline_traj, trained_traj


def create_progression_video(model_path, output_path='waist_training_final.mp4'):
    """Create training progression video with proper 4-panel format."""

    print("\nCreating progression video...")

    # Run episodes
    print("  Running baseline...")
    baseline_traj, _ = run_episode(None, seed=42)

    print("  Running trained model...")
    trained_traj, _ = run_episode(model_path, seed=42)

    trajectories = [
        ('Baseline', baseline_traj, 'gray', len(baseline_traj)),
        ('2M steps', trained_traj, '#27ae60', len(trained_traj)),
    ]

    # Pad shorter trajectories
    max_frames = max(len(t[1]) for t in trajectories)
    for name, traj, color, orig_len in trajectories:
        last = traj[-1].copy()
        while len(traj) < max_frames:
            traj.append(last)

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    axes_foil = [fig.add_subplot(2, 2, i+1) for i in range(2)]
    ax_time = fig.add_subplot(2, 2, 3)
    ax_waist = fig.add_subplot(2, 2, 4)

    fig.suptitle('4-Limb Pump Foil with Waist - Training Result', fontsize=14)

    frame_skip = 2
    frames_to_render = list(range(0, max_frames, frame_skip))

    def draw_foil(ax, frame, color='black'):
        z = frame['z']
        theta = frame['theta']
        positions = frame['positions']

        water_y = 0.0
        board_center_y = water_y + 0.2 + z

        def rotate(x, y, angle):
            c, s = np.cos(angle), np.sin(angle)
            return x * c - y * s, x * s + y * c

        def transform(pt):
            rx, ry = rotate(pt[0], pt[1], theta)
            return (rx, ry + board_center_y)

        # Water
        ax.axhspan(-1.5, water_y, alpha=0.3, color='dodgerblue')
        ax.axhline(y=water_y, color='blue', linewidth=2, alpha=0.5)

        # Board
        board = [(-0.3, 0), (0.3, 0), (0.3, 0.05), (-0.3, 0.05)]
        ax.add_patch(Polygon([transform(p) for p in board], facecolor='saddlebrown', edgecolor='black'))

        # Mast
        mt, mb = transform((0, 0)), transform((0, -0.7))
        ax.plot([mt[0], mb[0]], [mt[1], mb[1]], 'dimgray', linewidth=4)

        # Wings
        ff, fr = transform((0.28, -0.7)), transform((-0.42, -0.7))
        ax.plot([ff[0], fr[0]], [ff[1], fr[1]], 'dimgray', linewidth=3)

        # Rider
        for side, c in [('left', 'royalblue'), ('right', 'crimson')]:
            ax.plot([positions[f'{side}_ankle'][0], positions[f'{side}_knee'][0]],
                   [positions[f'{side}_ankle'][1], positions[f'{side}_knee'][1]], c, linewidth=5)
            ax.plot([positions[f'{side}_knee'][0], positions[f'{side}_hip'][0]],
                   [positions[f'{side}_knee'][1], positions[f'{side}_hip'][1]], c, linewidth=6)

        # Torso (shows waist lean)
        ax.plot([positions['torso_base'][0], positions['torso_top'][0]],
               [positions['torso_base'][1], positions['torso_top'][1]], 'darkgreen', linewidth=8)

        # Arms
        for side, c in [('left', 'royalblue'), ('right', 'crimson')]:
            ax.plot([positions[f'{side}_shoulder'][0], positions[f'{side}_elbow'][0]],
                   [positions[f'{side}_shoulder'][1], positions[f'{side}_elbow'][1]], c, linewidth=4, alpha=0.8)

        # Head
        head = positions['head']
        ax.add_patch(Circle(head, 0.08, facecolor='peachpuff', edgecolor='black'))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.1, 2.2)
        ax.set_aspect('equal')
        ax.set_facecolor('lightcyan')

    def animate(frame_num):
        i = frames_to_render[frame_num]
        t = trajectories[0][1][min(i, len(trajectories[0][1])-1)]['t']

        # Draw foil panels
        for idx, (name, traj, color, orig_len) in enumerate(trajectories):
            ax = axes_foil[idx]
            ax.clear()

            frame = traj[min(i, len(traj)-1)]
            is_crashed = i >= orig_len

            draw_foil(ax, frame)

            if is_crashed:
                ax.text(0, frame['z'] + 0.6, 'CRASHED', ha='center', fontsize=10, color='red', fontweight='bold')

            waist_deg = np.degrees(frame['waist_pos'])
            duration = orig_len * 0.01
            status = f'({duration:.1f}s)' if not is_crashed else f'CRASHED @ {duration:.1f}s'
            ax.set_title(f'{name}\nwaist: {waist_deg:+.1f}° {status}', color=color if not is_crashed else 'red')

        # Time series - altitude
        ax_time.clear()
        for name, traj, color, orig_len in trajectories:
            times = [f['t'] for f in traj[:i+1]]
            z_vals = [f['z'] for f in traj[:i+1]]
            ls = '--' if 'Baseline' in name else '-'
            ax_time.plot(times, z_vals, color=color, linestyle=ls, linewidth=2, label=name)
        ax_time.axhline(y=0.15, color='blue', linestyle='--', alpha=0.5)
        ax_time.axvline(x=t, color='black', alpha=0.3)
        ax_time.set_xlim(0, max(5, max_frames * 0.01))
        ax_time.set_ylim(-0.6, 0.3)
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Altitude (m)')
        ax_time.set_title('Altitude')
        ax_time.legend(loc='upper right')
        ax_time.grid(True, alpha=0.3)

        # Waist action (trained only)
        ax_waist.clear()
        trained_traj = trajectories[1][1]
        times = [f['t'] for f in trained_traj[:i+1]]
        waist = [np.degrees(f['waist_pos']) for f in trained_traj[:i+1]]
        leg = [(f['left_leg'] + f['right_leg']) / 2 * 100 for f in trained_traj[:i+1]]
        ax_waist.plot(times, waist, 'orange', linewidth=2, label='Waist (deg)')
        ax_waist.plot(times, leg, 'purple', linewidth=2, label='Legs (cm)')
        ax_waist.axhline(y=0, color='gray', linestyle=':')
        ax_waist.axvline(x=t, color='black', alpha=0.3)
        ax_waist.set_xlim(0, max(5, max_frames * 0.01))
        ax_waist.set_ylim(-20, 20)
        ax_waist.set_xlabel('Time (s)')
        ax_waist.set_ylabel('Position')
        ax_waist.set_title('Waist & Leg (Trained)')
        ax_waist.legend(loc='upper right')
        ax_waist.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render), interval=33, blit=False)

    writer = FFMpegWriter(fps=30, bitrate=3000)
    anim.save(output_path, writer=writer)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/model_4limb_modal.zip'

    print(f"Analyzing model: {model_path}")

    # Create comparison plots
    baseline_traj, trained_traj = create_comparison_plots(model_path)

    # Create progression video
    create_progression_video(model_path)

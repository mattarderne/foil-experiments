"""
Evaluation and visualization script for trained Pump Foil agents.

Loads a trained model and runs episodes with optional visualization.
"""

import argparse
import os
from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from stable_baselines3 import PPO, SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Only random policy available.")

from foil_env import PumpFoilEnv


def run_episode(
    env: PumpFoilEnv,
    model=None,
    render: bool = False,
    deterministic: bool = True,
    max_steps: int = 3000,
) -> Dict[str, Any]:
    """
    Run a single episode and collect statistics.

    Args:
        env: PumpFoilEnv instance
        model: Trained model (None for random policy)
        render: Whether to render
        deterministic: Use deterministic actions (for trained model)
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with episode statistics
    """
    obs, info = env.reset()

    # History for analysis
    history = {
        'z': [], 'vx': [], 'vz': [],
        'theta': [], 'alpha': [],
        'power': [], 'thrust': [],
        'action_leg': [], 'action_pitch': [],
        'reward': [],
    }

    total_reward = 0
    steps = 0

    while steps < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Record history
        history['z'].append(info['z'])
        history['vx'].append(info['vx'])
        history['vz'].append(info['vz'])
        history['theta'].append(info['theta_deg'])
        history['alpha'].append(info['alpha_deg'])
        history['power'].append(info['power'])
        history['thrust'].append(info['thrust'])
        history['action_leg'].append(action[0])
        history['action_pitch'].append(action[1])
        history['reward'].append(reward)

        if render:
            env.render()

        if terminated or truncated:
            break

    # Convert to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    # Calculate statistics
    stats = {
        'total_reward': total_reward,
        'steps': steps,
        'mean_altitude': np.mean(history['z']),
        'std_altitude': np.std(history['z']),
        'mean_velocity': np.mean(history['vx']),
        'std_velocity': np.std(history['vx']),
        'total_energy': np.sum(history['power']) * 0.01,  # Energy in Joules (dt=0.01)
        'mean_power': np.mean(history['power']),
        'max_power': np.max(history['power']),
        'termination_reason': info.get('termination_reason', 'timeout'),
        'history': history,
    }

    return stats


def plot_episode(stats: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create detailed plots of an episode.

    Args:
        stats: Statistics dictionary from run_episode
        save_path: Path to save figure (None for display)
    """
    h = stats['history']
    t = np.arange(len(h['z'])) * 0.01  # Time in seconds

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # Altitude
    ax = axes[0, 0]
    ax.plot(t, h['z'], 'b-', linewidth=1)
    ax.axhline(y=0.15, color='g', linestyle='--', alpha=0.5, label='Target')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Surface')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(t, h['vx'], 'b-', linewidth=1)
    ax.axhline(y=4.5, color='g', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pitch angle
    ax = axes[1, 0]
    ax.plot(t, h['theta'], 'b-', linewidth=1, label='Pitch')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Pitch Angle')
    ax.grid(True, alpha=0.3)

    # Angle of attack
    ax = axes[1, 1]
    ax.plot(t, h['alpha'], 'orange', linewidth=1)
    ax.axhline(y=13, color='r', linestyle='--', alpha=0.5, label='Stall')
    ax.axhline(y=-13, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AoA (deg)')
    ax.set_title('Angle of Attack')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power
    ax = axes[2, 0]
    ax.plot(t, h['power'], 'orange', linewidth=1)
    ax.axhline(y=400, color='r', linestyle='--', alpha=0.5, label='Max sustainable')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title(f'Power Output (Mean: {stats["mean_power"]:.0f}W)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Thrust
    ax = axes[2, 1]
    ax.plot(t, h['thrust'], 'g-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Net Thrust (Positive = Forward)')
    ax.grid(True, alpha=0.3)

    # Actions
    ax = axes[3, 0]
    ax.plot(t, h['action_leg'], 'b-', linewidth=1, alpha=0.7, label='Leg Force')
    ax.plot(t, h['action_pitch'], 'r-', linewidth=1, alpha=0.7, label='Pitch Torque')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action (normalized)')
    ax.set_title('Control Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[3, 1]
    ax.plot(t, np.cumsum(h['reward']), 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Cumulative Reward (Total: {stats["total_reward"]:.1f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_pumping_pattern(stats: Dict[str, Any]):
    """
    Analyze the learned pumping pattern.

    Args:
        stats: Statistics dictionary from run_episode
    """
    h = stats['history']

    # FFT analysis of altitude oscillations
    z_centered = h['z'] - np.mean(h['z'])
    if len(z_centered) > 100:
        fft = np.fft.fft(z_centered)
        freqs = np.fft.fftfreq(len(z_centered), d=0.01)

        # Only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power = np.abs(fft[pos_mask]) ** 2

        # Find dominant frequency
        peak_idx = np.argmax(power)
        dominant_freq = freqs_pos[peak_idx]
        amplitude = np.std(h['z']) * 2  # Approximate peak-to-peak

        print("\n=== Pumping Pattern Analysis ===")
        print(f"Dominant frequency: {dominant_freq:.2f} Hz ({dominant_freq * 60:.0f} pumps/min)")
        print(f"Altitude amplitude: {amplitude * 100:.1f} cm (peak-to-peak)")
        print(f"Mean altitude: {np.mean(h['z']) * 100:.1f} cm")
        print(f"Altitude std: {np.std(h['z']) * 100:.1f} cm")

        # Analyze phase relationship between actions
        if len(h['action_leg']) > 100:
            # Cross-correlation between leg and pitch actions
            leg_centered = h['action_leg'] - np.mean(h['action_leg'])
            pitch_centered = h['action_pitch'] - np.mean(h['action_pitch'])

            corr = np.correlate(leg_centered, pitch_centered, mode='full')
            lag = np.argmax(corr) - len(leg_centered)
            phase_deg = lag * 0.01 * dominant_freq * 360

            print(f"Leg-Pitch phase offset: {phase_deg:.0f} degrees")


def evaluate(
    model_path: Optional[str] = None,
    n_episodes: int = 5,
    render: bool = False,
    deterministic: bool = True,
    save_plots: bool = False,
    use_rk4: bool = False,
):
    """
    Evaluate a trained model or random policy.

    Args:
        model_path: Path to trained model (None for random policy)
        n_episodes: Number of episodes to run
        render: Whether to render
        deterministic: Use deterministic actions
        save_plots: Save plots to files
        use_rk4: Use RK4 integration
    """
    # Create environment
    render_mode = "human" if render else None
    env = PumpFoilEnv(render_mode=render_mode, use_rk4=use_rk4)

    # Load model
    model = None
    if model_path is not None:
        if not HAS_SB3:
            print("Error: stable-baselines3 required to load model")
            return

        # Determine algorithm from path
        if "ppo" in model_path.lower():
            model = PPO.load(model_path)
        elif "sac" in model_path.lower():
            model = SAC.load(model_path)
        else:
            # Try PPO first
            try:
                model = PPO.load(model_path)
            except:
                model = SAC.load(model_path)

        print(f"Loaded model from: {model_path}")
    else:
        print("Using random policy")

    # Run episodes
    all_stats = []
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
        stats = run_episode(env, model, render=render, deterministic=deterministic)
        all_stats.append(stats)

        print(f"  Steps: {stats['steps']}")
        print(f"  Total Reward: {stats['total_reward']:.1f}")
        print(f"  Mean Altitude: {stats['mean_altitude']:.3f}m")
        print(f"  Mean Velocity: {stats['mean_velocity']:.2f}m/s")
        print(f"  Mean Power: {stats['mean_power']:.0f}W")
        print(f"  Termination: {stats['termination_reason']}")

        if save_plots:
            plot_episode(stats, save_path=f"episode_{ep + 1}.png")

        # Analyze pumping pattern for longer episodes
        if stats['steps'] > 500:
            analyze_pumping_pattern(stats)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    rewards = [s['total_reward'] for s in all_stats]
    steps = [s['steps'] for s in all_stats]
    power = [s['mean_power'] for s in all_stats]

    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Mean Steps: {np.mean(steps):.0f} +/- {np.std(steps):.0f}")
    print(f"Mean Power: {np.mean(power):.0f}W")

    # Show single episode plot
    if not save_plots and n_episodes > 0:
        plot_episode(all_stats[-1])

    env.close()
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Pump Foil agent")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained model (omit for random policy)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default: deterministic)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of displaying"
    )
    parser.add_argument(
        "--use-rk4",
        action="store_true",
        help="Use RK4 integration instead of Euler"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        save_plots=args.save_plots,
        use_rk4=args.use_rk4,
    )


if __name__ == "__main__":
    main()

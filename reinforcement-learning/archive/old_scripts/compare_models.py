#!/usr/bin/env python3
"""
Compare the 5-action waist model against baseline to understand why it's not pumping.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import the 5-limb environment
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_episode(env, model=None, max_steps=1000, deterministic=True):
    """Run one episode, return trajectory data."""
    obs, info = env.reset()

    data = {
        'z': [], 'vx': [], 'vz': [], 'theta': [],
        'left_leg': [], 'right_leg': [], 'waist': [],
        'left_arm': [], 'right_arm': [],
        'actions': [], 'thrust': [], 'leg_force': [],
    }

    done = False
    step = 0
    while not done and step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            # Baseline: no action (just glide)
            action = np.zeros(5)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data['z'].append(env.state.z)
        data['vx'].append(env.state.vx)
        data['vz'].append(env.state.vz)
        data['theta'].append(np.degrees(env.state.theta))
        data['left_leg'].append(env.left_leg_pos * 100)  # cm
        data['right_leg'].append(env.right_leg_pos * 100)  # cm
        data['waist'].append(np.degrees(env.waist_pos))
        data['left_arm'].append(np.degrees(env.left_arm_pos))
        data['right_arm'].append(np.degrees(env.right_arm_pos))
        data['actions'].append(action.copy())
        data['thrust'].append(info.get('thrust', 0))
        data['leg_force'].append(info.get('leg_force', 0))

        step += 1

    for k in data:
        if k != 'actions':
            data[k] = np.array(data[k])
        else:
            data[k] = np.array(data[k])

    data['termination_reason'] = info.get('termination_reason', 'max_steps')
    data['duration'] = step * 0.01  # seconds

    return data


def analyze_pumping(data):
    """Analyze pumping behavior in trajectory."""
    # Leg statistics
    left_leg = data['left_leg']
    right_leg = data['right_leg']
    avg_leg = (left_leg + right_leg) / 2

    leg_amplitude = (np.max(avg_leg) - np.min(avg_leg)) / 2
    leg_std = np.std(avg_leg)
    leg_mean = np.mean(avg_leg)

    # Estimate frequency via zero crossings
    mean_centered = avg_leg - np.mean(avg_leg)
    zero_crossings = np.where(np.diff(np.sign(mean_centered)))[0]
    if len(zero_crossings) > 2:
        periods = np.diff(zero_crossings) * 0.01  # dt = 0.01
        freq = 1 / (2 * np.mean(periods))  # half-period between crossings
    else:
        freq = 0.0

    # Waist statistics
    waist = data['waist']
    waist_std = np.std(waist)
    waist_mean = np.mean(waist)

    # Velocity and thrust
    vx_loss = data['vx'][0] - data['vx'][-1]
    mean_thrust = np.mean(data['thrust'])

    # vz amplitude (pumping effect)
    vz = data['vz']
    vz_amplitude = (np.max(vz) - np.min(vz)) / 2

    return {
        'duration': data['duration'],
        'termination': data['termination_reason'],
        'vx_loss': vx_loss,
        'mean_thrust': mean_thrust,
        'leg_amplitude': leg_amplitude,
        'leg_std': leg_std,
        'leg_mean': leg_mean,
        'leg_freq': freq,
        'waist_std': waist_std,
        'waist_mean': waist_mean,
        'vz_amplitude': vz_amplitude,
    }


def main():
    print("="*60)
    print("COMPARISON: Trained 5-Action Model vs Passive Glide")
    print("="*60)

    env = PumpFoilEnv4Limb()

    # Load trained model
    try:
        model = PPO.load("checkpoints/model_4limb_modal.zip")
        print("\nLoaded trained model (5-action waist)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run trained model
    print("\n--- TRAINED MODEL ---")
    trained_data = run_episode(env, model, max_steps=3000)
    trained_stats = analyze_pumping(trained_data)

    print(f"Duration: {trained_stats['duration']:.1f}s")
    print(f"Termination: {trained_stats['termination']}")
    print(f"Velocity loss: {trained_stats['vx_loss']:.2f} m/s")
    print(f"Mean thrust: {trained_stats['mean_thrust']:.1f} N")
    print(f"\nLeg motion:")
    print(f"  Amplitude: {trained_stats['leg_amplitude']:.1f} cm")
    print(f"  Std dev: {trained_stats['leg_std']:.1f} cm")
    print(f"  Mean: {trained_stats['leg_mean']:.1f} cm")
    print(f"  Frequency: {trained_stats['leg_freq']:.2f} Hz")
    print(f"\nWaist:")
    print(f"  Mean: {trained_stats['waist_mean']:.1f}°")
    print(f"  Std dev: {trained_stats['waist_std']:.1f}°")
    print(f"\nVertical oscillation:")
    print(f"  vz amplitude: {trained_stats['vz_amplitude']:.3f} m/s")

    # Run passive glide (no action)
    print("\n--- PASSIVE GLIDE (no action) ---")
    passive_data = run_episode(env, None, max_steps=3000)
    passive_stats = analyze_pumping(passive_data)

    print(f"Duration: {passive_stats['duration']:.1f}s")
    print(f"Termination: {passive_stats['termination']}")
    print(f"Velocity loss: {passive_stats['vx_loss']:.2f} m/s")
    print(f"Mean thrust: {passive_stats['mean_thrust']:.1f} N")

    # Plot comparison
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    t_trained = np.arange(len(trained_data['z'])) * 0.01
    t_passive = np.arange(len(passive_data['z'])) * 0.01

    # Altitude
    ax = axes[0, 0]
    ax.plot(t_trained, trained_data['z'] * 100, 'b-', label='Trained')
    ax.plot(t_passive, passive_data['z'] * 100, 'r--', label='Passive')
    ax.axhline(15, color='g', linestyle=':', alpha=0.5, label='Target')
    ax.set_ylabel('Altitude (cm)')
    ax.set_title('Altitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(t_trained, trained_data['vx'], 'b-', label='Trained')
    ax.plot(t_passive, passive_data['vx'], 'r--', label='Passive')
    ax.axhline(4.5, color='g', linestyle=':', alpha=0.5, label='Target')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Leg position (trained only)
    ax = axes[1, 0]
    ax.plot(t_trained, trained_data['left_leg'], 'b-', label='Left leg', alpha=0.7)
    ax.plot(t_trained, trained_data['right_leg'], 'r-', label='Right leg', alpha=0.7)
    avg_leg = (trained_data['left_leg'] + trained_data['right_leg']) / 2
    ax.plot(t_trained, avg_leg, 'k-', linewidth=2, label='Average')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(15, color='gray', linestyle='--', alpha=0.3, label='Max')
    ax.axhline(-15, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Leg position (cm)')
    ax.set_title('Leg Motion (Trained Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Waist position (trained only)
    ax = axes[1, 1]
    ax.plot(t_trained, trained_data['waist'], 'purple', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(17.2, color='gray', linestyle='--', alpha=0.3, label='Max (±17°)')
    ax.axhline(-17.2, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Waist lean (°)')
    ax.set_title('Waist Position (Trained Model)')
    ax.grid(True, alpha=0.3)

    # Actions distribution (trained)
    ax = axes[2, 0]
    actions = trained_data['actions']
    action_labels = ['L Leg', 'R Leg', 'L Arm', 'R Arm', 'Waist']
    bp = ax.boxplot([actions[:, i] for i in range(5)], labels=action_labels)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Action value [-1, 1]')
    ax.set_title('Action Distribution (Trained Model)')
    ax.grid(True, alpha=0.3)

    # Thrust comparison
    ax = axes[2, 1]
    ax.plot(t_trained, trained_data['thrust'], 'b-', label='Trained', alpha=0.7)
    ax.plot(t_passive, passive_data['thrust'], 'r--', label='Passive', alpha=0.7)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Net Thrust')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to model_comparison.png")

    # Print diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if trained_stats['leg_std'] < 2.0:
        print("\n⚠️  LEGS BARELY MOVING!")
        print(f"   Leg std is only {trained_stats['leg_std']:.1f} cm (should be >5 cm for pumping)")

    if abs(trained_stats['waist_mean']) > 10:
        print("\n⚠️  WAIST STUCK AT LIMIT!")
        print(f"   Waist mean is {trained_stats['waist_mean']:.1f}° (limit is ±17°)")

    if trained_stats['mean_thrust'] < 0:
        print("\n⚠️  NEGATIVE THRUST!")
        print(f"   Mean thrust is {trained_stats['mean_thrust']:.1f} N (agent is slowing down)")

    if trained_stats['vz_amplitude'] < 0.05:
        print("\n⚠️  NO VERTICAL OSCILLATION!")
        print(f"   vz amplitude is {trained_stats['vz_amplitude']:.3f} m/s (need >0.1 for pumping)")

    trained_improvement = passive_stats['duration'] / trained_stats['duration'] if trained_stats['duration'] > 0 else 0
    print(f"\n📊 Trained survived {trained_stats['duration']:.1f}s vs passive {passive_stats['duration']:.1f}s")
    print(f"   That's {trained_stats['duration']/passive_stats['duration']:.1f}x longer")

    print("\n💡 CONCLUSION:")
    if trained_stats['leg_std'] < 2.0 and trained_stats['vz_amplitude'] < 0.05:
        print("   The agent learned to BALANCE but NOT to PUMP.")
        print("   It survives by using waist lean for pitch control,")
        print("   but is essentially doing a slow glide with no thrust generation.")
        print("\n   To fix this, the reward function needs:")
        print("   - Stronger velocity maintenance incentive")
        print("   - Explicit reward for leg oscillation amplitude")
        print("   - Penalty for waist being stuck at limits")


if __name__ == "__main__":
    main()

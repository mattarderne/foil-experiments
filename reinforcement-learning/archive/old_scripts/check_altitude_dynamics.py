#!/usr/bin/env python3
"""
Check altitude dynamics: why does pumping cause crashes?
"""

import numpy as np
import matplotlib.pyplot as plt
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb
from stable_baselines3 import PPO


def run_and_record(env, model=None, action_fn=None, max_steps=500, name=""):
    """Run episode and record trajectory."""
    obs, info = env.reset()

    data = {'t': [], 'z': [], 'vz': [], 'theta': [], 'leg_force': [], 'thrust': []}

    done = False
    step = 0
    while not done and step < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif action_fn is not None:
            action = action_fn(step)
        else:
            action = np.zeros(5)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data['t'].append(step * 0.01)
        data['z'].append(env.state.z * 100)  # cm
        data['vz'].append(env.state.vz)
        data['theta'].append(np.degrees(env.state.theta))
        data['leg_force'].append(info.get('leg_force', 0))
        data['thrust'].append(info.get('thrust', 0))

        step += 1

    for k in data:
        data[k] = np.array(data[k])

    data['duration'] = step * 0.01
    data['termination'] = info.get('termination_reason', 'max_steps')
    data['name'] = name

    return data


def main():
    env = PumpFoilEnv4Limb()

    # Get limits
    cfg = env.physics.config
    ceiling = cfg['riding_depth'] * 100  # cm (foil breach)
    floor = (cfg['riding_depth'] - cfg['mast_length']) * 100  # cm (board touchdown)

    print(f"Altitude limits: ceiling={ceiling:.1f}cm, floor={floor:.1f}cm")
    print(f"Safe range: {floor:.1f} to {ceiling:.1f} cm ({ceiling-floor:.1f}cm total)\n")

    # Run different scenarios
    scenarios = []

    # 1. Passive glide
    scenarios.append(run_and_record(env, name="Passive glide"))

    # 2. Aggressive pumping
    def aggressive_pump(step):
        t = step * 0.01
        leg = 0.8 * np.sin(2 * np.pi * 2.0 * t)
        return np.array([leg, leg, 0, 0, 0])
    scenarios.append(run_and_record(env, action_fn=aggressive_pump, name="Aggressive pump (0.8)"))

    # 3. Gentle pumping
    def gentle_pump(step):
        t = step * 0.01
        leg = 0.3 * np.sin(2 * np.pi * 2.0 * t)
        return np.array([leg, leg, 0, 0, 0])
    scenarios.append(run_and_record(env, action_fn=gentle_pump, name="Gentle pump (0.3)"))

    # 4. Trained model
    try:
        model = PPO.load("checkpoints/model_4limb_modal.zip")
        scenarios.append(run_and_record(env, model=model, name="Trained model"))
    except:
        print("Could not load trained model")

    # Print summary
    print("="*70)
    print(f"{'Scenario':<25} | Duration | z range | Termination")
    print("-"*70)
    for data in scenarios:
        z_min, z_max = np.min(data['z']), np.max(data['z'])
        print(f"{data['name']:<25} | {data['duration']:6.1f}s | [{z_min:+5.1f}, {z_max:+5.1f}]cm | {data['termination']}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'purple']

    # Altitude
    ax = axes[0, 0]
    for data, color in zip(scenarios, colors):
        ax.plot(data['t'], data['z'], color=color, label=data['name'], alpha=0.8)
    ax.axhline(ceiling, color='black', linestyle='--', alpha=0.5, label='Ceiling (breach)')
    ax.axhline(floor, color='black', linestyle=':', alpha=0.5, label='Floor (touchdown)')
    ax.axhline(15, color='gray', linestyle='-.', alpha=0.3, label='Target')
    ax.set_ylabel('Altitude (cm)')
    ax.set_title('Altitude vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(floor - 5, ceiling + 5)

    # vz
    ax = axes[0, 1]
    for data, color in zip(scenarios, colors):
        ax.plot(data['t'], data['vz'], color=color, label=data['name'], alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Vertical velocity (m/s)')
    ax.set_title('vz vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Leg force
    ax = axes[1, 0]
    for data, color in zip(scenarios, colors):
        ax.plot(data['t'], data['leg_force'], color=color, label=data['name'], alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Leg force (N)')
    ax.set_title('Leg Force vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pitch
    ax = axes[1, 1]
    for data, color in zip(scenarios, colors):
        ax.plot(data['t'], data['theta'], color=color, label=data['name'], alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Pitch vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('altitude_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to altitude_dynamics.png")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print(f"""
Safe altitude band: {floor:.1f}cm to {ceiling:.1f}cm = {ceiling-floor:.1f}cm total
Target altitude: 15cm (in the middle of the safe band)

Key observations:
""")

    for data in scenarios:
        z_range = np.max(data['z']) - np.min(data['z'])
        print(f"{data['name']}:")
        print(f"  - Duration: {data['duration']:.1f}s")
        print(f"  - z oscillation: {z_range:.1f}cm")
        print(f"  - vs safe band: {z_range/(ceiling-floor)*100:.0f}% of available space")
        print()


if __name__ == "__main__":
    main()

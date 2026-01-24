"""
Compare 4-limb agent behavior at different training stages.
Shows progression of learned pumping technique.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_episode(model_path):
    """Run episode and return data."""
    model = PPO.load(model_path)
    env = PumpFoilEnv4Limb()

    obs, _ = env.reset(seed=42)  # Fixed seed for comparison
    done = False

    times, z_vals, vx_vals, vz_vals = [], [], [], []
    left_leg, right_leg = [], []

    t = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        times.append(t)
        z_vals.append(env.state.z)
        vx_vals.append(env.state.vx)
        vz_vals.append(env.state.vz)
        left_leg.append(env.left_leg_pos)
        right_leg.append(env.right_leg_pos)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += env.dt

    return {
        'times': np.array(times),
        'z': np.array(z_vals),
        'vx': np.array(vx_vals),
        'vz': np.array(vz_vals),
        'left_leg': np.array(left_leg),
        'right_leg': np.array(right_leg),
        'duration': times[-1],
        'final_vx': vx_vals[-1],
        'termination': info.get('termination_reason', 'unknown')
    }


def main():
    # Define stages to compare
    stages = [
        ('50k steps', 'checkpoints/pump_foil_4limb_20260118_073532/checkpoint_50000_steps.zip'),
        ('200k steps', 'checkpoints/pump_foil_4limb_20260118_073532/checkpoint_200000_steps.zip'),
        ('500k steps', 'checkpoints/pump_foil_4limb_20260118_073532/checkpoint_500000_steps.zip'),
        ('2M steps (Modal)', 'checkpoints/model_4limb_modal.zip'),
    ]

    # Run episodes for each stage
    print("Running episodes for each training stage...")
    results = {}
    for name, path in stages:
        print(f"  {name}...")
        try:
            results[name] = run_episode(path)
            print(f"    Duration: {results[name]['duration']:.2f}s, Final vx: {results[name]['final_vx']:.2f} m/s")
        except Exception as e:
            print(f"    Error: {e}")
            results[name] = None

    # Create comparison figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('Training Progression: 4-Limb Pump Foil Agent', fontsize=14)

    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

    for col, (name, path) in enumerate(stages):
        data = results.get(name)
        if data is None:
            continue

        color = colors[col]

        # Row 0: Altitude
        ax = axes[0, col]
        ax.plot(data['times'], data['z'], color=color, linewidth=1.5)
        ax.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
        ax.set_ylim(-0.6, 0.3)
        ax.set_ylabel('Altitude (m)' if col == 0 else '')
        ax.set_title(f'{name}\n{data["duration"]:.2f}s flight')
        ax.grid(True, alpha=0.3)

        # Row 1: Velocity
        ax = axes[1, col]
        ax.plot(data['times'], data['vx'], color='green', linewidth=1.5, label='vx')
        ax.plot(data['times'], data['vz'], color='blue', linewidth=1, alpha=0.7, label='vz')
        ax.set_ylim(-2, 6)
        ax.set_ylabel('Velocity (m/s)' if col == 0 else '')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Leg positions
        ax = axes[2, col]
        ax.plot(data['times'], data['left_leg'], 'b-', linewidth=1, label='Left')
        ax.plot(data['times'], data['right_leg'], 'r-', linewidth=1, label='Right')
        ax.set_ylim(-0.2, 0.2)
        ax.set_ylabel('Leg Position (m)' if col == 0 else '')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 3: Metrics summary
        ax = axes[3, col]
        ax.axis('off')
        metrics = f"""
Duration: {data['duration']:.2f}s
Final vx: {data['final_vx']:.2f} m/s
Δvx: {data['final_vx'] - 4.5:+.2f} m/s
Termination: {data['termination']}

Leg amplitude:
  L: {(data['left_leg'].max() - data['left_leg'].min())/2:.3f}m
  R: {(data['right_leg'].max() - data['right_leg'].min())/2:.3f}m
"""
        ax.text(0.1, 0.9, metrics, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add row labels
    for row, label in enumerate(['Altitude', 'Velocity', 'Leg Pumping', 'Metrics']):
        axes[row, 0].annotate(label, xy=(-0.3, 0.5), xycoords='axes fraction',
                              fontsize=11, fontweight='bold', rotation=90,
                              ha='center', va='center')

    plt.tight_layout()
    plt.savefig('training_progression.png', dpi=150, bbox_inches='tight')
    print('\nSaved: training_progression.png')

    # Summary table
    print('\n' + '='*60)
    print('TRAINING PROGRESSION SUMMARY')
    print('='*60)
    print(f'{"Stage":<20} {"Duration":>10} {"Final vx":>10} {"Δvx":>10}')
    print('-'*60)
    for name, _ in stages:
        data = results.get(name)
        if data:
            print(f'{name:<20} {data["duration"]:>10.2f}s {data["final_vx"]:>10.2f} {data["final_vx"]-4.5:>+10.2f}')
    print('='*60)


if __name__ == '__main__':
    main()

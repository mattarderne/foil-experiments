"""
Analyze trained 4-limb pump foil agent.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb
import os
import glob


def find_latest_model():
    """Find the most recent 4-limb model."""
    # First check for Modal model
    if os.path.exists("./checkpoints/model_4limb_modal.zip"):
        return "./checkpoints/model_4limb_modal.zip"

    checkpoints = glob.glob("./checkpoints/pump_foil_4limb_*/final_model.zip")
    if not checkpoints:
        # Try best_model
        checkpoints = glob.glob("./checkpoints/pump_foil_4limb_*/best_model.zip")
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None


def analyze_agent(model_path=None):
    """Run and analyze the trained 4-limb agent."""

    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No model found!")
            return

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = PumpFoilEnv4Limb()

    # Run episode
    obs, _ = env.reset()
    done = False

    # Recording
    times = []
    z_vals = []
    vx_vals = []
    vz_vals = []
    theta_vals = []
    left_leg = []
    right_leg = []
    left_arm = []
    right_arm = []
    actions_left_leg = []
    actions_right_leg = []
    actions_left_arm = []
    actions_right_arm = []
    rewards = []

    t = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Record state
        times.append(t)
        z_vals.append(env.state.z)
        vx_vals.append(env.state.vx)
        vz_vals.append(env.state.vz)
        theta_vals.append(np.degrees(env.state.theta))
        left_leg.append(env.left_leg_pos)
        right_leg.append(env.right_leg_pos)
        left_arm.append(env.left_arm_pos)
        right_arm.append(env.right_arm_pos)
        actions_left_leg.append(action[0])
        actions_right_leg.append(action[1])
        actions_left_arm.append(action[2])
        actions_right_arm.append(action[3])

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        t += env.dt

    times = np.array(times)

    # Create analysis figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"4-Limb Agent Analysis\nEpisode: {len(times)} steps ({times[-1]:.2f}s), Final vx: {vx_vals[-1]:.2f} m/s\nTermination: {info.get('termination_reason', 'unknown')}", fontsize=12)

    # Altitude
    ax = axes[0, 0]
    ax.plot(times, z_vals, 'b-', label='z')
    ax.axhline(y=0.2, color='r', linestyle='--', label='Target (0.2m)')
    ax.axhline(y=0.25, color='orange', linestyle=':', label='Ceiling')
    ax.axhline(y=-0.5, color='red', linestyle=':', label='Floor')
    ax.set_ylabel('Altitude (m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Altitude')

    # Velocity
    ax = axes[0, 1]
    ax.plot(times, vx_vals, 'g-', label='vx (forward)')
    ax.plot(times, vz_vals, 'b-', alpha=0.5, label='vz (vertical)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Velocity')

    # Pitch
    ax = axes[1, 0]
    ax.plot(times, theta_vals, 'purple')
    ax.set_ylabel('Pitch (deg)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Pitch Angle')

    # Rewards
    ax = axes[1, 1]
    ax.plot(times, rewards, 'orange')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Rewards (Total: {sum(rewards):.1f})')

    # Leg positions
    ax = axes[2, 0]
    ax.plot(times, left_leg, 'b-', label='Left leg')
    ax.plot(times, right_leg, 'r-', label='Right leg')
    ax.set_ylabel('Leg Position')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Leg Positions')

    # Arm positions
    ax = axes[2, 1]
    ax.plot(times, left_arm, 'b-', label='Left arm')
    ax.plot(times, right_arm, 'r-', label='Right arm')
    ax.set_ylabel('Arm Position')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Arm Positions')

    # Leg actions
    ax = axes[3, 0]
    ax.plot(times, actions_left_leg, 'b-', alpha=0.7, label='Left leg action')
    ax.plot(times, actions_right_leg, 'r-', alpha=0.7, label='Right leg action')
    ax.set_ylabel('Action')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Leg Actions (velocity commands)')

    # Arm actions
    ax = axes[3, 1]
    ax.plot(times, actions_left_arm, 'b-', alpha=0.7, label='Left arm action')
    ax.plot(times, actions_right_arm, 'r-', alpha=0.7, label='Right arm action')
    ax.set_ylabel('Action')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Arm Actions (velocity commands)')

    plt.tight_layout()
    plt.savefig('4limb_agent_analysis.png', dpi=150)
    print('Saved: 4limb_agent_analysis.png')

    # Summary stats
    print(f"\n=== Episode Summary ===")
    print(f"Duration: {times[-1]:.2f}s ({len(times)} steps)")
    print(f"Termination: {info.get('termination_reason', 'unknown')}")
    print(f"Final vx: {vx_vals[-1]:.2f} m/s (started at {vx_vals[0]:.2f})")
    print(f"Velocity change: {vx_vals[-1] - vx_vals[0]:.2f} m/s")
    print(f"Total reward: {sum(rewards):.1f}")

    # Check for oscillations
    print(f"\n=== Limb Motion Analysis ===")
    for name, data in [('Left leg', left_leg), ('Right leg', right_leg),
                       ('Left arm', left_arm), ('Right arm', right_arm)]:
        data = np.array(data)
        amp = (data.max() - data.min()) / 2
        print(f"{name}: range [{data.min():.2f}, {data.max():.2f}], amplitude {amp:.2f}")

    # FFT for frequency analysis
    if len(times) > 50:
        print(f"\n=== Frequency Analysis ===")
        from scipy import signal

        for name, data in [('Left leg', left_leg), ('Right leg', right_leg)]:
            data = np.array(data)
            # Remove DC
            data = data - data.mean()
            if data.std() > 0.01:  # Only if there's signal
                freqs = np.fft.rfftfreq(len(data), d=env.dt)
                fft = np.abs(np.fft.rfft(data))
                peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
                peak_freq = freqs[peak_idx]
                print(f"{name}: dominant frequency {peak_freq:.2f} Hz")


if __name__ == '__main__':
    analyze_agent()

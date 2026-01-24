"""Evaluate the curriculum-trained model for pumping behavior."""
import sys
sys.path.insert(0, '.')

import numpy as np
from stable_baselines3 import PPO
from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

# Training foil config
config = {'S_stab': 0.035, 'stab_angle': -4.0, 'S': 0.18, 'AR': 8}

# Load model
model = PPO.load("checkpoints/model_curriculum_pump.zip")

# Create env in phase 2 (full control)
env = PumpFoilEnvCurriculum(config=config, curriculum_phase=2)

print("=== CURRICULUM MODEL EVALUATION ===\n")

for trial in range(5):
    obs, _ = env.reset(seed=42 + trial)

    leg_positions = []
    leg_velocities = []

    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        leg_positions.append(env.left_leg_pos)
        leg_velocities.append(env.left_leg_vel)

    leg_positions = np.array(leg_positions)
    leg_velocities = np.array(leg_velocities)

    # Analyze pumping
    leg_range = np.max(leg_positions) - np.min(leg_positions)
    leg_range_pct = leg_range / (2 * env.MAX_LEG_EXTENSION) * 100

    # Estimate frequency via zero crossings
    zero_crossings = np.where(np.diff(np.signbit(leg_positions)))[0]
    if len(zero_crossings) >= 2:
        avg_half_period = np.mean(np.diff(zero_crossings)) * env.dt
        freq = 0.5 / avg_half_period if avg_half_period > 0 else 0
    else:
        freq = 0

    avg_leg_speed = np.mean(np.abs(leg_velocities))

    print(f"Trial {trial}: {env.time:.2f}s | "
          f"range={leg_range:.3f}m ({leg_range_pct:.0f}%) | "
          f"freq={freq:.2f}Hz | "
          f"avg_speed={avg_leg_speed:.2f}m/s | "
          f"reward={total_reward:.0f} | "
          f"{info.get('termination_reason', 'unknown')}")

print("\n=== COMPARISON TO TARGETS ===")
print(f"Target frequency: 2.0 Hz")
print(f"Target leg range: 100% (full extension)")
print(f"Target flight time: >10s")

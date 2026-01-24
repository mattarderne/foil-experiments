#!/usr/bin/env python3
"""
Debug script to understand why 5-action model can't generate thrust
while 2-action model could achieve near-neutral thrust.
"""

import numpy as np
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_manual_pumping(env, frequency=2.0, leg_amplitude=1.0, waist_amplitude=0.0,
                       pitch_bias=0.0, max_steps=1000):
    """Run with manual sinusoidal pumping pattern."""
    obs, info = env.reset()

    data = {
        'z': [], 'vx': [], 'vz': [], 'theta': [],
        'thrust': [], 'leg_force': [], 'pitch_torque': [],
        'leg_pos': [], 'waist_pos': [],
    }

    done = False
    step = 0
    while not done and step < max_steps:
        t = step * 0.01  # time in seconds
        phase = 2 * np.pi * frequency * t

        # Sinusoidal leg pumping (both legs in sync)
        leg_action = leg_amplitude * np.sin(phase)

        # Optional waist motion (opposite phase to legs for momentum transfer)
        waist_action = waist_amplitude * np.sin(phase + np.pi)

        # Optional pitch bias
        # Arms could be used for pitch, but let's use waist for now

        action = np.array([leg_action, leg_action, 0, 0, waist_action + pitch_bias])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data['z'].append(env.state.z)
        data['vx'].append(env.state.vx)
        data['vz'].append(env.state.vz)
        data['theta'].append(np.degrees(env.state.theta))
        data['thrust'].append(info.get('thrust', 0))
        data['leg_force'].append(info.get('leg_force', 0))
        data['pitch_torque'].append(info.get('pitch_torque', 0))
        data['leg_pos'].append((env.left_leg_pos + env.right_leg_pos) / 2 * 100)  # cm
        data['waist_pos'].append(np.degrees(env.waist_pos))

        step += 1

    for k in data:
        data[k] = np.array(data[k])

    data['duration'] = step * 0.01
    data['termination'] = info.get('termination_reason', 'max_steps')
    data['vx_loss'] = data['vx'][0] - data['vx'][-1]
    data['mean_thrust'] = np.mean(data['thrust'])

    return data


def main():
    env = PumpFoilEnv4Limb()

    print("="*60)
    print("PHYSICS DEBUG: Understanding Thrust Generation")
    print("="*60)

    # Test 1: Pure leg pumping at different frequencies
    print("\n--- Test 1: Pure Leg Pumping (no waist) ---")
    for freq in [1.5, 2.0, 2.5]:
        data = run_manual_pumping(env, frequency=freq, leg_amplitude=0.8, waist_amplitude=0)
        print(f"Freq {freq:.1f} Hz: {data['duration']:.1f}s, vx_loss: {data['vx_loss']:.2f} m/s, thrust: {data['mean_thrust']:.1f} N")

    # Test 2: Leg pumping with different amplitudes
    print("\n--- Test 2: Different Leg Amplitudes (2.0 Hz) ---")
    for amp in [0.4, 0.6, 0.8, 1.0]:
        data = run_manual_pumping(env, frequency=2.0, leg_amplitude=amp, waist_amplitude=0)
        print(f"Amp {amp:.1f}: {data['duration']:.1f}s, vx_loss: {data['vx_loss']:.2f} m/s, thrust: {data['mean_thrust']:.1f} N")

    # Test 3: Add waist motion (opposite phase)
    print("\n--- Test 3: Leg + Waist Motion (opposite phase) ---")
    for waist_amp in [0.0, 0.3, 0.6, 0.9]:
        data = run_manual_pumping(env, frequency=2.0, leg_amplitude=0.8, waist_amplitude=waist_amp)
        print(f"Waist amp {waist_amp:.1f}: {data['duration']:.1f}s, vx_loss: {data['vx_loss']:.2f} m/s, thrust: {data['mean_thrust']:.1f} N")

    # Test 4: Just glide (no pumping)
    print("\n--- Test 4: Passive Glide (baseline) ---")
    data = run_manual_pumping(env, frequency=2.0, leg_amplitude=0.0, waist_amplitude=0.0)
    print(f"Passive: {data['duration']:.1f}s, vx_loss: {data['vx_loss']:.2f} m/s, thrust: {data['mean_thrust']:.1f} N")

    # Test 5: Detailed look at best config
    print("\n--- Test 5: Detailed Analysis (2.0 Hz, 0.8 amp) ---")
    data = run_manual_pumping(env, frequency=2.0, leg_amplitude=0.8, waist_amplitude=0.0, max_steps=500)

    print(f"Duration: {data['duration']:.1f}s")
    print(f"Final vx: {data['vx'][-1]:.2f} m/s (started {data['vx'][0]:.2f})")
    print(f"Velocity loss: {data['vx_loss']:.2f} m/s")
    print(f"Mean thrust: {data['mean_thrust']:.1f} N")
    print(f"Thrust range: [{np.min(data['thrust']):.1f}, {np.max(data['thrust']):.1f}] N")
    print(f"Leg force range: [{np.min(data['leg_force']):.1f}, {np.max(data['leg_force']):.1f}] N")
    print(f"Pitch torque range: [{np.min(data['pitch_torque']):.1f}, {np.max(data['pitch_torque']):.1f}] Nm")
    print(f"vz range: [{np.min(data['vz']):.3f}, {np.max(data['vz']):.3f}] m/s")
    print(f"z range: [{np.min(data['z'])*100:.1f}, {np.max(data['z'])*100:.1f}] cm")
    print(f"theta range: [{np.min(data['theta']):.1f}, {np.max(data['theta']):.1f}] deg")

    # Key physics insight
    print("\n" + "="*60)
    print("KEY PHYSICS INSIGHT")
    print("="*60)

    # From CLAUDE.md: The 2-action model achieved -0.5 N thrust with optimal pumping
    # That was with pitch_action=0.40, leg_amplitude=0.7, frequency=2.5Hz
    # The key difference: the 2-action model directly controlled pitch torque

    print("""
The 2-action model (from CLAUDE.md) achieved:
  - Thrust: -0.5 N (near neutral!)
  - Velocity loss: -0.06 m/s
  - Duration: energy_exhausted (ran full duration)

It used: pitch_action=0.40, leg_amplitude=0.7, frequency=2.5Hz

Key differences from 5-action model:
1. 2-action had DIRECT pitch torque control
2. 5-action derives pitch from arm/waist accelerations (indirect)

The 5-action model's pitch control comes from:
  - Leg asymmetry (left vs right) → small torque
  - Arm swing acceleration → momentum transfer
  - Waist lean acceleration → momentum transfer
  - Waist lean CoM shift → static torque

But these are indirect effects - the agent must coordinate multiple
actions to achieve what the 2-action model did with one action.
    """)


if __name__ == "__main__":
    main()

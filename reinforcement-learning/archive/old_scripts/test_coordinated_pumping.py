#!/usr/bin/env python3
"""
Test coordinated pumping: use waist/arms for pitch control during leg pumping.
"""

import numpy as np
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb


def run_coordinated_pumping(env, frequency=2.0, leg_amplitude=0.8,
                             pitch_phase_offset=0.0, pitch_amplitude=0.5,
                             max_steps=3000):
    """
    Run with coordinated pumping:
    - Legs pump sinusoidally
    - Waist provides pitch torque with phase offset relative to legs
    """
    obs, info = env.reset()

    data = {'z': [], 'vx': [], 'vz': [], 'theta': [], 'thrust': []}

    done = False
    step = 0
    while not done and step < max_steps:
        t = step * 0.01
        phase = 2 * np.pi * frequency * t

        # Legs pump in sync
        leg_action = leg_amplitude * np.sin(phase)

        # Waist provides pitch control with phase offset
        # Phase offset determines when pitch torque is applied relative to leg motion
        waist_action = pitch_amplitude * np.sin(phase + pitch_phase_offset)

        action = np.array([leg_action, leg_action, 0, 0, waist_action])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data['z'].append(env.state.z)
        data['vx'].append(env.state.vx)
        data['vz'].append(env.state.vz)
        data['theta'].append(np.degrees(env.state.theta))
        data['thrust'].append(info.get('thrust', 0))

        step += 1

    for k in data:
        data[k] = np.array(data[k])

    data['duration'] = step * 0.01
    data['termination'] = info.get('termination_reason', 'max_steps')
    data['vx_loss'] = data['vx'][0] - data['vx'][-1]
    data['mean_thrust'] = np.mean(data['thrust'])
    data['z_range'] = np.max(data['z']) - np.min(data['z'])

    return data


def main():
    env = PumpFoilEnv4Limb()

    print("="*60)
    print("COORDINATED PUMPING: Testing Phase Relationships")
    print("="*60)
    print("\nHypothesis: Proper phase between leg pumping and pitch control")
    print("can prevent excessive altitude excursions while maintaining thrust.\n")

    # Test different phase offsets
    print("--- Phase Offset Sweep (2.0 Hz, leg=0.8, pitch=0.5) ---")
    print("Phase | Duration | vx_loss | Thrust | z_range | Termination")
    print("-" * 70)

    best_duration = 0
    best_config = None

    for phase_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        phase_rad = np.radians(phase_deg)
        data = run_coordinated_pumping(
            env, frequency=2.0, leg_amplitude=0.8,
            pitch_phase_offset=phase_rad, pitch_amplitude=0.5
        )
        print(f"{phase_deg:5d}° | {data['duration']:7.1f}s | {data['vx_loss']:+7.2f} | {data['mean_thrust']:+6.1f}N | {data['z_range']*100:5.1f}cm | {data['termination']}")

        if data['duration'] > best_duration:
            best_duration = data['duration']
            best_config = (phase_deg, data)

    print(f"\nBest: {best_config[0]}° with {best_duration:.1f}s duration")

    # Test different pitch amplitudes at best phase
    if best_config:
        best_phase = np.radians(best_config[0])
        print(f"\n--- Pitch Amplitude Sweep (phase={best_config[0]}°) ---")
        print("Pitch | Duration | vx_loss | Thrust | z_range | Termination")
        print("-" * 70)

        for pitch_amp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            data = run_coordinated_pumping(
                env, frequency=2.0, leg_amplitude=0.8,
                pitch_phase_offset=best_phase, pitch_amplitude=pitch_amp
            )
            print(f" {pitch_amp:4.1f} | {data['duration']:7.1f}s | {data['vx_loss']:+7.2f} | {data['mean_thrust']:+6.1f}N | {data['z_range']*100:5.1f}cm | {data['termination']}")

    # Test lower leg amplitudes (safer)
    print(f"\n--- Lower Leg Amplitude (safer pumping) ---")
    print("Leg | Duration | vx_loss | Thrust | z_range | Termination")
    print("-" * 70)

    for leg_amp in [0.3, 0.4, 0.5, 0.6, 0.7]:
        data = run_coordinated_pumping(
            env, frequency=2.0, leg_amplitude=leg_amp,
            pitch_phase_offset=np.radians(180), pitch_amplitude=0.3
        )
        print(f"{leg_amp:4.1f} | {data['duration']:7.1f}s | {data['vx_loss']:+7.2f} | {data['mean_thrust']:+6.1f}N | {data['z_range']*100:5.1f}cm | {data['termination']}")

    # Try optimal config from 2-action model
    print(f"\n--- Attempting 2-action 'optimal' config (freq=2.5, leg=0.7) ---")
    for pitch_amp in [0.0, 0.3, 0.5, 0.7]:
        data = run_coordinated_pumping(
            env, frequency=2.5, leg_amplitude=0.7,
            pitch_phase_offset=np.radians(180), pitch_amplitude=pitch_amp
        )
        print(f"pitch={pitch_amp:.1f} | {data['duration']:.1f}s | vx_loss={data['vx_loss']:+.2f} | thrust={data['mean_thrust']:+.1f}N | term={data['termination']}")


if __name__ == "__main__":
    main()

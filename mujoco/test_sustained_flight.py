"""
Test tuning for sustained flight (60+ seconds).

Key is balancing:
- Drag (velocity decreases)
- Pump thrust (from vertical oscillation)
- Lift (must equal weight)
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from foil_env.foil_physics import FoilPhysics, FoilState, CONFIG


def test_sustained_flight(
    pump_efficiency=0.30,
    pump_amplitude=0.08,
    pump_freq=2.0,
    initial_vx=5.0,
    S=0.22,
    Cd0=0.006,
    verbose=True
):
    """Test for sustained flight with tuned parameters."""
    if verbose:
        print(f"pump_eff={pump_efficiency}, amp={pump_amplitude}m, freq={pump_freq}Hz, "
              f"vx0={initial_vx}m/s, S={S}m², Cd0={Cd0}")

    mjcf = """
    <mujoco model="rigid_rider">
      <compiler angle="radian"/>
      <option timestep="0.01" gravity="0 0 -9.81"/>
      <worldbody>
        <body name="board" pos="0 0 0.2">
          <freejoint name="board_joint"/>
          <geom name="board_geom" type="box" size="0.7 0.25 0.025" mass="6"/>
          <geom name="rider" type="capsule" fromto="0 0 0.05  0 0 1.5"
                size="0.12" mass="80"/>
        </body>
      </worldbody>
    </mujoco>
    """

    mj_model = mujoco.MjModel.from_xml_string(mjcf)
    mj_data = mujoco.MjData(mj_model)

    board_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "board")

    foil_config = {**CONFIG, 'S': S, 'S_stab': 0.044, 'Cd0': Cd0, 'Cd_mast': 0.04}
    foil_physics = FoilPhysics(foil_config)

    # Find equilibrium pitch for this velocity
    W = 86 * 9.81
    theta_eq = 0.0635  # ~3.6° - approximate

    mj_data.qpos[2] = 0.0
    mj_data.qvel[0] = initial_vx
    mj_data.qpos[3] = np.cos(theta_eq / 2)
    mj_data.qpos[5] = np.sin(theta_eq / 2)

    mujoco.mj_forward(mj_model, mj_data)

    # Control gains
    base_pitch = theta_eq
    Kp_alt = 2.0
    Kd_alt = 1.0
    Kp_pitch = 500
    Kd_pitch = 200

    dt = mj_model.opt.timestep
    max_steps = 6000  # 60 seconds

    for step in range(max_steps):
        t = step * dt

        x, z = mj_data.qpos[0], mj_data.qpos[2]
        vx, vz = mj_data.qvel[0], mj_data.qvel[2]
        qw, qy = mj_data.qpos[3], mj_data.qpos[5]
        theta = 2 * np.arctan2(qy, qw)
        omega = mj_data.qvel[4]

        state = FoilState(x=x, z=z, vx=vx, vz=vz, theta=theta, omega=omega)
        forces = foil_physics.calculate_forces(state)

        # Pump thrust from vertical velocity
        if abs(vz) > 0.1:
            pump_thrust = pump_efficiency * abs(vz) * abs(forces.L)
        else:
            pump_thrust = 0

        Fx = forces.F_hydro_x + pump_thrust
        Fz = forces.F_hydro_z

        # Oscillating target altitude
        target_z = pump_amplitude * np.sin(2 * np.pi * pump_freq * t)
        target_vz = 2 * np.pi * pump_freq * pump_amplitude * np.cos(2 * np.pi * pump_freq * t)

        # Altitude control
        z_error = target_z - z
        vz_error = target_vz - vz
        target_pitch = base_pitch + Kp_alt * z_error + Kd_alt * vz_error
        target_pitch = np.clip(target_pitch, np.radians(-5), np.radians(15))

        # Pitch control
        pitch_error = theta - target_pitch
        My = -Kp_pitch * pitch_error - Kd_pitch * omega

        mj_data.xfrc_applied[board_id] = [Fx, 0, Fz, 0, My, 0]
        mujoco.mj_step(mj_model, mj_data)

        if verbose and step % 500 == 0:
            print(f"t={t:.1f}s: z={z:.3f}m, vx={vx:.2f}m/s, "
                  f"pitch={np.degrees(theta):.1f}°, pump={pump_thrust:.0f}N, D={forces.D:.0f}N")

        if z > 0.3:
            if verbose:
                print(f"[BREACH] at t={step*dt:.2f}s")
            return step * dt
        if z < -0.5:
            if verbose:
                print(f"[TOUCHDOWN] at t={step*dt:.2f}s")
            return step * dt
        if vx < 1.5:
            if verbose:
                print(f"[STALL] at t={step*dt:.2f}s")
            return step * dt

    if verbose:
        print(f"[SUCCESS] Survived 60.0s!")
    return max_steps * dt


def grid_search():
    """Search for parameters that achieve sustained flight."""
    print("=" * 70)
    print("GRID SEARCH FOR SUSTAINED FLIGHT PARAMETERS")
    print("=" * 70)

    best_time = 0
    best_params = None

    for pump_eff in [0.25, 0.30, 0.35]:
        for amp in [0.06, 0.08, 0.10]:
            for freq in [1.5, 2.0, 2.5]:
                for vx0 in [4.5, 5.0, 5.5]:
                    t = test_sustained_flight(
                        pump_efficiency=pump_eff,
                        pump_amplitude=amp,
                        pump_freq=freq,
                        initial_vx=vx0,
                        verbose=False
                    )
                    if t > best_time:
                        best_time = t
                        best_params = (pump_eff, amp, freq, vx0)
                        print(f"NEW BEST: {t:.1f}s | eff={pump_eff}, amp={amp}, freq={freq}, vx0={vx0}")

    print(f"\nBest result: {best_time:.1f}s")
    print(f"Best params: eff={best_params[0]}, amp={best_params[1]}, "
          f"freq={best_params[2]}, vx0={best_params[3]}")

    # Run best params with verbose
    print("\n" + "=" * 70)
    print("RUNNING BEST PARAMETERS")
    print("=" * 70)
    test_sustained_flight(
        pump_efficiency=best_params[0],
        pump_amplitude=best_params[1],
        pump_freq=best_params[2],
        initial_vx=best_params[3],
        verbose=True
    )


def test_reduced_drag():
    """Test with reduced drag (modern carbon foil)."""
    print("\n" + "=" * 70)
    print("TESTING WITH REDUCED DRAG (modern carbon foil)")
    print("=" * 70)

    return test_sustained_flight(
        pump_efficiency=0.30,
        pump_amplitude=0.08,
        pump_freq=2.0,
        initial_vx=5.0,
        Cd0=0.004,  # Ultra-low drag
        verbose=True
    )


if __name__ == "__main__":
    print("Testing baseline...")
    t1 = test_sustained_flight(verbose=True)

    t2 = test_reduced_drag()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline: {t1:.1f}s")
    print(f"Reduced drag: {t2:.1f}s")

    # Run grid search if baseline is poor
    if t1 < 30:
        grid_search()

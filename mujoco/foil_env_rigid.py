"""
Simplified MuJoCo Pump Foil Environment with RIGID rider.

Uses a single rigid body for the rider (no articulation) to verify
that the physics simulation works correctly before adding complexity.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from foil_env.foil_physics import FoilPhysics, FoilState, CONFIG


@dataclass
class RigidFoilConfig:
    """Configuration for the rigid foil environment."""
    # Foil physics (sized for 86kg model)
    S: float = 0.22
    S_stab: float = 0.044
    AR: float = 8.0
    stab_angle: float = -1.0
    Cd0: float = 0.006
    Cd_mast: float = 0.04
    pump_thrust_efficiency: float = 0.30

    # Environment
    max_episode_steps: int = 6000
    initial_velocity: float = 4.5
    initial_height: float = 0.0
    initial_pitch: float = 0.0635  # ~3.64°

    # Termination
    breach_height: float = 0.3
    touchdown_depth: float = -0.5
    stall_velocity: float = 1.5

    # Pitch control (high gains for fast tracking)
    pitch_Kp: float = 1000.0
    pitch_Kd: float = 300.0

    # Altitude target
    altitude_target: float = 0.0
    velocity_target: float = 4.5


class PumpFoilEnvRigid(gym.Env):
    """Pump foil environment with rigid rider."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, config: Optional[RigidFoilConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or RigidFoilConfig()
        self.render_mode = render_mode

        # Load rigid model
        model_path = Path(__file__).parent / "rider_rigid.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.board_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "board")

        # Initialize foil physics
        foil_config = {
            **CONFIG,
            'S': self.config.S,
            'S_stab': self.config.S_stab,
            'AR': self.config.AR,
            'stab_angle': self.config.stab_angle,
            'Cd0': self.config.Cd0,
            'Cd_mast': self.config.Cd_mast,
        }
        self.foil_physics = FoilPhysics(foil_config)

        # Simple action space for testing (not used with rigid rider)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos[2] = self.config.initial_height
        self.mj_data.qvel[0] = self.config.initial_velocity

        pitch = self.config.initial_pitch
        self.mj_data.qpos[3] = np.cos(pitch / 2)
        self.mj_data.qpos[5] = np.sin(pitch / 2)

        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        state = self._get_state()
        forces = self.foil_physics.calculate_forces(state)

        # Pump thrust from vertical oscillation
        # Active when pumping (vz significant) and altitude near nominal
        pump_thrust = 0.0
        if abs(state.vz) > 0.1 and abs(state.z) < 0.20:
            pump_thrust = self.config.pump_thrust_efficiency * abs(state.vz) * abs(forces.L)
            pump_thrust = min(pump_thrust, 50.0)

        Fx = forces.F_hydro_x + pump_thrust
        Fz = forces.F_hydro_z

        # Oscillating altitude target to induce pumping motion
        t = self.steps * 0.01  # Current time
        pump_freq = 1.0  # Hz (slower for better tracking)
        pump_amp = 0.05  # meters
        osc_target_z = pump_amp * np.sin(2 * np.pi * pump_freq * t)
        osc_target_vz = 2 * np.pi * pump_freq * pump_amp * np.cos(2 * np.pi * pump_freq * t)

        # Altitude control: track oscillating target
        base_pitch = self.config.initial_pitch
        z_error = osc_target_z - state.z
        vz_error = osc_target_vz - state.vz

        Kp_alt = 2.0  # rad per meter
        Kd_alt = 1.0  # rad per m/s
        target_pitch = base_pitch + Kp_alt * z_error + Kd_alt * vz_error
        target_pitch = np.clip(target_pitch, np.radians(-5), np.radians(15))

        # Pitch control to track target pitch
        pitch_error = state.theta - target_pitch
        My = -self.config.pitch_Kp * pitch_error - self.config.pitch_Kd * state.omega

        self.mj_data.xfrc_applied[self.board_id] = [Fx, 0, Fz, 0, My, 0]
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.steps += 1

        state = self._get_state()
        terminated, reason = self._check_termination(state)

        reward = 0.1  # Survival
        reward -= abs(state.z - self.config.altitude_target) * 0.5
        reward -= abs(state.vx - self.config.velocity_target) * 0.2

        info = {
            'z': state.z, 'vx': state.vx, 'vz': state.vz,
            'pitch': np.degrees(state.theta), 'termination': reason,
            'pump_thrust': pump_thrust, 'Fz': Fz, 'My': My,
            'target_pitch': np.degrees(target_pitch),
        }

        truncated = self.steps >= self.config.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_state(self):
        x, z = self.mj_data.qpos[0], self.mj_data.qpos[2]
        vx, vz = self.mj_data.qvel[0], self.mj_data.qvel[2]
        qw, qy = self.mj_data.qpos[3], self.mj_data.qpos[5]
        theta = 2 * np.arctan2(qy, qw)
        omega = self.mj_data.qvel[4]
        return FoilState(x=x, z=z, vx=vx, vz=vz, theta=theta, omega=omega)

    def _get_obs(self):
        state = self._get_state()
        return np.array([state.z, state.vz, state.theta, state.omega, state.vx], dtype=np.float32)

    def _check_termination(self, state):
        if state.z >= self.config.breach_height:
            return True, "foil_breached"
        if state.z <= self.config.touchdown_depth:
            return True, "board_touchdown"
        if state.vx < self.config.stall_velocity:
            return True, "stalled_velocity"
        return False, None

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.mj_data)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()


def test_rigid_env():
    """Test the rigid environment."""
    print("Testing RIGID MuJoCo environment...")
    print("=" * 60)

    env = PumpFoilEnvRigid()
    obs, _ = env.reset()

    print(f"Initial: z={obs[0]:.3f}m, vx={obs[4]:.2f}m/s, pitch={np.degrees(obs[2]):.1f}°")
    print()

    for step in range(6000):  # 60 seconds
        action = np.zeros(3)  # No action
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 500 == 0:
            print(f"t={step*0.01:.1f}s: z={info['z']:.3f}m, vx={info['vx']:.2f}m/s, "
                  f"pitch={info['pitch']:.1f}° (target {info['target_pitch']:.1f}°), "
                  f"Fz={info['Fz']:.0f}N")

        if terminated:
            print(f"\nTerminated: {info['termination']}")
            break

    print(f"\nFinal duration: {step*0.01:.2f}s")
    env.close()


if __name__ == "__main__":
    test_rigid_env()

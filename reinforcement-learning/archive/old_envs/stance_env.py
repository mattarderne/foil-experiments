"""
Stance-Only Environment for learning optimal rider position.

Single action: stance offset (forward/back position on board).
No body movement - pure passive glide with stance adjustment.
Goal: Learn optimal stance position to maximize glide duration.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from foil_env.foil_physics import FoilPhysics, FoilState, ForceOutput, CONFIG, compute_trim_angle


class StanceEnv(gym.Env):
    """
    Stance-only environment.

    Action Space (1 dimension):
        [stance_offset] in [-1, 1] -> maps to [-0.20, +0.20] meters

    Observation Space (7 dimensions):
        [z, vx, vz, theta, omega, stance_offset, is_stalled]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    MAX_STANCE_OFFSET = 0.20  # m - max forward/back from neutral
    TARGET_ALTITUDE = 0.15    # m
    MIN_VELOCITY = 1.5        # m/s
    MAX_STEPS = 500           # 5s at dt=0.01

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.base_config = config or {}

        # Action: stance offset only
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space
        obs_dim = 7
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces: Optional[ForceOutput] = None
        self.physics: Optional[FoilPhysics] = None
        self.stance_offset = 0.0
        self.step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0

        # Random initial stance (agent must learn to adjust)
        if self.np_random is not None:
            self.stance_offset = self.np_random.uniform(-0.10, 0.10)
        else:
            self.stance_offset = 0.0

        # Create physics with current stance
        config = {**self.base_config, 'rider_com_offset': self.stance_offset}
        self.physics = FoilPhysics(config)

        # Start at trim
        initial_vx = 4.5
        trim_theta = compute_trim_angle(initial_vx, config)

        self.state = FoilState(
            x=0.0,
            z=self.TARGET_ALTITUDE,
            vx=initial_vx,
            vz=0.0,
            theta=trim_theta,
            omega=0.0,
        )

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Update stance offset from action
        new_stance = float(action[0]) * self.MAX_STANCE_OFFSET

        # Smooth stance change (can't teleport instantly)
        max_stance_change = 0.02  # m per step (2cm per 0.01s = 2m/s max)
        stance_delta = np.clip(new_stance - self.stance_offset, -max_stance_change, max_stance_change)
        self.stance_offset += stance_delta
        self.stance_offset = np.clip(self.stance_offset, -self.MAX_STANCE_OFFSET, self.MAX_STANCE_OFFSET)

        # Update physics config with new stance
        self.physics.config['rider_com_offset'] = self.stance_offset

        # Physics step - NO leg force, NO pitch torque (passive glide)
        prev_vx = self.state.vx
        self.state = self.physics.step(self.state, leg_force=0, pitch_torque=0, dt=self.dt)
        self.forces = self.physics.calculate_forces(self.state)

        self.step_count += 1

        # Termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.MAX_STEPS

        # Reward: survive + maintain velocity
        reward = self._compute_reward(prev_vx, terminated, termination_reason)

        info = self._get_info()
        info["termination_reason"] = termination_reason

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        s = self.state
        return np.array([
            s.z / 0.5,                    # Normalized altitude
            s.vx / 5.0,                   # Normalized velocity
            s.vz / 2.0,                   # Normalized vertical velocity
            s.theta / 0.5,                # Normalized pitch
            s.omega / 2.0,                # Normalized pitch rate
            self.stance_offset / self.MAX_STANCE_OFFSET,  # Normalized stance
            1.0 if self.forces.is_stalled else 0.0,
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "z": self.state.z,
            "vx": self.state.vx,
            "vz": self.state.vz,
            "theta_deg": np.degrees(self.state.theta),
            "stance_offset": self.stance_offset,
            "is_stalled": self.forces.is_stalled,
            "M_weight": self.forces.M_weight,
            "step_count": self.step_count,
        }

    def _check_termination(self) -> Tuple[bool, str]:
        s = self.state
        cfg = self.physics.config

        riding_depth = cfg['riding_depth']
        mast_length = cfg['mast_length']

        if s.z >= riding_depth:
            return True, "foil_breached"
        if s.z <= riding_depth - mast_length:
            return True, "board_touchdown"
        if s.vx < self.MIN_VELOCITY:
            return True, "stalled_velocity"
        if abs(s.theta) > np.radians(45):
            return True, "excessive_pitch"

        return False, ""

    def _compute_reward(self, prev_vx: float, terminated: bool, reason: str) -> float:
        s = self.state

        # Survival reward
        reward = 1.0

        # Velocity maintenance bonus
        reward += s.vx / 5.0

        # Altitude stability bonus (near target)
        alt_error = abs(s.z - self.TARGET_ALTITUDE)
        reward += max(0, 0.5 - alt_error * 2)

        # Pitch stability bonus
        pitch_error = abs(s.theta)
        reward += max(0, 0.3 - pitch_error)

        # Termination penalty
        if terminated:
            reward -= 10.0

        return reward

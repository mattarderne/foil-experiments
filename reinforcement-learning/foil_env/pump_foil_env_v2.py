"""
Pump Foil Environment V2 - Improved pure RL with SOTA insights.

Key improvements:
1. Phase encoding: sin/cos of expected pump phase in observations
2. Jerk penalty: Smooth actions for periodic motion
3. Simplified reward: 5 well-chosen terms instead of 10+
4. Direct velocity control (not CPG constrained)

Based on curriculum env but streamlined for better learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, compute_trim_angle
from foil_env.body_model import BodyModel


class PumpFoilEnvV2(gym.Env):
    """
    Improved pure RL environment with phase encoding and jerk penalty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Physical constants
    RIDER_MASS = 70.0
    LEG_MASS = 15.0
    ARM_MASS = 5.0
    BOARD_MASS = 6.0

    MAX_LEG_EXTENSION = 0.15
    MAX_ARM_SWING = 1.5
    MAX_WAIST_ANGLE = 0.3
    MAX_LEG_VELOCITY = 2.0
    MAX_ARM_VELOCITY = 12.0
    MAX_WAIST_VELOCITY = 2.0

    MAX_POWER = 300.0
    MAX_ENERGY = 18000.0

    TARGET_ALTITUDE = 0.10
    TARGET_VELOCITY = 4.5
    MIN_VELOCITY = 1.5
    MAX_STEPS = 6000  # 60 seconds

    # Expected pump frequency for phase encoding
    PUMP_FREQUENCY = 2.5  # Hz

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.physics = FoilPhysics(config)
        self.body = BodyModel()

        # Action space: [left_leg, right_leg, left_arm, right_arm, waist] velocities
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation space:
        # 5 (foil) + 10 (limb pos/vel) + 3 (body) + 2 (phase) + 2 (goals) + 1 (energy) = 23
        obs_dim = 23
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces = None
        self.step_count = 0
        self.time = 0.0

        # Limb states
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = 0.0
        self.left_arm_vel = 0.0
        self.right_arm_pos = 0.0
        self.right_arm_vel = 0.0
        self.waist_pos = 0.1
        self.waist_vel = 0.0

        self.prev_hip_height = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

        # Action history for jerk penalty
        self.action_history = [np.zeros(5), np.zeros(5), np.zeros(5)]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        initial_velocity = options.get("initial_velocity", self.TARGET_VELOCITY)

        z_noise = self.np_random.uniform(-0.02, 0.02) if seed else 0
        vx_noise = self.np_random.uniform(-0.2, 0.2) if seed else 0

        actual_vx = initial_velocity + vx_noise
        trim_theta = compute_trim_angle(actual_vx, self.physics.config)

        self.state = FoilState(
            x=0.0,
            z=self.TARGET_ALTITUDE + z_noise,
            vx=actual_vx,
            vz=0.0,
            theta=trim_theta,
            omega=0.0,
        )

        # Reset limbs
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = 0.0
        self.left_arm_vel = 0.0
        self.right_arm_pos = 0.0
        self.right_arm_vel = 0.0
        self.waist_pos = 0.1
        self.waist_vel = 0.0

        # Reset action history
        self.action_history = [np.zeros(5), np.zeros(5), np.zeros(5)]

        self._update_body_model()
        self.prev_hip_height = self.body.get_hip_height()

        self.step_count = 0
        self.time = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None

        action = np.array(action, dtype=np.float32)

        # Update action history for jerk penalty
        self.action_history.pop(0)
        self.action_history.append(action.copy())

        # Parse action as direct velocity commands
        left_leg_vel_cmd = float(action[0]) * self.MAX_LEG_VELOCITY
        right_leg_vel_cmd = float(action[1]) * self.MAX_LEG_VELOCITY
        left_arm_vel_cmd = float(action[2]) * self.MAX_ARM_VELOCITY
        right_arm_vel_cmd = float(action[3]) * self.MAX_ARM_VELOCITY
        waist_vel_cmd = float(action[4]) * self.MAX_WAIST_VELOCITY

        prev_vx = self.state.vx
        prev_z = self.state.z

        # === LIMB DYNAMICS ===
        MAX_LEG_ACCEL = 15.0
        MAX_ARM_ACCEL = 30.0
        MAX_WAIST_ACCEL = 15.0

        def update_limb(pos, vel, vel_cmd, max_accel, max_ext, dt):
            accel = np.clip((vel_cmd - vel) / dt, -max_accel, max_accel)
            new_vel = vel + accel * dt
            new_pos = pos + new_vel * dt

            if new_pos > max_ext * 0.9:
                new_vel = min(0, new_vel * (1 - (new_pos - max_ext * 0.9) / (max_ext * 0.1)))
                new_pos = min(max_ext, new_pos)
            elif new_pos < -max_ext * 0.9:
                new_vel = max(0, new_vel * (1 - (-max_ext * 0.9 - new_pos) / (max_ext * 0.1)))
                new_pos = max(-max_ext, new_pos)

            actual_accel = (new_vel - vel) / dt
            return new_pos, new_vel, actual_accel

        # Update limbs
        self.left_leg_pos, self.left_leg_vel, left_leg_accel = update_limb(
            self.left_leg_pos, self.left_leg_vel, left_leg_vel_cmd,
            MAX_LEG_ACCEL, self.MAX_LEG_EXTENSION, self.dt
        )
        self.right_leg_pos, self.right_leg_vel, right_leg_accel = update_limb(
            self.right_leg_pos, self.right_leg_vel, right_leg_vel_cmd,
            MAX_LEG_ACCEL, self.MAX_LEG_EXTENSION, self.dt
        )
        self.left_arm_pos, self.left_arm_vel, left_arm_accel = update_limb(
            self.left_arm_pos, self.left_arm_vel, left_arm_vel_cmd,
            MAX_ARM_ACCEL, self.MAX_ARM_SWING, self.dt
        )
        self.right_arm_pos, self.right_arm_vel, right_arm_accel = update_limb(
            self.right_arm_pos, self.right_arm_vel, right_arm_vel_cmd,
            MAX_ARM_ACCEL, self.MAX_ARM_SWING, self.dt
        )
        self.waist_pos, self.waist_vel, waist_accel = update_limb(
            self.waist_pos, self.waist_vel, waist_vel_cmd,
            MAX_WAIST_ACCEL, self.MAX_WAIST_ANGLE, self.dt
        )

        # === FORCES FROM LIMB MOTION ===
        left_leg_force = -self.LEG_MASS * left_leg_accel
        right_leg_force = -self.LEG_MASS * right_leg_accel
        total_leg_force = left_leg_force + right_leg_force

        STANCE_WIDTH = 0.30
        leg_pitch_torque = (right_leg_force - left_leg_force) * STANCE_WIDTH / 2

        ARM_MOMENT = 0.3
        left_arm_torque = -self.ARM_MASS * left_arm_accel * ARM_MOMENT
        right_arm_torque = -self.ARM_MASS * right_arm_accel * ARM_MOMENT
        arm_pitch_torque = left_arm_torque + right_arm_torque

        left_arm_vertical_force = -self.ARM_MASS * left_arm_accel * np.cos(self.left_arm_pos)
        right_arm_vertical_force = -self.ARM_MASS * right_arm_accel * np.cos(self.right_arm_pos)
        arm_vertical_force = left_arm_vertical_force + right_arm_vertical_force

        TORSO_MASS = 35.0
        TORSO_LENGTH = 0.5
        g = 9.81
        waist_com_offset = 0.5 * TORSO_LENGTH * np.sin(self.waist_pos)
        waist_com_torque = -TORSO_MASS * g * waist_com_offset
        I_torso = (1/3) * TORSO_MASS * TORSO_LENGTH**2
        waist_momentum_torque = -I_torso * waist_accel
        waist_pitch_torque = waist_com_torque + waist_momentum_torque

        torso_com_vertical_accel = -0.5 * TORSO_LENGTH * (
            np.cos(self.waist_pos) * self.waist_vel**2 +
            np.sin(self.waist_pos) * waist_accel
        )
        torso_vertical_force = -TORSO_MASS * torso_com_vertical_accel

        total_pitch_torque = leg_pitch_torque + arm_pitch_torque + waist_pitch_torque
        total_vertical_force = total_leg_force + arm_vertical_force + torso_vertical_force

        MAX_VERTICAL_FORCE = 1500.0
        MAX_PITCH_TORQUE = 100.0
        total_vertical_force = np.clip(total_vertical_force, -MAX_VERTICAL_FORCE, MAX_VERTICAL_FORCE)
        total_pitch_torque = np.clip(total_pitch_torque, -MAX_PITCH_TORQUE, MAX_PITCH_TORQUE)

        # Physics step
        self.state = self.physics.step(
            self.state, total_vertical_force, total_pitch_torque, self.dt
        )
        self.forces = self.physics.calculate_forces(self.state)

        self._update_body_model()
        hip_height = self.body.get_hip_height()
        self.prev_hip_height = hip_height

        self.time += self.dt
        self.step_count += 1

        # Energy
        leg_power = abs(total_leg_force * (self.left_leg_vel + self.right_leg_vel) / 2)
        arm_power = abs(arm_pitch_torque * self.state.omega) + abs(arm_vertical_force * 0.1)
        waist_power = abs(waist_pitch_torque * self.waist_vel)
        self.power = leg_power + arm_power + waist_power + abs(total_vertical_force) * 0.02
        self.cumulative_energy += self.power * self.dt

        # Termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.MAX_STEPS

        # Simplified reward
        reward = self._compute_reward(prev_vx, terminated, termination_reason)

        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = total_leg_force

        return self._get_observation(), reward, terminated, truncated, info

    def _update_body_model(self):
        self.body.state.board_z = self.state.z
        self.body.state.board_pitch = self.state.theta
        avg_leg_ext = (self.left_leg_pos + self.right_leg_pos) / 2 / self.MAX_LEG_EXTENSION
        self.body.set_pose_from_leg_extension(avg_leg_ext)
        self.body.set_arm_pose(
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_pos / self.MAX_ARM_SWING
        )
        self.body.set_waist_lean(self.waist_pos, self.waist_vel)

    def _get_observation(self) -> np.ndarray:
        s = self.state
        hip_height = self.body.get_hip_height()
        com = self.body.get_center_of_mass()

        # Phase encoding based on expected pump frequency
        phase = 2 * np.pi * self.PUMP_FREQUENCY * self.time

        obs = np.array([
            # Foil state (5)
            s.z, s.vx, s.vz, s.theta, s.omega,
            # Limb positions (5)
            self.left_leg_pos / self.MAX_LEG_EXTENSION,
            self.right_leg_pos / self.MAX_LEG_EXTENSION,
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_pos / self.MAX_ARM_SWING,
            self.waist_pos / self.MAX_WAIST_ANGLE,
            # Limb velocities (5)
            self.left_leg_vel / self.MAX_LEG_VELOCITY,
            self.right_leg_vel / self.MAX_LEG_VELOCITY,
            self.left_arm_vel / self.MAX_ARM_VELOCITY,
            self.right_arm_vel / self.MAX_ARM_VELOCITY,
            self.waist_vel / self.MAX_WAIST_VELOCITY,
            # Body (3)
            hip_height,
            com[1],
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,
            # Phase encoding (2) - helps learn periodicity
            np.sin(phase),
            np.cos(phase),
            # Goals (2)
            s.z - self.TARGET_ALTITUDE,
            s.vx - self.TARGET_VELOCITY,
            # Energy (1)
            min(1.0, self.power / self.MAX_POWER),
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "x": self.state.x,
            "z": self.state.z,
            "vx": self.state.vx,
            "vz": self.state.vz,
            "theta_deg": np.degrees(self.state.theta),
            "alpha_deg": np.degrees(self.forces.alpha),
            "thrust": self.forces.F_hydro_x,
            "power": self.power,
            "cumulative_energy": self.cumulative_energy,
            "is_stalled": self.forces.is_stalled,
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
        if abs(s.theta) > np.radians(30):
            return True, "excessive_pitch"
        if self.cumulative_energy > self.MAX_ENERGY:
            return True, "energy_exhausted"

        return False, ""

    def _compute_reward(
        self,
        prev_vx: float,
        terminated: bool,
        termination_reason: str,
    ) -> float:
        """
        V3 reward: phase encoding benefits + pumping incentives + jerk penalty.
        """
        s = self.state

        # 1. Velocity tracking (exponential - gradient everywhere)
        velocity_error = abs(s.vx - self.TARGET_VELOCITY)
        velocity_reward = np.exp(-2.0 * velocity_error)  # 1.0 at target

        # Bonus for velocity gain
        velocity_change = s.vx - prev_vx
        velocity_bonus = max(0, velocity_change) * 10.0

        # 2. Alive bonus
        alive_bonus = 0.2

        # 3. Jerk penalty (smooth periodic motion)
        a0, a1, a2 = self.action_history
        jerk = np.mean(np.abs(a2 - 2*a1 + a0))
        jerk_penalty = -0.2 * jerk

        # 4. Energy efficiency
        energy_penalty = -0.02 * self.power / self.MAX_POWER

        # 5. Altitude (soft constraint)
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        altitude_penalty = -3.0 * altitude_error

        # === PUMPING INCENTIVES (added back from curriculum env) ===
        # 6. Reward leg movement speed
        leg_speed = (abs(self.left_leg_vel) + abs(self.right_leg_vel)) / 2
        leg_speed_norm = leg_speed / self.MAX_LEG_VELOCITY
        pumping_reward = leg_speed_norm * 3.0

        # 7. Reward using full leg range
        leg_amplitude = max(abs(self.left_leg_pos), abs(self.right_leg_pos)) / self.MAX_LEG_EXTENSION
        amplitude_reward = leg_amplitude * 1.5

        # 8. Frequency reward: match expected 2.5Hz pump pattern
        # Use phase encoding to reward legs following the expected rhythm
        phase = 2 * np.pi * self.PUMP_FREQUENCY * self.time
        expected_leg_sign = np.sin(phase)
        actual_leg_sign = self.left_leg_pos / (self.MAX_LEG_EXTENSION + 0.01)
        phase_match = expected_leg_sign * actual_leg_sign  # Positive if in phase
        frequency_reward = max(0, phase_match) * 2.0

        # Penalize being still
        stillness_penalty = 0.0
        if leg_speed_norm < 0.3:
            stillness_penalty = -2.0

        # Crash penalty
        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -5.0  # Mild - means we survived long
            else:
                crash_penalty = -50.0

        reward = (
            velocity_reward +
            velocity_bonus +
            alive_bonus +
            jerk_penalty +
            energy_penalty +
            altitude_penalty +
            pumping_reward +
            amplitude_reward +
            frequency_reward +
            stillness_penalty +
            crash_penalty
        )

        return reward

    def get_body_positions(self) -> dict:
        return self.body.get_joint_positions()

    def close(self):
        pass


# Register environment
gym.register(
    id="PumpFoilV2-v0",
    entry_point="foil_env.pump_foil_env_v2:PumpFoilEnvV2",
    max_episode_steps=6000,
)

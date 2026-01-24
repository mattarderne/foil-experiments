"""
Phase 1: Steady Glide with Pitch Control

Focus on learning to use arms, legs, and waist for pitch control.
Goal: Maintain altitude through active balance, NOT pumping for thrust.

Key differences from main env:
- Reward pitch stability, not velocity
- Reward USING limbs (movement), not staying still
- Penalize being passive
- Front/back weight shift matters
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, ForceOutput, CONFIG
from foil_env.body_model import BodyModel


class PumpFoilEnv4LimbPhase1(gym.Env):
    """
    Phase 1: Learn pitch control through limb coordination.

    Same action/observation space as main env, but reward focuses on:
    - Maintaining steady altitude
    - Active use of limbs for balance
    - Pitch angle stability
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Physical limits (same as main env)
    RIDER_MASS = 70.0
    LEG_MASS = 15.0
    ARM_MASS = 5.0
    BOARD_MASS = 6.0

    MAX_LEG_EXTENSION = 0.15
    MAX_ARM_SWING = 1.5
    MAX_WAIST_ANGLE = 0.3
    MAX_LEG_VELOCITY = 1.0
    MAX_ARM_VELOCITY = 3.0
    MAX_WAIST_VELOCITY = 2.0

    MAX_POWER = 300.0
    MAX_ENERGY = 6000.0  # More energy budget for learning

    TARGET_ALTITUDE = 0.15
    TARGET_VELOCITY = 4.5
    MIN_VELOCITY = 1.5
    MAX_STEPS = 3000

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

        # Action: 5 limb velocity commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation space (same as main env)
        obs_dim = 23
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces: Optional[ForceOutput] = None
        self.step_count = 0
        self.phase = 0.0

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

        # Track limb activity for reward
        self.limb_activity_history = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        initial_velocity = options.get("initial_velocity", self.TARGET_VELOCITY)

        from foil_env.foil_physics import compute_trim_angle
        trim_theta = compute_trim_angle(initial_velocity, self.physics.config)

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

        self._update_body_model()
        self.prev_hip_height = self.body.get_hip_height()

        self.step_count = 0
        self.phase = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0
        self.limb_activity_history = []

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None

        # Parse actions
        left_leg_vel_cmd = float(action[0]) * self.MAX_LEG_VELOCITY
        right_leg_vel_cmd = float(action[1]) * self.MAX_LEG_VELOCITY
        left_arm_vel_cmd = float(action[2]) * self.MAX_ARM_VELOCITY
        right_arm_vel_cmd = float(action[3]) * self.MAX_ARM_VELOCITY
        waist_vel_cmd = float(action[4]) * self.MAX_WAIST_VELOCITY

        prev_z = self.state.z
        prev_theta = self.state.theta

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

        # Leg asymmetry = pitch torque (front/back weight shift)
        STANCE_WIDTH = 0.30
        leg_pitch_torque = (right_leg_force - left_leg_force) * STANCE_WIDTH / 2

        # Arm swing = pitch torque
        ARM_MOMENT = 0.3
        left_arm_torque = -self.ARM_MASS * left_arm_accel * ARM_MOMENT
        right_arm_torque = -self.ARM_MASS * right_arm_accel * ARM_MOMENT
        arm_pitch_torque = left_arm_torque + right_arm_torque

        # Arm vertical force (deweighting effect)
        # When arms accelerate upward, reaction force pushes body down onto board
        left_arm_vertical_force = -self.ARM_MASS * left_arm_accel * np.cos(self.left_arm_pos)
        right_arm_vertical_force = -self.ARM_MASS * right_arm_accel * np.cos(self.right_arm_pos)
        arm_vertical_force = left_arm_vertical_force + right_arm_vertical_force

        # Waist torque
        TORSO_MASS = 35.0
        TORSO_LENGTH = 0.5
        g = 9.81
        waist_com_offset = 0.5 * TORSO_LENGTH * np.sin(self.waist_pos)
        waist_com_torque = -TORSO_MASS * g * waist_com_offset
        I_torso = (1/3) * TORSO_MASS * TORSO_LENGTH**2
        waist_momentum_torque = -I_torso * waist_accel
        waist_pitch_torque = waist_com_torque + waist_momentum_torque

        # Torso vertical force (deweighting from waist motion)
        # When torso leans/unleans, its CoM moves in an arc, creating vertical acceleration
        torso_com_vertical_accel = -0.5 * TORSO_LENGTH * (
            np.cos(self.waist_pos) * self.waist_vel**2 +
            np.sin(self.waist_pos) * waist_accel
        )
        torso_vertical_force = -TORSO_MASS * torso_com_vertical_accel

        total_pitch_torque = leg_pitch_torque + arm_pitch_torque + waist_pitch_torque

        # Total vertical force (legs + arms + torso deweighting)
        total_vertical_force = total_leg_force + arm_vertical_force + torso_vertical_force

        # Clamp
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

        self.phase += self.dt * 2 * np.pi * 2.0
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # Energy
        leg_power = abs(total_leg_force * (self.left_leg_vel + self.right_leg_vel) / 2)
        arm_pitch_power = abs(arm_pitch_torque * self.state.omega)
        arm_vertical_power = abs(arm_vertical_force * (self.left_arm_vel + self.right_arm_vel) / 2 * 0.3)
        arm_power = arm_pitch_power + arm_vertical_power
        waist_power = abs(waist_pitch_torque * self.waist_vel) + abs(torso_vertical_force * self.waist_vel * 0.25)
        self.power = leg_power + arm_power + waist_power + abs(total_vertical_force) * 0.02
        self.cumulative_energy += self.power * self.dt

        # Track limb activity
        activity = (
            abs(self.left_leg_vel) + abs(self.right_leg_vel) +
            abs(self.left_arm_vel) + abs(self.right_arm_vel) +
            abs(self.waist_vel)
        )
        self.limb_activity_history.append(activity)
        if len(self.limb_activity_history) > 100:
            self.limb_activity_history.pop(0)

        self.step_count += 1

        # Termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.MAX_STEPS

        # Reward (Phase 1: pitch control focus)
        reward = self._compute_reward_phase1(
            prev_z, prev_theta, total_pitch_torque,
            terminated, termination_reason
        )

        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = total_leg_force
        info["arm_vertical_force"] = arm_vertical_force
        info["torso_vertical_force"] = torso_vertical_force
        info["total_vertical_force"] = total_vertical_force
        info["pitch_torque"] = total_pitch_torque

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

        obs = np.array([
            s.z, s.vx, s.vz, s.theta, s.omega,
            self.left_leg_pos / self.MAX_LEG_EXTENSION,
            self.left_leg_vel / self.MAX_LEG_VELOCITY,
            self.right_leg_pos / self.MAX_LEG_EXTENSION,
            self.right_leg_vel / self.MAX_LEG_VELOCITY,
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.left_arm_vel / self.MAX_ARM_VELOCITY,
            self.right_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_vel / self.MAX_ARM_VELOCITY,
            self.waist_pos / self.MAX_WAIST_ANGLE,
            self.waist_vel / self.MAX_WAIST_VELOCITY,
            hip_height,
            com[1],
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,
            np.sin(self.phase),
            np.cos(self.phase),
            s.z - self.TARGET_ALTITUDE,
            s.vx - self.TARGET_VELOCITY,
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
            "left_leg_pos": self.left_leg_pos,
            "right_leg_pos": self.right_leg_pos,
            "waist_pos": self.waist_pos,
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

    def _compute_reward_phase1(
        self,
        prev_z: float,
        prev_theta: float,
        pitch_torque: float,
        terminated: bool,
        termination_reason: str,
    ) -> float:
        """Phase 1 reward: focus on pitch control and active balance."""
        s = self.state

        # === SURVIVAL (being alive is good) ===
        survival_bonus = 3.0

        # === ALTITUDE STABILITY (primary goal) ===
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        altitude_reward = np.exp(-altitude_error * 20.0) * 5.0

        # Penalize altitude drift
        altitude_drift = abs(s.z - self.TARGET_ALTITUDE) - abs(prev_z - self.TARGET_ALTITUDE)
        altitude_drift_penalty = -max(0, altitude_drift) * 100.0

        # Danger zones
        cfg = self.physics.config
        riding_depth = cfg['riding_depth']
        mast_length = cfg['mast_length']
        danger_penalty = 0.0
        if s.z < riding_depth - mast_length + 0.1:
            danger_penalty -= 3.0
        if s.z > riding_depth - 0.1:
            danger_penalty -= 3.0

        # === PITCH STABILITY (key for glide) ===
        # Reward small pitch angles (level flight)
        pitch_reward = np.exp(-abs(s.theta) * 5.0) * 3.0

        # Penalize pitch rate (smooth control)
        pitch_rate_penalty = -abs(s.omega) * 2.0

        # === ACTIVE LIMB USE (must move to learn control!) ===
        # Reward velocity of limbs (not position, velocity = active control)
        arm_activity = (abs(self.left_arm_vel) + abs(self.right_arm_vel)) / 2 / self.MAX_ARM_VELOCITY
        leg_activity = (abs(self.left_leg_vel) + abs(self.right_leg_vel)) / 2 / self.MAX_LEG_VELOCITY
        waist_activity = abs(self.waist_vel) / self.MAX_WAIST_VELOCITY

        # Reward arm shaking (important for pitch control)
        arm_reward = arm_activity * 4.0

        # Reward leg bending (weight shift)
        leg_reward = leg_activity * 3.0

        # Reward waist lean (CoM shift)
        waist_reward = waist_activity * 2.0

        # Bonus for using multiple limbs (coordination)
        multi_limb_bonus = 0.0
        if arm_activity > 0.1 and leg_activity > 0.1:
            multi_limb_bonus = 2.0
        if waist_activity > 0.1 and (arm_activity > 0.1 or leg_activity > 0.1):
            multi_limb_bonus += 1.0

        # === PENALIZE PASSIVITY ===
        # If not moving at all, penalize
        total_activity = arm_activity + leg_activity + waist_activity
        passivity_penalty = 0.0
        if total_activity < 0.05:
            passivity_penalty = -3.0

        # === FRONT/BACK WEIGHT (leg asymmetry for pitch) ===
        # Reward using different leg positions (weight shift)
        leg_diff = abs(self.left_leg_pos - self.right_leg_pos) / self.MAX_LEG_EXTENSION
        weight_shift_reward = leg_diff * 2.0

        # === VELOCITY (secondary, just don't stall) ===
        velocity_reward = 0.0
        if s.vx > 3.0:  # Above stall
            velocity_reward = 1.0
        elif s.vx > 2.0:
            velocity_reward = 0.5

        # === ENERGY (mild penalty) ===
        power_normalized = self.power / self.MAX_POWER
        energy_penalty = -power_normalized * 0.05

        # === CRASH PENALTY ===
        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -10.0  # OK to run out of energy
            else:
                crash_penalty = -50.0

        reward = (
            survival_bonus +
            altitude_reward +
            altitude_drift_penalty +
            danger_penalty +
            pitch_reward +
            pitch_rate_penalty +
            arm_reward +
            leg_reward +
            waist_reward +
            multi_limb_bonus +
            passivity_penalty +
            weight_shift_reward +
            velocity_reward +
            energy_penalty +
            crash_penalty
        )

        return reward

    def get_body_positions(self) -> dict:
        return self.body.get_joint_positions()

    def close(self):
        pass


# Register
gym.register(
    id="PumpFoil4LimbPhase1-v0",
    entry_point="foil_env.pump_foil_env_4limb_phase1:PumpFoilEnv4LimbPhase1",
    max_episode_steps=3000,
)

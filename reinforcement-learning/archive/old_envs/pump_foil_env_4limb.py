"""
Pump Foil Environment with 4-Limb Body Model

A rider with 2 legs and 2 arms standing on a pump foil board.
Each limb is independently controlled, creating more realistic pumping dynamics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, ForceOutput, CONFIG
from foil_env.body_model import BodyModel


class PumpFoilEnv4Limb(gym.Env):
    """
    4-Limb Pump Foil Environment with Waist Articulation.

    Action Space (5 dimensions):
        [left_leg_vel, right_leg_vel, left_arm_vel, right_arm_vel, waist_vel] in [-1, 1]

    Observation Space (23 dimensions):
        Foil state: z, vx, vz, theta, omega (5)
        Left leg: position, velocity (2)
        Right leg: position, velocity (2)
        Left arm: position, velocity (2)
        Right arm: position, velocity (2)
        Waist: position, velocity (2)
        Derived: hip_height, com_z, hip_velocity (3)
        Phase hint: sin, cos (2)
        Errors: altitude_error, velocity_error (2)
        Flags: energy_rate (1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Physical limits
    RIDER_MASS = 70.0          # kg total
    LEG_MASS = 15.0            # kg per leg (moves board)
    ARM_MASS = 5.0             # kg per arm (balance/momentum)
    BOARD_MASS = 6.0           # kg

    MAX_LEG_EXTENSION = 0.15   # m - leg range
    MAX_ARM_SWING = 1.5        # rad - arm swing range
    MAX_WAIST_ANGLE = 0.3      # rad - waist lean range (±17°)
    MAX_LEG_VELOCITY = 1.0     # m/s
    MAX_ARM_VELOCITY = 3.0     # rad/s
    MAX_WAIST_VELOCITY = 2.0   # rad/s

    MAX_POWER = 300.0          # W sustainable
    MAX_ENERGY = 4500.0        # J budget

    TARGET_ALTITUDE = 0.15     # m
    TARGET_VELOCITY = 4.5      # m/s
    MIN_VELOCITY = 1.5         # m/s
    MAX_STEPS = 3000           # 30s

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        stance_offset: float = 0.0,  # Rider CoM offset from center of lift (m)
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.initial_stance_offset = stance_offset

        # Merge stance offset into physics config
        physics_config = config.copy() if config else {}
        physics_config['rider_com_offset'] = stance_offset

        self.physics = FoilPhysics(physics_config)
        self.body = BodyModel()

        # Action: 4 limb + waist velocity commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Observation space
        obs_dim = 23
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces: Optional[ForceOutput] = None
        self.step_count = 0
        self.phase = 0.0

        # Limb states (normalized -1 to 1)
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = 0.0
        self.left_arm_vel = 0.0
        self.right_arm_pos = 0.0
        self.right_arm_vel = 0.0
        # Waist state (in radians)
        self.waist_pos = 0.1  # Start with slight forward lean
        self.waist_vel = 0.0

        # Previous hip height for velocity calculation
        self.prev_hip_height = 0.0

        # Energy tracking
        self.cumulative_energy = 0.0
        self.power = 0.0

        # For rendering
        self._renderer = None

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

        # Small perturbations
        z_noise = self.np_random.uniform(-0.01, 0.01) if seed else 0
        vx_noise = self.np_random.uniform(-0.1, 0.1) if seed else 0

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

        # Reset limbs to neutral
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = 0.0
        self.left_arm_vel = 0.0
        self.right_arm_pos = 0.0
        self.right_arm_vel = 0.0
        # Reset waist to slight forward lean
        self.waist_pos = 0.1
        self.waist_vel = 0.0

        # Update body model
        self._update_body_model()
        self.prev_hip_height = self.body.get_hip_height()

        self.step_count = 0
        self.phase = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

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

        # Store previous state
        prev_vx = self.state.vx
        prev_z = self.state.z
        prev_hip_height = self.prev_hip_height

        # === LIMB DYNAMICS ===
        MAX_LEG_ACCEL = 15.0  # m/s^2
        MAX_ARM_ACCEL = 30.0  # rad/s^2

        # Update each limb
        def update_limb(pos, vel, vel_cmd, max_accel, max_ext, dt):
            accel = np.clip((vel_cmd - vel) / dt, -max_accel, max_accel)
            new_vel = vel + accel * dt
            new_pos = pos + new_vel * dt

            # Soft limits
            if new_pos > max_ext * 0.9:
                new_vel = min(0, new_vel * (1 - (new_pos - max_ext * 0.9) / (max_ext * 0.1)))
                new_pos = min(max_ext, new_pos)
            elif new_pos < -max_ext * 0.9:
                new_vel = max(0, new_vel * (1 - (-max_ext * 0.9 - new_pos) / (max_ext * 0.1)))
                new_pos = max(-max_ext, new_pos)

            actual_accel = (new_vel - vel) / dt
            return new_pos, new_vel, actual_accel

        # Legs (in meters)
        self.left_leg_pos, self.left_leg_vel, left_leg_accel = update_limb(
            self.left_leg_pos, self.left_leg_vel, left_leg_vel_cmd,
            MAX_LEG_ACCEL, self.MAX_LEG_EXTENSION, self.dt
        )
        self.right_leg_pos, self.right_leg_vel, right_leg_accel = update_limb(
            self.right_leg_pos, self.right_leg_vel, right_leg_vel_cmd,
            MAX_LEG_ACCEL, self.MAX_LEG_EXTENSION, self.dt
        )

        # Arms (in radians, normalized to swing range)
        self.left_arm_pos, self.left_arm_vel, left_arm_accel = update_limb(
            self.left_arm_pos, self.left_arm_vel, left_arm_vel_cmd,
            MAX_ARM_ACCEL, self.MAX_ARM_SWING, self.dt
        )
        self.right_arm_pos, self.right_arm_vel, right_arm_accel = update_limb(
            self.right_arm_pos, self.right_arm_vel, right_arm_vel_cmd,
            MAX_ARM_ACCEL, self.MAX_ARM_SWING, self.dt
        )

        # Waist (in radians)
        MAX_WAIST_ACCEL = 15.0  # rad/s^2
        self.waist_pos, self.waist_vel, waist_accel = update_limb(
            self.waist_pos, self.waist_vel, waist_vel_cmd,
            MAX_WAIST_ACCEL, self.MAX_WAIST_ANGLE, self.dt
        )

        # === FORCES FROM LIMB MOTION ===
        # Leg forces: push board vertically
        left_leg_force = -self.LEG_MASS * left_leg_accel
        right_leg_force = -self.LEG_MASS * right_leg_accel
        total_leg_force = left_leg_force + right_leg_force

        # Leg asymmetry creates pitch torque
        STANCE_WIDTH = 0.30  # m
        leg_pitch_torque = (right_leg_force - left_leg_force) * STANCE_WIDTH / 2

        # Arm motion creates pitch torque (forward swing = pitch forward)
        ARM_MOMENT = 0.3  # m effective moment arm
        left_arm_torque = -self.ARM_MASS * left_arm_accel * ARM_MOMENT
        right_arm_torque = -self.ARM_MASS * right_arm_accel * ARM_MOMENT
        arm_pitch_torque = left_arm_torque + right_arm_torque

        # Arm vertical force (deweighting effect)
        # When arms accelerate upward, reaction force pushes body down onto board
        # Vertical component scales with cos(arm_angle) - at arm_pos=0 (down), full vertical
        left_arm_vertical_force = -self.ARM_MASS * left_arm_accel * np.cos(self.left_arm_pos)
        right_arm_vertical_force = -self.ARM_MASS * right_arm_accel * np.cos(self.right_arm_pos)
        arm_vertical_force = left_arm_vertical_force + right_arm_vertical_force

        # === WAIST TORQUE ===
        # Constants for waist physics
        TORSO_MASS = 35.0  # kg (from body_model)
        TORSO_LENGTH = 0.5  # m
        g = 9.81  # m/s^2

        # 1. CoM shift torque: Forward lean shifts center of mass forward → nose-down moment
        waist_com_offset = 0.5 * TORSO_LENGTH * np.sin(self.waist_pos)
        waist_com_torque = -TORSO_MASS * g * waist_com_offset

        # 2. Angular momentum torque: Changing lean creates reaction torque
        # I_torso = (1/3) * mass * length^2 for rod rotating about end
        I_torso = (1/3) * TORSO_MASS * TORSO_LENGTH**2  # ≈ 2.9 kg·m²
        waist_momentum_torque = -I_torso * waist_accel

        # Total waist contribution
        waist_pitch_torque = waist_com_torque + waist_momentum_torque

        # Torso vertical force (deweighting from waist motion)
        # When torso leans/unleans, its CoM moves in an arc, creating vertical acceleration
        # CoM height = 0.5 * TORSO_LENGTH * cos(waist_pos)
        # vertical accel = -0.5 * L * (cos(θ)*ω² + sin(θ)*α) where ω=waist_vel, α=waist_accel
        # Reaction force = -mass * accel (positive = pushes down on board = more lift)
        torso_com_vertical_accel = -0.5 * TORSO_LENGTH * (
            np.cos(self.waist_pos) * self.waist_vel**2 +
            np.sin(self.waist_pos) * waist_accel
        )
        torso_vertical_force = -TORSO_MASS * torso_com_vertical_accel

        # Total pitch torque
        total_pitch_torque = leg_pitch_torque + arm_pitch_torque + waist_pitch_torque

        # Total vertical force (legs + arms + torso deweighting)
        total_vertical_force = total_leg_force + arm_vertical_force + torso_vertical_force

        # Clamp forces
        MAX_VERTICAL_FORCE = 1500.0  # N total (increased to allow arm+torso contribution)
        MAX_PITCH_TORQUE = 100.0  # Nm
        total_vertical_force = np.clip(total_vertical_force, -MAX_VERTICAL_FORCE, MAX_VERTICAL_FORCE)
        total_pitch_torque = np.clip(total_pitch_torque, -MAX_PITCH_TORQUE, MAX_PITCH_TORQUE)

        # === PHYSICS STEP ===
        self.state = self.physics.step(
            self.state, total_vertical_force, total_pitch_torque, self.dt
        )
        self.forces = self.physics.calculate_forces(self.state)

        # Update body model for visualization
        self._update_body_model()
        hip_height = self.body.get_hip_height()
        hip_velocity = (hip_height - prev_hip_height) / self.dt
        self.prev_hip_height = hip_height

        # Update phase
        self.phase += self.dt * 2 * np.pi * 2.0  # 2 Hz reference
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # === ENERGY ===
        leg_power = abs(total_leg_force * (self.left_leg_vel + self.right_leg_vel) / 2)
        # Arm power includes both pitch torque work and vertical force work
        arm_pitch_power = abs(arm_pitch_torque * self.state.omega)
        arm_vertical_power = abs(arm_vertical_force * (self.left_arm_vel + self.right_arm_vel) / 2 * 0.3)  # scaled by moment arm
        arm_power = arm_pitch_power + arm_vertical_power
        waist_power = abs(waist_pitch_torque * self.waist_vel) + abs(torso_vertical_force * self.waist_vel * 0.25)
        self.power = leg_power + arm_power + waist_power + abs(total_vertical_force) * 0.02  # Small holding cost
        self.cumulative_energy += self.power * self.dt

        self.step_count += 1

        # Termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.MAX_STEPS

        # Reward
        reward = self._compute_reward(
            prev_vx, prev_z, total_leg_force, total_pitch_torque,
            terminated, termination_reason
        )

        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = total_leg_force
        info["arm_vertical_force"] = arm_vertical_force
        info["torso_vertical_force"] = torso_vertical_force
        info["total_vertical_force"] = total_vertical_force
        info["pitch_torque"] = total_pitch_torque
        info["hip_height"] = hip_height
        info["hip_velocity"] = hip_velocity

        return self._get_observation(), reward, terminated, truncated, info

    def _update_body_model(self):
        """Sync body model with current state."""
        self.body.state.board_z = self.state.z
        self.body.state.board_pitch = self.state.theta

        # Map leg positions to body model extension
        avg_leg_ext = (self.left_leg_pos + self.right_leg_pos) / 2 / self.MAX_LEG_EXTENSION
        self.body.set_pose_from_leg_extension(avg_leg_ext)

        # Map arm positions
        self.body.set_arm_pose(
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_pos / self.MAX_ARM_SWING
        )

        # Map waist lean
        self.body.set_waist_lean(self.waist_pos, self.waist_vel)

    def _get_observation(self) -> np.ndarray:
        s = self.state
        f = self.forces

        hip_height = self.body.get_hip_height()
        com = self.body.get_center_of_mass()

        obs = np.array([
            # Foil state (5)
            s.z,
            s.vx,
            s.vz,
            s.theta,
            s.omega,
            # Left leg (2)
            self.left_leg_pos / self.MAX_LEG_EXTENSION,
            self.left_leg_vel / self.MAX_LEG_VELOCITY,
            # Right leg (2)
            self.right_leg_pos / self.MAX_LEG_EXTENSION,
            self.right_leg_vel / self.MAX_LEG_VELOCITY,
            # Left arm (2)
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.left_arm_vel / self.MAX_ARM_VELOCITY,
            # Right arm (2)
            self.right_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_vel / self.MAX_ARM_VELOCITY,
            # Waist (2)
            self.waist_pos / self.MAX_WAIST_ANGLE,
            self.waist_vel / self.MAX_WAIST_VELOCITY,
            # Derived (3)
            hip_height,
            com[1],  # CoM z
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,  # hip vel
            # Phase (2)
            np.sin(self.phase),
            np.cos(self.phase),
            # Errors (2)
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
            "left_leg_pos": self.left_leg_pos,
            "right_leg_pos": self.right_leg_pos,
            "left_arm_pos": self.left_arm_pos,
            "right_arm_pos": self.right_arm_pos,
            "waist_pos": self.waist_pos,
            "stance_offset": self.physics.config['rider_com_offset'],
            "M_weight": self.forces.M_weight,
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
        prev_z: float,
        leg_force: float,
        pitch_torque: float,
        terminated: bool,
        termination_reason: str,
    ) -> float:
        s = self.state
        f = self.forces

        # Survival bonus (reduced - we want pumping, not just survival)
        survival_bonus = 2.0

        # === ALTITUDE (CRITICAL - must maintain to stay alive) ===
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        # Strong exponential penalty for altitude error
        altitude_reward = np.exp(-altitude_error * 30.0) * 3.0

        # Penalize moving away from target altitude
        altitude_drift = abs(s.z - self.TARGET_ALTITUDE) - abs(prev_z - self.TARGET_ALTITUDE)
        altitude_drift_penalty = -max(0, altitude_drift) * 200.0

        # Hard penalty for being near crash zones
        danger_zone_penalty = 0.0
        cfg = self.physics.config
        riding_depth = cfg['riding_depth']
        mast_length = cfg['mast_length']

        if s.z < riding_depth - mast_length + 0.1:  # Near floor
            danger_zone_penalty -= 5.0
        if s.z > riding_depth - 0.1:  # Near ceiling
            danger_zone_penalty -= 5.0

        # === VELOCITY (MUCH MORE IMPORTANT - need to pump to maintain) ===
        velocity_error = abs(s.vx - self.TARGET_VELOCITY)
        velocity_reward = np.exp(-velocity_error * 0.3) * 8.0  # Increased from 3.0

        velocity_loss = prev_vx - s.vx
        velocity_loss_penalty = -max(0, velocity_loss) * 150.0  # Increased from 50.0

        # Thrust bonus (important!)
        thrust = f.F_hydro_x
        thrust_reward = max(0, thrust) * 0.1  # Increased from 0.02

        # === PUMPING (CRITICAL - must pump legs to generate thrust) ===
        # Reward leg velocity (pumping motion)
        leg_vel_mag = (abs(self.left_leg_vel) + abs(self.right_leg_vel)) / 2 / self.MAX_LEG_VELOCITY
        pumping_reward = leg_vel_mag * 5.0  # Increased from 2.0

        # Reward leg amplitude (actual movement range)
        # Track leg amplitude over time window isn't easy, so use position variance
        leg_range_used = (abs(self.left_leg_pos) + abs(self.right_leg_pos)) / 2 / self.MAX_LEG_EXTENSION
        leg_amplitude_reward = leg_range_used * 3.0  # NEW: reward using leg range

        # Reward vertical velocity during leg motion (actual pumping effect)
        if abs(leg_force) > 100:  # Only when legs are active
            vz_reward = abs(s.vz) * 5.0  # Increased from 2.0
        else:
            vz_reward = 0.0

        # Coordination: reward opposite arm swing (natural running motion)
        arm_phase_diff = self.left_arm_pos * self.right_arm_pos  # Negative if opposite
        coordination_reward = -arm_phase_diff * 0.3  # Reward opposite phase

        # Penalize leg position bias (should oscillate around center)
        avg_leg_pos = (self.left_leg_pos + self.right_leg_pos) / 2
        leg_bias_penalty = -abs(avg_leg_pos) / self.MAX_LEG_EXTENSION * 2.0  # Increased

        # === WAIST CONTROL (don't get stuck at limits!) ===
        # Penalize waist being stuck at limits
        waist_limit_penalty = 0.0
        waist_normalized = abs(self.waist_pos) / self.MAX_WAIST_ANGLE
        if waist_normalized > 0.8:  # More than 80% to limit
            waist_limit_penalty = -(waist_normalized - 0.8) * 10.0  # NEW

        # === ENERGY ===
        power_normalized = self.power / self.MAX_POWER
        energy_penalty = -power_normalized * 0.05  # Reduced from 0.1

        # === PENALTIES ===
        stall_penalty = -3.0 if f.is_stalled else 0.0

        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -20.0  # Ran out of energy is OK
            else:
                crash_penalty = -100.0  # Crash is very bad

        reward = (
            survival_bonus +
            altitude_reward +
            altitude_drift_penalty +
            danger_zone_penalty +
            velocity_reward +
            velocity_loss_penalty +
            thrust_reward +
            pumping_reward +
            leg_amplitude_reward +
            vz_reward +
            coordination_reward +
            leg_bias_penalty +
            waist_limit_penalty +
            energy_penalty +
            stall_penalty +
            crash_penalty
        )

        return reward

    def get_body_positions(self) -> dict:
        """Get all body joint positions for visualization."""
        return self.body.get_joint_positions()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register
gym.register(
    id="PumpFoil4Limb-v0",
    entry_point="foil_env.pump_foil_env_4limb:PumpFoilEnv4Limb",
    max_episode_steps=3000,
)

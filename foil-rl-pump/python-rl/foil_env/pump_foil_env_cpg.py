"""
CPG-based Pump Foil Environment.

Key improvements from SOTA research:
1. CPG (Central Pattern Generator) action space - RL learns parameters, CPG generates rhythm
2. Explicit phase encoding in observations
3. Jerk penalty for smooth periodic motion
4. Residual learning option on top of manual baseline
5. Simplified reward structure

The CPG approach guarantees periodic output and dramatically reduces the action space,
making it easier for RL to learn stable pumping behavior.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, compute_trim_angle
from foil_env.body_model import BodyModel


class PumpFoilEnvCPG(gym.Env):
    """
    CPG-based environment for pump foil.

    Action space (4D or 6D):
        [0]: leg_amplitude - controls pump intensity (mapped to 0.3-1.0)
        [1]: freq_offset - adjusts pump frequency around 2.5Hz (mapped to -0.5 to +0.5 Hz)
        [2]: pitch_bias - forward/back lean (mapped to -0.5 to +0.5)
        [3]: arm_amplitude - arm swing intensity (mapped to 0.3-1.0)

    Optional residual mode adds:
        [4]: leg_residual - small adjustment to CPG output
        [5]: arm_residual - small adjustment to CPG output

    The CPG generates sinusoidal leg/arm motion. RL only controls the parameters.
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

    # CPG base parameters (from grid search optimal)
    BASE_FREQUENCY = 2.5  # Hz
    BASE_LEG_AMPLITUDE = 0.7
    BASE_PITCH_BIAS = 0.4
    BASE_ARM_AMPLITUDE = 0.7

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        use_residuals: bool = False,  # Add residual learning
        residual_scale: float = 0.2,  # How much residuals can adjust
        use_baseline: bool = True,  # Start from manual baseline
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.physics = FoilPhysics(config)
        self.body = BodyModel()

        self.use_residuals = use_residuals
        self.residual_scale = residual_scale
        self.use_baseline = use_baseline

        # Action space
        if use_residuals:
            # 4 CPG params + 2 residuals
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )
        else:
            # Just 4 CPG params
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )

        # Observation space:
        # 5 (foil state) + 5 (limb pos) + 5 (limb vel) + 4 (body) + 2 (phase encoding)
        # + 2 (goals) + 1 (energy) + 2 (cpg state) = 26
        obs_dim = 26
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces = None
        self.step_count = 0
        self.time = 0.0

        # CPG internal state
        self.cpg_phase = 0.0
        self.current_freq = self.BASE_FREQUENCY
        self.current_leg_amp = self.BASE_LEG_AMPLITUDE
        self.current_arm_amp = self.BASE_ARM_AMPLITUDE
        self.current_pitch_bias = self.BASE_PITCH_BIAS

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

        # Action history for jerk penalty (last 3 actions)
        self.action_history = [np.zeros(4), np.zeros(4), np.zeros(4)]

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

        # Reset CPG
        self.cpg_phase = 0.0
        self.current_freq = self.BASE_FREQUENCY
        self.current_leg_amp = self.BASE_LEG_AMPLITUDE
        self.current_arm_amp = self.BASE_ARM_AMPLITUDE
        self.current_pitch_bias = self.BASE_PITCH_BIAS

        # Reset action history
        self.action_history = [np.zeros(4), np.zeros(4), np.zeros(4)]

        self._update_body_model()
        self.prev_hip_height = self.body.get_hip_height()

        self.step_count = 0
        self.time = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_observation(), self._get_info()

    def _parse_action(self, action: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Parse RL action into CPG parameters.

        Returns:
            (leg_amplitude, frequency, pitch_bias, arm_amplitude, leg_residual, arm_residual)
        """
        action = np.array(action, dtype=np.float32)

        if self.use_baseline:
            # Residual mode: action modulates around baseline
            leg_amp = self.BASE_LEG_AMPLITUDE + action[0] * 0.3  # 0.4 to 1.0
            freq = self.BASE_FREQUENCY + action[1] * 0.5  # 2.0 to 3.0 Hz
            pitch_bias = self.BASE_PITCH_BIAS + action[2] * 0.2  # 0.2 to 0.6
            arm_amp = self.BASE_ARM_AMPLITUDE + action[3] * 0.3  # 0.4 to 1.0
        else:
            # Full range learning
            leg_amp = 0.65 + action[0] * 0.35  # 0.3 to 1.0
            freq = 2.5 + action[1] * 1.0  # 1.5 to 3.5 Hz
            pitch_bias = action[2] * 0.5  # -0.5 to 0.5
            arm_amp = 0.65 + action[3] * 0.35  # 0.3 to 1.0

        # Clip to valid ranges
        leg_amp = np.clip(leg_amp, 0.3, 1.0)
        freq = np.clip(freq, 1.5, 3.5)
        pitch_bias = np.clip(pitch_bias, -0.5, 0.6)
        arm_amp = np.clip(arm_amp, 0.3, 1.0)

        # Residuals (optional)
        if self.use_residuals and len(action) >= 6:
            leg_residual = action[4] * self.residual_scale
            arm_residual = action[5] * self.residual_scale
        else:
            leg_residual = 0.0
            arm_residual = 0.0

        return leg_amp, freq, pitch_bias, arm_amp, leg_residual, arm_residual

    def _cpg_step(
        self,
        leg_amp: float,
        freq: float,
        pitch_bias: float,
        arm_amp: float,
        leg_residual: float,
        arm_residual: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        CPG generates target velocities from parameters.

        Returns:
            (left_leg_vel, right_leg_vel, left_arm_vel, right_arm_vel, waist_vel)
        """
        # Update CPG phase
        omega = 2 * np.pi * freq
        self.cpg_phase += omega * self.dt
        if self.cpg_phase > 2 * np.pi:
            self.cpg_phase -= 2 * np.pi

        # Store current params
        self.current_freq = freq
        self.current_leg_amp = leg_amp
        self.current_arm_amp = arm_amp
        self.current_pitch_bias = pitch_bias

        # === ALTITUDE-RESPONSIVE DC OFFSET ===
        # If too low, shift oscillation center down (extend legs more) to generate lift
        # If too high, shift up (retract legs) to reduce lift
        altitude_error = self.state.z - self.TARGET_ALTITUDE  # positive = too high
        # Aggressive altitude correction: 0.5 of max extension per 10cm error
        altitude_offset = -altitude_error * 0.5 * self.MAX_LEG_EXTENSION / 0.10
        altitude_offset = np.clip(altitude_offset, -0.5 * self.MAX_LEG_EXTENSION, 0.5 * self.MAX_LEG_EXTENSION)

        # === LEG CPG ===
        # Target position: sinusoidal + altitude offset
        target_leg_pos = leg_amp * self.MAX_LEG_EXTENSION * np.sin(self.cpg_phase) + altitude_offset

        # PD control to track target
        kp_leg = 15.0
        avg_leg_pos = (self.left_leg_pos + self.right_leg_pos) / 2
        leg_vel_cmd = kp_leg * (target_leg_pos - avg_leg_pos)

        # Add feedforward velocity (derivative of target)
        target_leg_vel = leg_amp * self.MAX_LEG_EXTENSION * omega * np.cos(self.cpg_phase)
        leg_vel_cmd += 0.5 * target_leg_vel

        # Add STRONGER residual (increased from 1x to 2x for more freedom)
        leg_vel_cmd += leg_residual * self.MAX_LEG_VELOCITY * 2.0

        # Clip
        leg_vel_cmd = np.clip(leg_vel_cmd, -self.MAX_LEG_VELOCITY, self.MAX_LEG_VELOCITY)

        # === ARM CPG ===
        # Arms same phase as legs (matches pure RL discovery r=+0.44)
        arm_phase = self.cpg_phase
        target_arm_pos = arm_amp * self.MAX_ARM_SWING * np.sin(arm_phase)

        kp_arm = 12.0
        avg_arm_pos = (self.left_arm_pos + self.right_arm_pos) / 2
        arm_vel_cmd = kp_arm * (target_arm_pos - avg_arm_pos)

        # Feedforward
        target_arm_vel = arm_amp * self.MAX_ARM_SWING * omega * np.cos(arm_phase)
        arm_vel_cmd += 0.5 * target_arm_vel

        # Add STRONGER residual
        arm_vel_cmd += arm_residual * self.MAX_ARM_VELOCITY * 2.0

        arm_vel_cmd = np.clip(arm_vel_cmd, -self.MAX_ARM_VELOCITY, self.MAX_ARM_VELOCITY)

        # === WAIST ===
        # Waist follows pitch bias with small oscillation
        waist_target = pitch_bias * 0.5  # Scale pitch bias to waist angle
        waist_vel_cmd = 5.0 * (waist_target - self.waist_pos)
        waist_vel_cmd = np.clip(waist_vel_cmd, -self.MAX_WAIST_VELOCITY, self.MAX_WAIST_VELOCITY)

        return leg_vel_cmd, leg_vel_cmd, arm_vel_cmd, arm_vel_cmd, waist_vel_cmd

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None

        # Parse action into CPG parameters
        leg_amp, freq, pitch_bias, arm_amp, leg_residual, arm_residual = self._parse_action(action)

        # Update action history for jerk penalty
        self.action_history.pop(0)
        self.action_history.append(action[:4].copy())

        # Get velocity commands from CPG
        left_leg_vel_cmd, right_leg_vel_cmd, left_arm_vel_cmd, right_arm_vel_cmd, waist_vel_cmd = \
            self._cpg_step(leg_amp, freq, pitch_bias, arm_amp, leg_residual, arm_residual)

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

        # Arm vertical force (deweighting)
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

        # Torso vertical force
        torso_com_vertical_accel = -0.5 * TORSO_LENGTH * (
            np.cos(self.waist_pos) * self.waist_vel**2 +
            np.sin(self.waist_pos) * waist_accel
        )
        torso_vertical_force = -TORSO_MASS * torso_com_vertical_accel

        total_pitch_torque = leg_pitch_torque + arm_pitch_torque + waist_pitch_torque
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

        # Reward (simplified with jerk penalty)
        reward = self._compute_reward(prev_vx, prev_z, terminated, termination_reason)

        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = total_leg_force
        info["arm_vertical_force"] = arm_vertical_force
        info["cpg_freq"] = self.current_freq
        info["cpg_leg_amp"] = self.current_leg_amp
        info["cpg_arm_amp"] = self.current_arm_amp

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

        # Jerk from action history (for penalty computation reference)
        a0, a1, a2 = self.action_history
        jerk = np.mean(np.abs(a2 - 2*a1 + a0))

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
            # Body state (4)
            hip_height,
            com[1],
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,
            jerk,  # Include jerk in observation
            # Phase encoding (2) - CRITICAL for learning periodicity
            np.sin(self.cpg_phase),
            np.cos(self.cpg_phase),
            # Goals (2)
            s.z - self.TARGET_ALTITUDE,
            s.vx - self.TARGET_VELOCITY,
            # Energy (1)
            min(1.0, self.power / self.MAX_POWER),
            # CPG state (2)
            self.current_freq / self.BASE_FREQUENCY - 1.0,  # Normalized around 1
            self.current_leg_amp,
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
            "cpg_phase": self.cpg_phase,
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
        terminated: bool,
        termination_reason: str,
    ) -> float:
        """
        Simplified reward with jerk penalty.

        Components:
        1. Velocity tracking (exponential) - primary objective
        2. Alive bonus
        3. Jerk penalty - smooth actions
        4. Energy efficiency
        """
        s = self.state

        # 1. Velocity tracking (exponential for gradient everywhere)
        velocity_error = abs(s.vx - self.TARGET_VELOCITY)
        velocity_reward = np.exp(-2.0 * velocity_error)  # 1.0 at target, decays smoothly

        # Bonus for velocity improvement
        velocity_change = s.vx - prev_vx
        velocity_bonus = max(0, velocity_change) * 5.0

        # 2. Alive bonus
        alive_bonus = 0.1

        # 3. Jerk penalty (smooth periodic motion)
        a0, a1, a2 = self.action_history
        jerk = np.mean(np.abs(a2 - 2*a1 + a0))
        jerk_penalty = -0.5 * jerk

        # 4. Energy efficiency
        energy_penalty = -0.01 * self.power / self.MAX_POWER

        # 5. Altitude (soft boundaries)
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        altitude_penalty = -2.0 * altitude_error

        # 6. Pitch stability
        pitch_penalty = -abs(s.theta) * 1.0

        # Crash penalty
        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -10.0  # Less severe - means we survived long
            else:
                crash_penalty = -50.0

        reward = (
            velocity_reward +
            velocity_bonus +
            alive_bonus +
            jerk_penalty +
            energy_penalty +
            altitude_penalty +
            pitch_penalty +
            crash_penalty
        )

        return reward

    def get_body_positions(self) -> dict:
        return self.body.get_joint_positions()

    def close(self):
        pass


# Register environments
gym.register(
    id="PumpFoilCPG-v0",
    entry_point="foil_env.pump_foil_env_cpg:PumpFoilEnvCPG",
    max_episode_steps=6000,
    kwargs={"use_baseline": True, "use_residuals": False},
)

gym.register(
    id="PumpFoilCPG-v1",
    entry_point="foil_env.pump_foil_env_cpg:PumpFoilEnvCPG",
    max_episode_steps=6000,
    kwargs={"use_baseline": True, "use_residuals": True},
)

gym.register(
    id="PumpFoilCPG-v2",
    entry_point="foil_env.pump_foil_env_cpg:PumpFoilEnvCPG",
    max_episode_steps=6000,
    kwargs={"use_baseline": False, "use_residuals": False},
)

"""
Curriculum learning environment for pump foil.

Starts with forced leg pumping, gradually releases control to agent.
- Phase 0: Legs auto-pump at 2Hz, agent controls amplitude + arms/waist
- Phase 1: Agent gets partial leg control (blended with auto-pump)
- Phase 2: Full agent control with pumping incentive
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, compute_trim_angle
from foil_env.body_model import BodyModel
from foil_env.config import LimbConfig, DEFAULT_LIMBS


class PumpFoilEnvCurriculum(gym.Env):
    """
    Curriculum environment that forces pumping behavior initially.
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
    MAX_FOOT_OFFSET = 0.15  # Max foot position shift fore/aft (m)
    MAX_LEG_VELOCITY = 2.0  # Increased for full-range 2Hz pumping (v = 2πfA ≈ 1.9 m/s)
    MAX_ARM_VELOCITY = 12.0  # Realistic pump foiling arm speed (was 3.0, too slow)
    MAX_WAIST_VELOCITY = 2.0
    MAX_FOOT_VELOCITY = 1.0  # Foot position change rate (m/s)

    MAX_POWER = 300.0
    MAX_ENERGY = 30000.0  # Increased from 6000 for longer flights

    TARGET_ALTITUDE = 0.10  # 10cm - balanced margin (10cm to breach, 60cm to touchdown)
    TARGET_VELOCITY = 4.5
    MIN_VELOCITY = 1.5
    MAX_STEPS = 6000  # Increased from 3000 for 60s max flight

    # Curriculum parameters
    PUMP_FREQUENCY = 2.0  # Hz - forced pump frequency

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        limb_config: Optional[LimbConfig] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        curriculum_phase: int = 0,  # 0=forced pump, 1=blended, 2=free
        agent_blend: float = 0.0,  # How much agent controls legs (0=auto, 1=full agent)
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.physics = FoilPhysics(config)
        self.body = BodyModel()
        self.limb_config = limb_config if limb_config is not None else DEFAULT_LIMBS

        self.curriculum_phase = curriculum_phase
        self.agent_blend = agent_blend

        # Action space: [left_leg, right_leg, left_arm, right_arm, waist, foot_position]
        # In phase 0: leg actions control amplitude/phase of auto-pump
        # In phase 2: leg actions are direct velocity commands
        # foot_position: shifts weight fore/aft for trim control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation space
        obs_dim = 26  # Added pump phase + foot position
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # State
        self.state: Optional[FoilState] = None
        self.forces = None
        self.step_count = 0
        self.time = 0.0
        self.pump_phase = 0.0

        # Limb states - arms start at DOWN position (π/2), not forward (0)
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = self.limb_config.arm_start_position  # π/2 = DOWN
        self.left_arm_vel = 0.0
        self.right_arm_pos = self.limb_config.arm_start_position  # π/2 = DOWN
        self.right_arm_vel = 0.0
        self.waist_pos = self.limb_config.waist_start_position
        self.waist_vel = 0.0
        self.foot_offset = 0.0  # Foot position fore/aft (m)
        self.foot_vel = 0.0

        self.prev_hip_height = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

    def set_curriculum(self, phase: int = None, blend: float = None):
        """Update curriculum parameters."""
        if phase is not None:
            self.curriculum_phase = phase
        if blend is not None:
            self.agent_blend = np.clip(blend, 0.0, 1.0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        initial_velocity = options.get("initial_velocity", self.TARGET_VELOCITY)

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

        # Reset limbs - arms start at DOWN position (π/2), not forward (0)
        self.left_leg_pos = 0.0
        self.left_leg_vel = 0.0
        self.right_leg_pos = 0.0
        self.right_leg_vel = 0.0
        self.left_arm_pos = self.limb_config.arm_start_position  # π/2 = DOWN
        self.left_arm_vel = 0.0
        self.right_arm_pos = self.limb_config.arm_start_position  # π/2 = DOWN
        self.right_arm_vel = 0.0
        self.waist_pos = self.limb_config.waist_start_position
        self.waist_vel = 0.0
        self.foot_offset = 0.0
        self.foot_vel = 0.0

        self._update_body_model()
        self.prev_hip_height = self.body.get_hip_height()

        self.step_count = 0
        self.time = 0.0
        self.pump_phase = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None

        # Parse actions based on curriculum phase
        if self.curriculum_phase == 0:
            # Forced pumping: action[0] = amplitude, action[1] = phase offset
            pump_amplitude = 0.5 + 0.5 * float(action[0])  # 0.5 to 1.0
            phase_offset = float(action[1]) * 0.5  # Small phase adjustment

            # Auto-generate leg velocity commands from sine wave
            self.pump_phase = 2 * np.pi * self.PUMP_FREQUENCY * self.time + phase_offset
            target_leg_pos = pump_amplitude * self.MAX_LEG_EXTENSION * np.sin(self.pump_phase)

            # PD control to follow target
            leg_vel_cmd = 10.0 * (target_leg_pos - self.left_leg_pos)
            left_leg_vel_cmd = np.clip(leg_vel_cmd, -self.MAX_LEG_VELOCITY, self.MAX_LEG_VELOCITY)
            right_leg_vel_cmd = left_leg_vel_cmd  # Symmetric pumping

        elif self.curriculum_phase == 1:
            # Blended: mix auto-pump with agent control
            pump_amplitude = 0.7
            self.pump_phase = 2 * np.pi * self.PUMP_FREQUENCY * self.time
            auto_leg_pos = pump_amplitude * self.MAX_LEG_EXTENSION * np.sin(self.pump_phase)
            auto_leg_vel = 10.0 * (auto_leg_pos - self.left_leg_pos)

            agent_leg_vel = float(action[0]) * self.MAX_LEG_VELOCITY

            # Blend
            left_leg_vel_cmd = (1 - self.agent_blend) * auto_leg_vel + self.agent_blend * agent_leg_vel
            right_leg_vel_cmd = (1 - self.agent_blend) * auto_leg_vel + self.agent_blend * float(action[1]) * self.MAX_LEG_VELOCITY

        else:
            # Full agent control
            left_leg_vel_cmd = float(action[0]) * self.MAX_LEG_VELOCITY
            right_leg_vel_cmd = float(action[1]) * self.MAX_LEG_VELOCITY

        # Arms, waist, and foot position always agent-controlled
        left_arm_vel_cmd = float(action[2]) * self.MAX_ARM_VELOCITY
        right_arm_vel_cmd = float(action[3]) * self.MAX_ARM_VELOCITY
        waist_vel_cmd = float(action[4]) * self.MAX_WAIST_VELOCITY
        foot_vel_cmd = float(action[5]) * self.MAX_FOOT_VELOCITY

        prev_vx = self.state.vx
        prev_z = self.state.z

        # === LIMB DYNAMICS ===
        MAX_LEG_ACCEL = self.limb_config.max_leg_accel
        MAX_ARM_ACCEL = self.limb_config.max_arm_accel
        MAX_WAIST_ACCEL = self.limb_config.max_waist_accel

        def update_limb(pos, vel, vel_cmd, max_accel, max_ext, dt):
            """Update leg/waist limbs (symmetric around 0)."""
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

        def update_arm(pos, vel, vel_cmd, dt, cfg):
            """Update arm with centering spring (oscillates around neutral position)."""
            neutral = cfg.arm_start_position
            max_swing = cfg.max_arm_swing
            max_accel = cfg.max_arm_accel
            spring_k = cfg.arm_centering_spring

            # Spring force pulls toward neutral position
            spring_accel = -spring_k * (pos - neutral)

            # Command acceleration
            cmd_accel = np.clip((vel_cmd - vel) / dt, -max_accel, max_accel)

            # Total acceleration = command + spring
            total_accel = cmd_accel + spring_accel

            # Clamp total acceleration
            total_accel = np.clip(total_accel, -max_accel, max_accel)

            new_vel = vel + total_accel * dt
            new_pos = pos + new_vel * dt

            # Hard limits around neutral
            if new_pos > neutral + max_swing:
                new_pos = neutral + max_swing
                new_vel = min(0, new_vel)
            elif new_pos < neutral - max_swing:
                new_pos = neutral - max_swing
                new_vel = max(0, new_vel)

            actual_accel = (new_vel - vel) / dt
            return new_pos, new_vel, actual_accel

        # Update limbs
        self.left_leg_pos, self.left_leg_vel, left_leg_accel = update_limb(
            self.left_leg_pos, self.left_leg_vel, left_leg_vel_cmd,
            MAX_LEG_ACCEL, self.limb_config.max_leg_extension, self.dt
        )
        self.right_leg_pos, self.right_leg_vel, right_leg_accel = update_limb(
            self.right_leg_pos, self.right_leg_vel, right_leg_vel_cmd,
            MAX_LEG_ACCEL, self.limb_config.max_leg_extension, self.dt
        )
        self.left_arm_pos, self.left_arm_vel, left_arm_accel = update_arm(
            self.left_arm_pos, self.left_arm_vel, left_arm_vel_cmd,
            self.dt, self.limb_config
        )
        self.right_arm_pos, self.right_arm_vel, right_arm_accel = update_arm(
            self.right_arm_pos, self.right_arm_vel, right_arm_vel_cmd,
            self.dt, self.limb_config
        )
        self.waist_pos, self.waist_vel, waist_accel = update_limb(
            self.waist_pos, self.waist_vel, waist_vel_cmd,
            MAX_WAIST_ACCEL, self.limb_config.max_waist_angle, self.dt
        )

        # Foot position (fore/aft weight shift)
        MAX_FOOT_ACCEL = 5.0  # m/s^2
        self.foot_offset, self.foot_vel, foot_accel = update_limb(
            self.foot_offset, self.foot_vel, foot_vel_cmd,
            MAX_FOOT_ACCEL, self.MAX_FOOT_OFFSET, self.dt
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

        # Torso vertical force (deweighting)
        torso_com_vertical_accel = -0.5 * TORSO_LENGTH * (
            np.cos(self.waist_pos) * self.waist_vel**2 +
            np.sin(self.waist_pos) * waist_accel
        )
        torso_vertical_force = -TORSO_MASS * torso_com_vertical_accel

        total_pitch_torque = leg_pitch_torque + arm_pitch_torque + waist_pitch_torque
        total_vertical_force = total_leg_force + arm_vertical_force + torso_vertical_force

        # Foot offset affects rider CoM position (pitch trim)
        # Positive foot_offset = feet forward = CoM aft = nose up tendency
        # Apply as additional pitch torque from weight shift
        rider_weight = self.RIDER_MASS * 9.81
        foot_pitch_torque = rider_weight * self.foot_offset * 0.5  # 50% effectiveness
        total_pitch_torque += foot_pitch_torque

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
        info["pump_phase"] = self.pump_phase

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
            self.foot_offset / self.MAX_FOOT_OFFSET,  # Foot position
            self.foot_vel / self.MAX_FOOT_VELOCITY,   # Foot velocity
            hip_height,
            com[1],
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,
            np.sin(self.pump_phase),
            np.cos(self.pump_phase),
            s.z - self.TARGET_ALTITUDE,
            s.vx - self.TARGET_VELOCITY,
            min(1.0, self.power / self.MAX_POWER),
            self.agent_blend,  # Tell agent how much control it has
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
            "curriculum_phase": self.curriculum_phase,
            "agent_blend": self.agent_blend,
        }

    def _check_termination(self) -> Tuple[bool, str]:
        s = self.state
        cfg = self.physics.config

        riding_depth = cfg['riding_depth']
        mast_length = cfg['mast_length']

        if s.z >= riding_depth:
            return True, "foil_breached"
        # Board touches water when z <= riding_depth - mast_length
        # (board_y = mast_length - riding_depth + z = 0)
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

        # Survival
        reward = 1.0

        # Velocity maintenance (primary goal)
        velocity_reward = -abs(s.vx - self.TARGET_VELOCITY) * 2.0
        velocity_change = s.vx - prev_vx
        if velocity_change > 0:
            velocity_reward += velocity_change * 10.0

        # === PUMPING INCENTIVE ===
        # Reward leg velocity (fast pumping)
        leg_speed = (abs(self.left_leg_vel) + abs(self.right_leg_vel)) / 2
        leg_speed_norm = leg_speed / self.MAX_LEG_VELOCITY
        pumping_reward = leg_speed_norm * 5.0  # Strong reward for fast movement

        # Reward using full leg range
        leg_amplitude = max(abs(self.left_leg_pos), abs(self.right_leg_pos)) / self.MAX_LEG_EXTENSION
        amplitude_reward = leg_amplitude * 2.0

        # FREQUENCY REWARD: Reward being at ~2Hz pump cycle
        # Use pump_phase as reference (2Hz = full cycle every 0.5s)
        # Reward if legs are following the expected pump pattern
        expected_leg_sign = np.sin(2 * np.pi * self.PUMP_FREQUENCY * self.time)
        actual_leg_sign = self.left_leg_pos / (self.MAX_LEG_EXTENSION + 0.01)
        phase_match = expected_leg_sign * actual_leg_sign  # Positive if in phase
        frequency_reward = max(0, phase_match) * 3.0  # Reward following 2Hz pattern

        # Penalize being still
        if leg_speed_norm < 0.3:
            pumping_reward -= 3.0

        # === ARM-LEG COORDINATION REWARD ===
        # Arms should move in rhythm with legs (coordinated pumping)
        # Use velocity correlation: reward arms moving WITH or AGAINST legs in a pattern

        avg_leg_vel = (self.left_leg_vel + self.right_leg_vel) / 2
        avg_arm_vel = (self.left_arm_vel + self.right_arm_vel) / 2

        # Normalize velocities
        leg_vel_norm = avg_leg_vel / self.limb_config.max_leg_velocity
        arm_vel_norm = avg_arm_vel / self.limb_config.max_arm_velocity

        # Reward correlation (positive = in-phase, negative = anti-phase)
        # Both are valid pumping patterns, so reward absolute correlation
        velocity_correlation = leg_vel_norm * arm_vel_norm
        arm_coordination_reward = abs(velocity_correlation) * 2.0

        # Small penalty for completely static arms (not participating)
        arm_speed = (abs(self.left_arm_vel) + abs(self.right_arm_vel)) / 2
        arm_speed_norm = arm_speed / self.limb_config.max_arm_velocity
        if arm_speed_norm < 0.1:
            arm_coordination_reward -= 1.0

        # Altitude (stay in safe zone)
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        altitude_reward = -altitude_error * 5.0

        # Pitch stability
        pitch_reward = -abs(s.theta) * 2.0

        # Energy efficiency - reward low power while maintaining flight
        energy_penalty = -self.power / self.MAX_POWER * 0.1

        # Efficiency bonus: reward flight time per energy spent
        # Higher efficiency = longer flight with less energy
        if self.cumulative_energy > 100:  # Avoid division issues early
            efficiency = (self.step_count * 0.01) / (self.cumulative_energy / 1000)
            efficiency_bonus = min(efficiency * 0.5, 2.0)  # Cap at 2.0
        else:
            efficiency_bonus = 0.0

        # Crash penalty
        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -20.0
            else:
                crash_penalty = -100.0

        reward += velocity_reward + altitude_reward + pitch_reward + energy_penalty + crash_penalty
        reward += pumping_reward + amplitude_reward + frequency_reward  # Leg pumping incentive
        reward += arm_coordination_reward  # Arm-leg coordination incentive
        reward += efficiency_bonus  # Efficiency incentive

        return reward

    def get_body_positions(self) -> dict:
        return self.body.get_joint_positions()

    def close(self):
        pass


# Register environments for different curriculum phases
gym.register(
    id="PumpFoilCurriculum-v0",
    entry_point="foil_env.pump_foil_env_curriculum:PumpFoilEnvCurriculum",
    max_episode_steps=3000,
    kwargs={"curriculum_phase": 0},
)

gym.register(
    id="PumpFoilCurriculum-v1",
    entry_point="foil_env.pump_foil_env_curriculum:PumpFoilEnvCurriculum",
    max_episode_steps=3000,
    kwargs={"curriculum_phase": 1, "agent_blend": 0.5},
)

gym.register(
    id="PumpFoilCurriculum-v2",
    entry_point="foil_env.pump_foil_env_curriculum:PumpFoilEnvCurriculum",
    max_episode_steps=3000,
    kwargs={"curriculum_phase": 2},
)

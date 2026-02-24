"""
Hybrid ACMPC-style Pump Foil Environment.

Combines:
- MPC-like trajectory planning for coordinated arm-leg motion
- RL for learning pump parameters and residual adjustments

Architecture inspired by ACMPC (Actor-Critic MPC):
- Short-term planning via PumpCyclePlanner (structured coordination)
- Long-term learning via PPO (exploration, adaptation)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, compute_trim_angle
from foil_env.body_model import BodyModel
from foil_env.pump_cycle_planner import PumpCyclePlanner, HybridPumpController


class PumpFoilEnvHybrid(gym.Env):
    """
    Hybrid environment with MPC-style trajectory planning.

    Action space options:
    - 'params_only': RL controls pump parameters only (5D)
    - 'params_residual': RL controls parameters + residuals (10D)
    - 'full_hybrid': Full hybrid control with blending (10D)
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

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        control_mode: str = "params_only",  # 'params_only', 'params_residual', 'full_hybrid'
        residual_scale: float = 0.3,
    ):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.physics = FoilPhysics(config)
        self.body = BodyModel()
        self.control_mode = control_mode
        self.residual_scale = residual_scale

        # Initialize hybrid controller
        self.controller = HybridPumpController(
            max_leg_ext=self.MAX_LEG_EXTENSION,
            max_arm_swing=self.MAX_ARM_SWING,
            max_waist_angle=self.MAX_WAIST_ANGLE,
            max_leg_vel=self.MAX_LEG_VELOCITY,
            max_arm_vel=self.MAX_ARM_VELOCITY,
            max_waist_vel=self.MAX_WAIST_VELOCITY,
            dt=dt,
            residual_scale=residual_scale,
        )

        # Action space depends on control mode
        if control_mode == "params_only":
            # [freq, leg_amp, arm_amp, arm_phase, waist_amp]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            )
        else:
            # [params (5) + residuals (5)]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(10,), dtype=np.float32
            )

        # Observation space: state + planner info
        # 5 (foil) + 5 (limb pos) + 3 (limb vel) + 3 (body) + 4 (planner) + 3 (tracking) + 2 (goals) + 1 (energy) = 26
        obs_dim = 26
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

        # Track planned vs actual for analysis
        self.planned_trajectory = None

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
        self.time = 0.0
        self.cumulative_energy = 0.0
        self.power = 0.0

        # Reset controller
        self.controller.reset()

        self.forces = self.physics.calculate_forces(self.state)

        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None

        # Expand action if params_only mode
        if self.control_mode == "params_only":
            full_action = np.concatenate([action, np.zeros(5)])
        else:
            full_action = action

        # Get commands from hybrid controller
        avg_leg_pos = (self.left_leg_pos + self.right_leg_pos) / 2
        avg_arm_pos = (self.left_arm_pos + self.right_arm_pos) / 2

        left_leg_cmd, right_leg_cmd, left_arm_cmd, right_arm_cmd, waist_cmd = \
            self.controller.compute_action(
                full_action,
                current_leg_pos=avg_leg_pos,
                current_arm_pos=avg_arm_pos,
                current_waist_pos=self.waist_pos,
            )

        # Convert normalized commands to velocity commands
        left_leg_vel_cmd = left_leg_cmd * self.MAX_LEG_VELOCITY
        right_leg_vel_cmd = right_leg_cmd * self.MAX_LEG_VELOCITY
        left_arm_vel_cmd = left_arm_cmd * self.MAX_ARM_VELOCITY
        right_arm_vel_cmd = right_arm_cmd * self.MAX_ARM_VELOCITY
        waist_vel_cmd = waist_cmd * self.MAX_WAIST_VELOCITY

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

        # Arm vertical force
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

        # Reward
        reward = self._compute_reward(
            prev_vx, prev_z, total_leg_force, total_pitch_torque,
            terminated, termination_reason
        )

        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = total_leg_force
        info["arm_vertical_force"] = arm_vertical_force
        info["total_vertical_force"] = total_vertical_force
        info["pitch_torque"] = total_pitch_torque
        info["planner_phase"] = self.controller.planner.phase
        info["planner_freq"] = self.controller.planner.params.frequency

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
        planner = self.controller.planner

        # Get planned targets for next step
        leg_target, _, arm_target, _, waist_target, _ = planner.step()
        # Step back to not advance phase
        planner.phase -= 2 * np.pi * planner.params.frequency * self.dt
        if planner.phase < 0:
            planner.phase += 2 * np.pi

        obs = np.array([
            # Foil state (5)
            s.z, s.vx, s.vz, s.theta, s.omega,
            # Limb positions normalized (6)
            self.left_leg_pos / self.MAX_LEG_EXTENSION,
            self.right_leg_pos / self.MAX_LEG_EXTENSION,
            self.left_arm_pos / self.MAX_ARM_SWING,
            self.right_arm_pos / self.MAX_ARM_SWING,
            self.waist_pos / self.MAX_WAIST_ANGLE,
            # Limb velocities normalized (5)
            self.left_leg_vel / self.MAX_LEG_VELOCITY,
            self.left_arm_vel / self.MAX_ARM_VELOCITY,
            self.waist_vel / self.MAX_WAIST_VELOCITY,
            # Body state (3)
            hip_height,
            com[1],
            (self.prev_hip_height - hip_height) / self.dt if self.step_count > 0 else 0,
            # Planner state (4)
            np.sin(planner.phase),
            np.cos(planner.phase),
            planner.params.frequency / 3.0,  # Normalized frequency
            planner.params.arm_phase_offset / (np.pi / 4),  # Normalized phase offset
            # Tracking errors (3)
            (leg_target - (self.left_leg_pos + self.right_leg_pos) / 2) / self.MAX_LEG_EXTENSION,
            (arm_target - (self.left_arm_pos + self.right_arm_pos) / 2) / self.MAX_ARM_SWING,
            (waist_target - self.waist_pos) / self.MAX_WAIST_ANGLE,
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
        prev_z: float,
        leg_force: float,
        pitch_torque: float,
        terminated: bool,
        termination_reason: str,
    ) -> float:
        s = self.state
        planner = self.controller.planner

        # Base: distance traveled
        reward = s.vx * self.dt

        # Velocity maintenance
        velocity_ratio = s.vx / self.TARGET_VELOCITY
        reward += velocity_ratio * self.dt

        # Altitude stability
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        reward -= altitude_error * 0.5 * self.dt

        # Pitch stability
        reward -= abs(s.theta) * 0.2 * self.dt

        # === COORDINATION REWARD ===
        # Reward arm-leg phase synchronization (key insight from our analysis)
        avg_leg_pos = (self.left_leg_pos + self.right_leg_pos) / 2
        avg_arm_pos = (self.left_arm_pos + self.right_arm_pos) / 2

        # Same-phase correlation bonus
        leg_norm = avg_leg_pos / self.MAX_LEG_EXTENSION
        arm_norm = avg_arm_pos / self.MAX_ARM_SWING
        phase_sync = leg_norm * arm_norm  # Positive when in phase
        reward += max(0, phase_sync) * 0.5 * self.dt

        # Reward following the planned trajectory
        leg_target, _, arm_target, _, waist_target, _ = self.controller.planner.step()
        # Step back
        planner.phase -= 2 * np.pi * planner.params.frequency * self.dt
        if planner.phase < 0:
            planner.phase += 2 * np.pi

        tracking_error = (
            abs(leg_target - avg_leg_pos) / self.MAX_LEG_EXTENSION +
            abs(arm_target - avg_arm_pos) / self.MAX_ARM_SWING +
            abs(waist_target - self.waist_pos) / self.MAX_WAIST_ANGLE
        ) / 3
        reward -= tracking_error * 0.3 * self.dt

        # Crash penalty
        if terminated:
            if termination_reason == "energy_exhausted":
                reward -= 10.0
            else:
                reward -= 50.0

        return reward

    def close(self):
        pass


# Register environment
gym.register(
    id="PumpFoilHybrid-v0",
    entry_point="foil_env.pump_foil_env_hybrid:PumpFoilEnvHybrid",
    max_episode_steps=6000,
)

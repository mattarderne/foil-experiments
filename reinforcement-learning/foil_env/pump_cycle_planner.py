"""
Pump Cycle Planner - MPC-inspired trajectory generator for coordinated pumping.

This module provides structured short-term planning for arm-leg coordination,
inspired by the ACMPC (Actor-Critic MPC) approach. Instead of a full differentiable
MPC solver, we use a parametric trajectory generator that outputs coordinated
limb targets over one pump cycle.

The RL agent learns the parameters of the pump cycle, while this planner
ensures arm-leg synchronization is maintained structurally.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PumpCycleParams:
    """Parameters defining a pump cycle trajectory."""
    frequency: float = 2.5      # Pump frequency (Hz)
    leg_amplitude: float = 1.0  # Leg amplitude (0-1, fraction of MAX_LEG_EXTENSION)
    arm_amplitude: float = 1.0  # Arm amplitude (0-1, fraction of MAX_ARM_SWING)
    arm_phase_offset: float = 0.0  # Arm phase relative to legs (radians)
    waist_amplitude: float = 0.3   # Waist amplitude (0-1, fraction of MAX_WAIST_ANGLE)
    waist_phase_offset: float = np.pi/4  # Waist leads legs by ~45 deg


class PumpCyclePlanner:
    """
    MPC-inspired trajectory planner for coordinated pumping motion.

    This planner generates smooth, coordinated trajectories for legs, arms, and waist
    that maintain proper phase relationships. The RL agent controls the high-level
    parameters (amplitude, frequency) while this planner handles the coordination.

    Key insight from ACMPC: Structure the short-term planning (one pump cycle)
    while letting RL explore the parameter space.
    """

    def __init__(
        self,
        max_leg_ext: float = 0.15,
        max_arm_swing: float = 1.5,
        max_waist_angle: float = 0.3,
        dt: float = 0.01,
    ):
        self.max_leg_ext = max_leg_ext
        self.max_arm_swing = max_arm_swing
        self.max_waist_angle = max_waist_angle
        self.dt = dt

        # Internal state
        self.phase = 0.0
        self.params = PumpCycleParams()

    def reset(self):
        """Reset planner state."""
        self.phase = 0.0

    def set_params(self, params: PumpCycleParams):
        """Update cycle parameters from RL agent."""
        self.params = params

    def set_params_from_action(self, action: np.ndarray):
        """
        Convert RL action to pump cycle parameters.

        Action space (5D), all in [-1, 1]:
            [0]: frequency modifier -> 1.5 to 3.5 Hz
            [1]: leg amplitude -> 0.3 to 1.0
            [2]: arm amplitude -> 0.3 to 1.0
            [3]: arm phase offset -> -π/4 to π/4
            [4]: waist engagement -> 0.0 to 1.0
        """
        # Map [-1, 1] to appropriate ranges
        self.params.frequency = 2.5 + action[0] * 1.0  # 1.5 to 3.5 Hz
        self.params.leg_amplitude = 0.65 + action[1] * 0.35  # 0.3 to 1.0
        self.params.arm_amplitude = 0.65 + action[2] * 0.35  # 0.3 to 1.0
        self.params.arm_phase_offset = action[3] * np.pi / 4  # -45 to +45 deg
        self.params.waist_amplitude = (action[4] + 1) / 2  # 0.0 to 1.0

    def step(self) -> Tuple[float, float, float, float, float, float]:
        """
        Advance one timestep and return target positions and velocities.

        Returns:
            (leg_pos_target, leg_vel_target, arm_pos_target, arm_vel_target,
             waist_pos_target, waist_vel_target)

        All positions in actual units (m, rad), velocities in m/s or rad/s.
        """
        p = self.params
        omega = 2 * np.pi * p.frequency

        # Update phase
        self.phase += omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # Leg trajectory (sinusoidal)
        leg_pos = p.leg_amplitude * self.max_leg_ext * np.sin(self.phase)
        leg_vel = p.leg_amplitude * self.max_leg_ext * omega * np.cos(self.phase)

        # Arm trajectory (same phase by default, can be offset)
        arm_phase = self.phase + p.arm_phase_offset
        arm_pos = p.arm_amplitude * self.max_arm_swing * np.sin(arm_phase)
        arm_vel = p.arm_amplitude * self.max_arm_swing * omega * np.cos(arm_phase)

        # Waist trajectory (slightly leads legs for thrust generation)
        waist_phase = self.phase + p.waist_phase_offset
        waist_pos = p.waist_amplitude * self.max_waist_angle * np.sin(waist_phase)
        waist_vel = p.waist_amplitude * self.max_waist_angle * omega * np.cos(waist_phase)

        return leg_pos, leg_vel, arm_pos, arm_vel, waist_pos, waist_vel

    def get_horizon_trajectory(self, horizon_steps: int = 50) -> dict:
        """
        Generate trajectory over a planning horizon (like MPC).

        This could be used for visualization or more sophisticated planning.

        Args:
            horizon_steps: Number of steps to plan ahead

        Returns:
            Dict with trajectory arrays for each limb
        """
        p = self.params
        omega = 2 * np.pi * p.frequency

        t = np.arange(horizon_steps) * self.dt
        phases = self.phase + omega * t

        trajectory = {
            'leg_pos': p.leg_amplitude * self.max_leg_ext * np.sin(phases),
            'leg_vel': p.leg_amplitude * self.max_leg_ext * omega * np.cos(phases),
            'arm_pos': p.arm_amplitude * self.max_arm_swing * np.sin(phases + p.arm_phase_offset),
            'arm_vel': p.arm_amplitude * self.max_arm_swing * omega * np.cos(phases + p.arm_phase_offset),
            'waist_pos': p.waist_amplitude * self.max_waist_angle * np.sin(phases + p.waist_phase_offset),
            'waist_vel': p.waist_amplitude * self.max_waist_angle * omega * np.cos(phases + p.waist_phase_offset),
            'time': t,
        }

        return trajectory


class HybridPumpController:
    """
    Hybrid controller combining MPC-like trajectory planning with RL-learned modulation.

    Architecture:
    1. PumpCyclePlanner generates coordinated baseline trajectories
    2. RL agent outputs:
       - Cycle parameters (frequency, amplitudes, phases)
       - Residual adjustments for fine control
    3. Final commands = planned trajectory + residual

    This mirrors ACMPC's approach where MPC provides structure and RL provides adaptivity.
    """

    def __init__(
        self,
        max_leg_ext: float = 0.15,
        max_arm_swing: float = 1.5,
        max_waist_angle: float = 0.3,
        max_leg_vel: float = 2.0,
        max_arm_vel: float = 12.0,
        max_waist_vel: float = 2.0,
        dt: float = 0.01,
        residual_scale: float = 0.3,  # How much RL can deviate from plan
    ):
        self.planner = PumpCyclePlanner(
            max_leg_ext=max_leg_ext,
            max_arm_swing=max_arm_swing,
            max_waist_angle=max_waist_angle,
            dt=dt,
        )

        self.max_leg_vel = max_leg_vel
        self.max_arm_vel = max_arm_vel
        self.max_waist_vel = max_waist_vel
        self.residual_scale = residual_scale

    def reset(self):
        self.planner.reset()

    def compute_action(
        self,
        rl_action: np.ndarray,
        current_leg_pos: float,
        current_arm_pos: float,
        current_waist_pos: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute final velocity commands combining planner + RL residual.

        Args:
            rl_action: 10D action from RL agent
                [0-4]: Cycle parameters (freq, leg_amp, arm_amp, arm_phase, waist_amp)
                [5-9]: Residual adjustments (left_leg, right_leg, left_arm, right_arm, waist)
            current_*_pos: Current limb positions for tracking

        Returns:
            (left_leg_vel, right_leg_vel, left_arm_vel, right_arm_vel, waist_vel)
            All normalized to [-1, 1] for the environment
        """
        # Parse RL action
        cycle_params = rl_action[:5]
        residuals = rl_action[5:10] if len(rl_action) >= 10 else np.zeros(5)

        # Update planner parameters
        self.planner.set_params_from_action(cycle_params)

        # Get planned trajectory targets
        leg_target, leg_vel_target, arm_target, arm_vel_target, waist_target, waist_vel_target = self.planner.step()

        # Compute tracking velocity commands (PD-like)
        Kp_leg = 10.0
        Kp_arm = 8.0
        Kp_waist = 6.0

        # Base velocity from trajectory + tracking error
        leg_vel = leg_vel_target + Kp_leg * (leg_target - current_leg_pos)
        arm_vel = arm_vel_target + Kp_arm * (arm_target - current_arm_pos)
        waist_vel = waist_vel_target + Kp_waist * (waist_target - current_waist_pos)

        # Add RL residuals (allows fine-tuning beyond the plan)
        leg_vel += residuals[0] * self.residual_scale * self.max_leg_vel
        arm_vel += residuals[2] * self.residual_scale * self.max_arm_vel
        waist_vel += residuals[4] * self.residual_scale * self.max_waist_vel

        # Normalize to [-1, 1]
        left_leg_vel_norm = np.clip(leg_vel / self.max_leg_vel, -1, 1)
        right_leg_vel_norm = np.clip(leg_vel / self.max_leg_vel, -1, 1)  # Symmetric
        left_arm_vel_norm = np.clip(arm_vel / self.max_arm_vel, -1, 1)
        right_arm_vel_norm = np.clip(arm_vel / self.max_arm_vel, -1, 1)  # Symmetric
        waist_vel_norm = np.clip(waist_vel / self.max_waist_vel, -1, 1)

        # Apply asymmetric residuals for left/right if specified
        left_leg_vel_norm += residuals[0] * self.residual_scale
        right_leg_vel_norm += residuals[1] * self.residual_scale
        left_arm_vel_norm += residuals[2] * self.residual_scale
        right_arm_vel_norm += residuals[3] * self.residual_scale

        return (
            np.clip(left_leg_vel_norm, -1, 1),
            np.clip(right_leg_vel_norm, -1, 1),
            np.clip(left_arm_vel_norm, -1, 1),
            np.clip(right_arm_vel_norm, -1, 1),
            np.clip(waist_vel_norm, -1, 1),
        )

    def get_planned_trajectory(self, horizon_steps: int = 50) -> dict:
        """Get the current planned trajectory for visualization."""
        return self.planner.get_horizon_trajectory(horizon_steps)

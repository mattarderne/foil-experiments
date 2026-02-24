"""
Survival-focused pump foil environment with configurable rewards.

Reward modes:
- distance: reward = vx * dt (distance traveled)
- velocity: reward = vx / target_vx (maintain speed)
- pump: reward includes bonus for pumping frequency
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, List

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum


def analyze_pumping(leg_positions: List[float], dt: float = 0.01) -> dict:
    """
    Analyze pumping behavior from leg position history.
    Uses velocity zero-crossings to detect oscillation frequency.
    """
    if len(leg_positions) < 20:
        return {'freq': 0, 'range_pct': 0, 'amplitude': 0}

    leg = np.array(leg_positions)
    leg_vel = np.diff(leg) / dt
    leg_range = leg.max() - leg.min()
    range_pct = leg_range / 0.30 * 100

    vel_crossings = np.where(np.diff(np.signbit(leg_vel)))[0]
    if len(vel_crossings) >= 2:
        avg_half_period = np.mean(np.diff(vel_crossings)) * dt
        freq = 0.5 / avg_half_period if avg_half_period > 0 else 0
    else:
        freq = 0

    return {'freq': freq, 'range_pct': range_pct, 'amplitude': leg_range / 2}


class PumpFoilEnvSurvival(PumpFoilEnvCurriculum):
    """
    Configurable reward environment for pump foil experiments.
    """

    MAX_STEPS = 6000  # 60 seconds
    MAX_ENERGY = 18000.0  # 3x budget for 60s

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        reward_mode: str = "distance",  # "distance", "velocity", "pump", "combined"
        pump_freq_target: float = 2.5,  # Target pump frequency for pump reward
        velocity_weight: float = 1.0,
        pump_weight: float = 0.5,
        exploration_bonus: float = 0.0,  # Entropy bonus for more exploration
    ):
        super().__init__(
            config=config,
            dt=dt,
            render_mode=render_mode,
            curriculum_phase=2,
            agent_blend=1.0,
        )
        self.MAX_ENERGY = 18000.0
        self.reward_mode = reward_mode
        self.pump_freq_target = pump_freq_target
        self.velocity_weight = velocity_weight
        self.pump_weight = pump_weight
        self.exploration_bonus = exploration_bonus

        # Track leg history for pump detection
        self.leg_history = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.leg_history = []
        return obs, info

    def step(self, action):
        # Track leg position before step
        self.leg_history.append((self.left_leg_pos + self.right_leg_pos) / 2)

        # Parent step handles physics
        obs, reward, terminated, truncated, info = super().step(action)

        # Override reward based on mode (parent reward is ignored)
        reward = self._compute_reward_custom(terminated)

        return obs, reward, terminated, truncated, info

    def _compute_reward_custom(self, terminated: bool) -> float:
        """Compute reward based on configured mode."""

        if self.reward_mode == "distance":
            # Simple: distance traveled this step
            return self.state.vx * self.dt

        elif self.reward_mode == "velocity":
            # Reward maintaining velocity (penalize slowing down)
            velocity_ratio = self.state.vx / self.TARGET_VELOCITY
            return velocity_ratio * self.dt * self.velocity_weight

        elif self.reward_mode == "pump":
            # Reward pumping behavior
            reward = self.state.vx * self.dt  # Base distance

            # Add pump frequency bonus (computed over recent history)
            if len(self.leg_history) >= 50:
                recent = self.leg_history[-50:]
                stats = analyze_pumping(recent, self.dt)

                # Reward being close to target frequency
                if stats['freq'] > 0:
                    freq_error = abs(stats['freq'] - self.pump_freq_target)
                    freq_bonus = max(0, 1.0 - freq_error / self.pump_freq_target)
                    reward += freq_bonus * self.pump_weight * self.dt

                # Reward amplitude
                amp_bonus = min(1.0, stats['range_pct'] / 50) * 0.5
                reward += amp_bonus * self.pump_weight * self.dt

            return reward

        elif self.reward_mode == "combined":
            # Combine velocity maintenance + pump bonus
            reward = 0

            # Velocity component (strong signal for maintaining speed)
            velocity_ratio = self.state.vx / self.TARGET_VELOCITY
            velocity_reward = velocity_ratio * self.velocity_weight
            reward += velocity_reward * self.dt

            # Pump component
            if len(self.leg_history) >= 50:
                recent = self.leg_history[-50:]
                stats = analyze_pumping(recent, self.dt)

                if stats['freq'] > 0.5:  # Only reward if actually pumping
                    freq_error = abs(stats['freq'] - self.pump_freq_target)
                    freq_bonus = max(0, 1.0 - freq_error / self.pump_freq_target)
                    reward += freq_bonus * self.pump_weight * self.dt

            # Leg movement bonus (encourage activity)
            leg_vel = abs(self.left_leg_vel) + abs(self.right_leg_vel)
            activity_bonus = min(1.0, leg_vel / self.MAX_LEG_VELOCITY) * 0.1
            reward += activity_bonus * self.dt

            return reward

        else:
            return self.state.vx * self.dt  # Default to distance

    def _compute_reward(self, prev_vx, prev_z, leg_force, pitch_torque, terminated, termination_reason):
        """Override parent - use custom reward instead."""
        return self._compute_reward_custom(terminated)


# Register environments
gym.register(
    id="PumpFoilSurvival-v0",
    entry_point="foil_env.pump_foil_env_survival:PumpFoilEnvSurvival",
    max_episode_steps=6000,
)

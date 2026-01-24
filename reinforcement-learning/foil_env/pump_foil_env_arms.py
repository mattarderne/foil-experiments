"""
Arm-sync curriculum environment for pump foil.

Curriculum phases:
- Phase 0: Forced leg pumping + forced arm swing (opposite phase)
- Phase 1: Blended control with arm sync bonus
- Phase 2: Full control with arm sync reward

Arms swing opposite to legs for deweighting:
- When legs push down, arms swing up (body lifts)
- When legs pull up, arms swing down
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum


class PumpFoilEnvArms(PumpFoilEnvCurriculum):
    """
    Environment with arm-swing curriculum.

    Key insight: Arms swinging opposite to legs amplifies deweighting.
    """

    MAX_STEPS = 6000  # 60 seconds
    MAX_ENERGY = 18000.0

    PUMP_FREQUENCY = 2.5  # Hz
    ARM_AMPLITUDE = 0.7   # Fraction of max swing

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        arm_phase: int = 0,  # 0=forced, 1=blended, 2=free
        arm_blend: float = 0.0,  # 0=fully forced, 1=fully agent
        reward_mode: str = "distance",
    ):
        super().__init__(
            config=config,
            dt=dt,
            render_mode=render_mode,
            curriculum_phase=2,  # Always full leg control from parent
            agent_blend=1.0,
        )
        self.MAX_ENERGY = 18000.0
        self.arm_phase = arm_phase
        self.arm_blend = arm_blend
        self.reward_mode = reward_mode

        # Track for analysis
        self.leg_history = []
        self.arm_history = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.leg_history = []
        self.arm_history = []
        return obs, info

    def _get_auto_arm_velocity(self) -> float:
        """
        Get automatic arm velocity - opposite phase to legs.

        Legs: A * sin(ωt)
        Arms: -A * sin(ωt) = A * sin(ωt + π)

        So arm velocity = -leg velocity (scaled for arm range)
        """
        omega = 2 * np.pi * self.PUMP_FREQUENCY

        # Target arm position (opposite to leg target)
        # Legs oscillate around extended position (~0.1m)
        # Arms oscillate around neutral (0)
        target_arm = -self.ARM_AMPLITUDE * self.MAX_ARM_SWING * np.sin(self.pump_phase)

        # Current arm position (average of both arms)
        current_arm = (self.left_arm_pos + self.right_arm_pos) / 2

        # PD control to track target
        kp = 15.0
        error = target_arm - current_arm
        target_vel = kp * error

        return np.clip(target_vel, -self.MAX_ARM_VELOCITY, self.MAX_ARM_VELOCITY)

    def step(self, action):
        """Override step to add arm curriculum."""
        action = np.array(action, dtype=np.float32)

        # Track positions before step
        self.leg_history.append((self.left_leg_pos + self.right_leg_pos) / 2)
        self.arm_history.append((self.left_arm_pos + self.right_arm_pos) / 2)

        # --- ARM CURRICULUM ---
        if self.arm_phase == 0:
            # Forced arm swing (opposite to legs)
            auto_arm_vel = self._get_auto_arm_velocity()
            arm_action = auto_arm_vel / self.MAX_ARM_VELOCITY
            action[2] = arm_action  # left arm
            action[3] = arm_action  # right arm

        elif self.arm_phase == 1:
            # Blended control
            auto_arm_vel = self._get_auto_arm_velocity()
            auto_action = auto_arm_vel / self.MAX_ARM_VELOCITY
            agent_action = (action[2] + action[3]) / 2
            blended = (1 - self.arm_blend) * auto_action + self.arm_blend * agent_action
            action[2] = blended
            action[3] = blended
        # Phase 2: full agent control (no modification)

        # Parent step handles physics
        obs, reward, terminated, truncated, info = super().step(action)

        # Override reward
        reward = self._compute_reward_custom(terminated, info)

        return obs, reward, terminated, truncated, info

    def _compute_reward_custom(self, terminated: bool, info: dict) -> float:
        """Compute reward with arm sync bonus."""

        # Base reward: distance traveled
        base_reward = self.state.vx * self.dt

        if self.reward_mode == "distance":
            return base_reward

        elif self.reward_mode == "sync":
            # Reward arm-leg synchronization (opposite phase)
            reward = base_reward

            if len(self.leg_history) >= 10 and len(self.arm_history) >= 10:
                # Check if arms are moving opposite to legs
                leg_vel = self.left_leg_vel + self.right_leg_vel
                arm_vel = self.left_arm_vel + self.right_arm_vel

                # Negative correlation = good (opposite motion)
                # Normalize by max velocities
                leg_norm = leg_vel / (2 * self.MAX_LEG_VELOCITY)
                arm_norm = arm_vel / (2 * self.MAX_ARM_VELOCITY)

                # Reward opposite motion (negative product)
                sync_score = -leg_norm * arm_norm  # Positive when opposite
                sync_bonus = max(0, sync_score) * 0.5 * self.dt
                reward += sync_bonus

            return reward

        elif self.reward_mode == "combined":
            # Velocity + sync bonus
            reward = 0

            # Velocity component
            velocity_ratio = self.state.vx / self.TARGET_VELOCITY
            reward += velocity_ratio * self.dt

            # Arm sync bonus
            leg_vel = self.left_leg_vel + self.right_leg_vel
            arm_vel = self.left_arm_vel + self.right_arm_vel

            if abs(leg_vel) > 0.1:  # Only when actually pumping
                leg_norm = leg_vel / (2 * self.MAX_LEG_VELOCITY)
                arm_norm = arm_vel / (2 * self.MAX_ARM_VELOCITY)
                sync_score = -leg_norm * arm_norm
                sync_bonus = max(0, sync_score) * 0.3 * self.dt
                reward += sync_bonus

            # Arm movement bonus (encourage arm activity)
            arm_activity = (abs(self.left_arm_vel) + abs(self.right_arm_vel)) / (2 * self.MAX_ARM_VELOCITY)
            activity_bonus = min(1.0, arm_activity) * 0.1 * self.dt
            reward += activity_bonus

            return reward

        return base_reward

    def _compute_reward(self, prev_vx, prev_z, leg_force, pitch_torque, terminated, termination_reason):
        """Override parent - use custom reward."""
        return self._compute_reward_custom(terminated, {})


def analyze_arm_sync(leg_positions, arm_positions, dt=0.01):
    """Analyze arm-leg synchronization."""
    if len(leg_positions) < 20 or len(arm_positions) < 20:
        return {'correlation': 0, 'arm_range_pct': 0}

    leg = np.array(leg_positions)
    arm = np.array(arm_positions)

    # Compute velocities
    leg_vel = np.diff(leg) / dt
    arm_vel = np.diff(arm) / dt

    # Correlation of velocities (should be negative for opposite motion)
    if len(leg_vel) > 10 and np.std(leg_vel) > 0 and np.std(arm_vel) > 0:
        correlation = np.corrcoef(leg_vel, arm_vel)[0, 1]
    else:
        correlation = 0

    arm_range = arm.max() - arm.min()
    arm_range_pct = arm_range / 0.60 * 100  # MAX_ARM_SWING is ~0.3 each way

    return {
        'correlation': correlation,  # Negative = good sync
        'arm_range_pct': arm_range_pct,
    }


# Register environment
gym.register(
    id="PumpFoilArms-v0",
    entry_point="foil_env.pump_foil_env_arms:PumpFoilEnvArms",
    max_episode_steps=6000,
)

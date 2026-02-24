"""
Exploration environment with configurable reward weights.
Extends curriculum env for full compatibility with baseline checkpoints.

Used to test which reward features show steepest improvement when
fine-tuning from a trained checkpoint.
"""

import numpy as np
from typing import Optional, Dict, Any

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum


class PumpFoilEnvExplore(PumpFoilEnvCurriculum):
    """
    Exploration environment with configurable reward weights.
    Extends curriculum env to ensure physics compatibility.

    Key features to test:
    - arm_amplitude: Reward larger arm movements
    - arm_speed: Reward fast arm motion
    - arm_freq: Reward arms at 2Hz pattern
    - arm_leg_sync: Reward arm-leg same-phase correlation
    - jerk: Penalty for jerky motion
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        # Reward weights (configurable)
        arm_amplitude_weight: float = 0.0,
        arm_speed_weight: float = 0.0,
        arm_freq_weight: float = 0.0,
        arm_leg_sync_weight: float = 0.0,
        jerk_weight: float = 0.0,
    ):
        # Initialize as phase 2 (full control) to match baseline training
        super().__init__(
            config=config,
            dt=dt,
            render_mode=render_mode,
            curriculum_phase=2,
            agent_blend=1.0,
        )

        # Store reward weights
        self.arm_amplitude_weight = arm_amplitude_weight
        self.arm_speed_weight = arm_speed_weight
        self.arm_freq_weight = arm_freq_weight
        self.arm_leg_sync_weight = arm_leg_sync_weight
        self.jerk_weight = jerk_weight

        # Action history for jerk calculation
        self.action_history = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.action_history = []
        return obs, info

    def step(self, action):
        # Track action for jerk calculation
        self.action_history.append(action.copy())
        if len(self.action_history) > 3:
            self.action_history.pop(0)

        # Use parent step for physics (ensures compatibility)
        obs, reward, terminated, truncated, info = super().step(action)

        # Add experimental rewards
        extra_reward = self._compute_extra_reward()
        reward += extra_reward

        return obs, reward, terminated, truncated, info

    def _compute_extra_reward(self) -> float:
        """Compute additional rewards based on configurable weights."""
        extra = 0.0

        # 1. Arm amplitude reward
        arm_amplitude = max(abs(self.left_arm_pos), abs(self.right_arm_pos)) / self.MAX_ARM_SWING
        extra += self.arm_amplitude_weight * arm_amplitude

        # 2. Arm speed reward
        arm_speed = (abs(self.left_arm_vel) + abs(self.right_arm_vel)) / 2
        arm_speed_norm = arm_speed / self.MAX_ARM_VELOCITY
        extra += self.arm_speed_weight * arm_speed_norm

        # 3. Arm frequency reward (same 2Hz pattern as legs)
        expected_sign = np.sin(2 * np.pi * self.PUMP_FREQUENCY * self.time)
        actual_arm_sign = self.left_arm_pos / (self.MAX_ARM_SWING + 0.01)
        arm_phase_match = expected_sign * actual_arm_sign
        extra += self.arm_freq_weight * max(0, arm_phase_match)

        # 4. Arm-leg synchronization reward (same phase)
        leg_norm = self.left_leg_pos / (self.MAX_LEG_EXTENSION + 0.01)
        arm_norm = self.left_arm_pos / (self.MAX_ARM_SWING + 0.01)
        sync_reward = leg_norm * arm_norm  # Positive when same sign
        extra += self.arm_leg_sync_weight * max(0, sync_reward)

        # 5. Jerk penalty
        if len(self.action_history) >= 3:
            a0, a1, a2 = self.action_history[-3], self.action_history[-2], self.action_history[-1]
            jerk = np.mean(np.abs(a2 - 2*a1 + a0))
            extra -= self.jerk_weight * jerk

        return extra

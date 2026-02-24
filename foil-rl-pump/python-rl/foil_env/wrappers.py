"""
Gym wrappers for pump foil environment.

Includes wrappers for forcing arm patterns, curriculum learning, etc.
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any


class ArmPatternWrapper(gym.Wrapper):
    """
    Wrapper that forces arms to follow a sinusoidal pattern.

    Uses P-control to track target arm positions, overriding
    agent actions for arms (action indices 2 and 3).
    """

    def __init__(
        self,
        env: gym.Env,
        frequency: float = 2.0,
        amplitude: float = 0.5,
        center: float = np.pi / 2,  # DOWN position
        alternating: bool = True,
        p_gain: float = 10.0,
    ):
        """
        Args:
            env: Underlying pump foil environment
            frequency: Arm oscillation frequency (Hz)
            amplitude: Arm swing amplitude (rad)
            center: Center position for arms (rad)
            alternating: If True, arms move opposite; if False, in sync
            p_gain: P-controller gain for tracking
        """
        super().__init__(env)
        self.frequency = frequency
        self.amplitude = amplitude
        self.center = center
        self.alternating = alternating
        self.p_gain = p_gain
        self.time = 0.0
        self.dt = getattr(env.unwrapped, 'dt', 0.01)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.time = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Copy action to avoid modifying original
        action = np.array(action, dtype=np.float32)

        # Compute target arm positions
        phase = 2 * np.pi * self.frequency * self.time
        left_target = self.center + self.amplitude * np.sin(phase)

        if self.alternating:
            right_target = self.center - self.amplitude * np.sin(phase)
        else:
            right_target = left_target

        # Get current arm positions from env
        env = self.env.unwrapped
        left_arm_pos = getattr(env, 'left_arm_pos', self.center)
        right_arm_pos = getattr(env, 'right_arm_pos', self.center)

        # P-control to compute velocity commands
        max_arm_vel = getattr(env, 'MAX_ARM_VELOCITY', 12.0)
        left_vel_cmd = self.p_gain * (left_target - left_arm_pos)
        right_vel_cmd = self.p_gain * (right_target - right_arm_pos)

        # Normalize to action space [-1, 1]
        action[2] = np.clip(left_vel_cmd / max_arm_vel, -1, 1)
        action[3] = np.clip(right_vel_cmd / max_arm_vel, -1, 1)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time += self.dt

        # Add wrapper info
        info['arm_wrapper'] = {
            'left_target': left_target,
            'right_target': right_target,
        }

        return obs, reward, terminated, truncated, info


class LegPatternWrapper(gym.Wrapper):
    """
    Wrapper that forces legs to follow a sinusoidal pattern.

    Useful for testing arm/waist control while legs pump automatically.
    """

    def __init__(
        self,
        env: gym.Env,
        frequency: float = 2.0,
        amplitude: float = 0.8,  # Fraction of max leg extension
        p_gain: float = 10.0,
    ):
        """
        Args:
            env: Underlying pump foil environment
            frequency: Leg pumping frequency (Hz)
            amplitude: Fraction of MAX_LEG_EXTENSION to use (0-1)
            p_gain: P-controller gain
        """
        super().__init__(env)
        self.frequency = frequency
        self.amplitude = amplitude
        self.p_gain = p_gain
        self.time = 0.0
        self.dt = getattr(env.unwrapped, 'dt', 0.01)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.time = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.array(action, dtype=np.float32)

        env = self.env.unwrapped
        max_leg_ext = getattr(env, 'MAX_LEG_EXTENSION', 0.15)
        max_leg_vel = getattr(env, 'MAX_LEG_VELOCITY', 2.0)

        # Target position
        phase = 2 * np.pi * self.frequency * self.time
        target = self.amplitude * max_leg_ext * np.sin(phase)

        # Get current positions
        left_leg_pos = getattr(env, 'left_leg_pos', 0.0)
        right_leg_pos = getattr(env, 'right_leg_pos', 0.0)

        # P-control
        left_vel_cmd = self.p_gain * (target - left_leg_pos)
        right_vel_cmd = self.p_gain * (target - right_leg_pos)

        # Override leg actions
        action[0] = np.clip(left_vel_cmd / max_leg_vel, -1, 1)
        action[1] = np.clip(right_vel_cmd / max_leg_vel, -1, 1)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time += self.dt

        return obs, reward, terminated, truncated, info


class FullPumpPatternWrapper(gym.Wrapper):
    """
    Wrapper that forces both legs and arms into a coordinated pump pattern.

    Useful for testing physics or demonstrating ideal pumping.
    """

    def __init__(
        self,
        env: gym.Env,
        frequency: float = 2.0,
        leg_amplitude: float = 0.8,
        arm_amplitude: float = 0.5,
        arm_phase_offset: float = np.pi / 2,  # Arms lead legs by 90°
        p_gain: float = 10.0,
    ):
        super().__init__(env)
        self.frequency = frequency
        self.leg_amplitude = leg_amplitude
        self.arm_amplitude = arm_amplitude
        self.arm_phase_offset = arm_phase_offset
        self.p_gain = p_gain
        self.time = 0.0
        self.dt = getattr(env.unwrapped, 'dt', 0.01)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.time = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.array(action, dtype=np.float32)
        env = self.env.unwrapped

        phase = 2 * np.pi * self.frequency * self.time

        # Legs
        max_leg_ext = getattr(env, 'MAX_LEG_EXTENSION', 0.15)
        max_leg_vel = getattr(env, 'MAX_LEG_VELOCITY', 2.0)
        leg_target = self.leg_amplitude * max_leg_ext * np.sin(phase)
        left_leg_pos = getattr(env, 'left_leg_pos', 0.0)
        leg_vel_cmd = self.p_gain * (leg_target - left_leg_pos)
        action[0] = np.clip(leg_vel_cmd / max_leg_vel, -1, 1)
        action[1] = action[0]  # Symmetric legs

        # Arms
        arm_center = np.pi / 2
        max_arm_vel = getattr(env, 'MAX_ARM_VELOCITY', 12.0)
        arm_phase = phase + self.arm_phase_offset
        left_arm_target = arm_center + self.arm_amplitude * np.sin(arm_phase)
        right_arm_target = arm_center - self.arm_amplitude * np.sin(arm_phase)

        left_arm_pos = getattr(env, 'left_arm_pos', arm_center)
        right_arm_pos = getattr(env, 'right_arm_pos', arm_center)

        action[2] = np.clip(self.p_gain * (left_arm_target - left_arm_pos) / max_arm_vel, -1, 1)
        action[3] = np.clip(self.p_gain * (right_arm_target - right_arm_pos) / max_arm_vel, -1, 1)

        # Waist: leave to agent (action[4] unchanged)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time += self.dt

        return obs, reward, terminated, truncated, info


class NormalizedObsWrapper(gym.ObservationWrapper):
    """
    Wrapper that ensures observations are properly normalized.

    Can be used to catch issues with observation scaling.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0):
        super().__init__(env)
        self.clip = clip

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(obs, -self.clip, self.clip)

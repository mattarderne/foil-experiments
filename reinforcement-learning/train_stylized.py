#!/usr/bin/env python3
"""
Train with style rewards for more natural/human-like motion.

Adds:
- Action smoothness (penalize jittery control)
- Energy penalty (penalize high torques)
- Jerk penalty (penalize sudden accelerations)
- Symmetry bonus (arms move together)
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, '.')

from foil_env.config import TrainingConfig, FoilConfig, LimbConfig
from foil_env.checkpoint_utils import save_checkpoint


class StyleRewardWrapper(gym.Wrapper):
    """Wrapper that adds style rewards for natural motion."""

    def __init__(self, env,
                 smoothness_weight=0.1,
                 energy_weight=0.01,
                 jerk_weight=0.05,
                 symmetry_weight=0.02):
        super().__init__(env)
        self.smoothness_weight = smoothness_weight
        self.energy_weight = energy_weight
        self.jerk_weight = jerk_weight
        self.symmetry_weight = symmetry_weight

        # History for computing derivatives
        self.prev_action = None
        self.prev_leg_pos = None
        self.prev_leg_vel = None
        self.prev_arm_angles = None

        # Stats for logging
        self.episode_style_rewards = {
            'smoothness': 0.0,
            'energy': 0.0,
            'jerk': 0.0,
            'symmetry': 0.0,
        }

    def reset(self, **kwargs):
        self.prev_action = None
        self.prev_leg_pos = None
        self.prev_leg_vel = None
        self.prev_arm_angles = None
        self.episode_style_rewards = {k: 0.0 for k in self.episode_style_rewards}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current state from env (not state object)
        env = self.env
        leg_pos = (env.left_leg_pos + env.right_leg_pos) / 2
        leg_vel = (env.left_leg_vel + env.right_leg_vel) / 2
        left_arm = getattr(env, 'left_arm_angle', 0.0)
        right_arm = getattr(env, 'right_arm_angle', 0.0)

        style_reward = 0.0

        # 1. Action smoothness - penalize rapid action changes
        if self.prev_action is not None:
            action_delta = np.sum((np.array(action) - np.array(self.prev_action))**2)
            smoothness_penalty = -self.smoothness_weight * action_delta
            style_reward += smoothness_penalty
            self.episode_style_rewards['smoothness'] += smoothness_penalty

        # 2. Energy penalty - penalize high action magnitudes (proxy for torque)
        action_magnitude = np.sum(np.array(action)**2)
        energy_penalty = -self.energy_weight * action_magnitude
        style_reward += energy_penalty
        self.episode_style_rewards['energy'] += energy_penalty

        # 3. Jerk penalty - penalize sudden changes in velocity
        if self.prev_leg_vel is not None:
            leg_jerk = (leg_vel - self.prev_leg_vel)**2
            jerk_penalty = -self.jerk_weight * leg_jerk
            style_reward += jerk_penalty
            self.episode_style_rewards['jerk'] += jerk_penalty

        # 4. Symmetry bonus - reward arms moving together (if applicable)
        if hasattr(env, 'left_arm_angle') and hasattr(env, 'right_arm_angle'):
            arm_diff = (left_arm - right_arm)**2
            symmetry_bonus = -self.symmetry_weight * arm_diff
            style_reward += symmetry_bonus
            self.episode_style_rewards['symmetry'] += symmetry_bonus

        # Update history
        self.prev_action = action.copy() if hasattr(action, 'copy') else action
        self.prev_leg_pos = leg_pos
        self.prev_leg_vel = leg_vel
        self.prev_arm_angles = (left_arm, right_arm)

        # Add style reward to total
        total_reward = reward + style_reward

        # Add style info
        info['style_reward'] = style_reward
        info['style_breakdown'] = self.episode_style_rewards.copy()

        return obs, total_reward, terminated, truncated, info


def train_stylized(
    total_timesteps: int = 1_000_000,
    output_dir: str = "checkpoints/stylized",
    smoothness: float = 0.1,
    energy: float = 0.01,
    jerk: float = 0.05,
    symmetry: float = 0.02,
    finetune_from: str = None,
):
    """Train with style rewards."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    # Use 1000cm² foil (our best performer)
    foil = FoilConfig(
        S=0.10,        # 1000 cm²
        S_stab=0.02,
        stab_angle=-4.0,
        AR=8,
        Cd0=0.013,
        Cd_mast=0.12,
        pump_thrust_efficiency=0.20,
    )

    limbs = LimbConfig(
        arm_start_position=np.pi/2,
        arm_centering_spring=2.0,
    )

    config = TrainingConfig(
        foil=foil,
        limbs=limbs,
        curriculum_phase=2,
        target_velocity=4.5,
        pump_frequency=2.0,
    )

    os.makedirs(output_dir, exist_ok=True)

    class BaseEnv(PumpFoilEnvCurriculum):
        MAX_STEPS = 6000  # 60s
        MAX_ENERGY = 50000
        TARGET_ALTITUDE = -0.15
        MIN_VELOCITY = 1.5

    class CheckpointCallback(BaseCallback):
        def __init__(self, total_timesteps, save_path, config):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.config = config
            self.saved = set()
            self.percentages = [1, 25, 50, 75, 100]

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    path = os.path.join(self.save_path, f"model_{p}pct")
                    save_checkpoint(self.model, self.config, path)
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = BaseEnv(
                config=config.foil.to_dict(),
                limb_config=config.limbs,
                curriculum_phase=config.curriculum_phase,
            )
            env = StyleRewardWrapper(
                env,
                smoothness_weight=smoothness,
                energy_weight=energy,
                jerk_weight=jerk,
                symmetry_weight=symmetry,
            )
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"\n{'='*60}")
    print(f"TRAINING WITH STYLE REWARDS - 1000cm²")
    print(f"{'='*60}")
    print(f"Style weights:")
    print(f"  Smoothness: {smoothness}")
    print(f"  Energy:     {energy}")
    print(f"  Jerk:       {jerk}")
    print(f"  Symmetry:   {symmetry}")
    if finetune_from:
        print(f"Fine-tuning from: {finetune_from}")
    print(f"Output: {output_dir}")

    n_envs = 4
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if finetune_from:
        # Load existing model and set new environment
        model = PPO.load(finetune_from, env=env)
        # Use lower learning rate for fine-tuning
        model.learning_rate = 1e-4
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    callback = CheckpointCallback(
        total_timesteps=total_timesteps,
        save_path=output_dir,
        config=config,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    print(f"\nDone! Saved to {output_dir}/")
    return output_dir


def evaluate_stylized(checkpoint_path: str, n_episodes: int = 5):
    """Evaluate stylized model and compare motion quality."""
    from stable_baselines3 import PPO
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
    from foil_env.config import LimbConfig

    foil_dict = {
        'S': 0.10, 'S_stab': 0.02, 'stab_angle': -4.0, 'AR': 8,
        'mast_length': 0.7, 'riding_depth': 0.2,
        'Cd0': 0.013, 'Cd_mast': 0.12, 'pump_thrust_efficiency': 0.20,
    }

    limbs = LimbConfig(arm_start_position=np.pi/2, arm_centering_spring=2.0)

    class BaseEnv(PumpFoilEnvCurriculum):
        MAX_STEPS = 6000
        MAX_ENERGY = 50000
        TARGET_ALTITUDE = -0.15
        MIN_VELOCITY = 1.5

    base_env = BaseEnv(config=foil_dict, limb_config=limbs, curriculum_phase=2)
    model = PPO.load(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"EVALUATING STYLIZED MODEL")
    print(f"{'='*60}")

    all_metrics = []

    for ep in range(n_episodes):
        obs, _ = base_env.reset(seed=ep)
        done = False
        t = 0.0

        # Track motion quality metrics
        actions = []
        leg_positions = []
        leg_vels = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = base_env.step(action)
            done = terminated or truncated

            actions.append(action.copy())
            leg_positions.append((base_env.left_leg_pos + base_env.right_leg_pos) / 2)
            leg_vels.append((base_env.left_leg_vel + base_env.right_leg_vel) / 2)
            t += 0.01

        # Compute motion quality metrics
        actions = np.array(actions)
        leg_positions = np.array(leg_positions)
        leg_vels = np.array(leg_vels)

        # Action smoothness (lower = smoother)
        if len(actions) > 1:
            action_deltas = np.diff(actions, axis=0)
            action_jitter = np.mean(np.sum(action_deltas**2, axis=1))
        else:
            action_jitter = 0.0

        # Velocity smoothness (lower = smoother)
        if len(leg_vels) > 1:
            vel_deltas = np.diff(leg_vels)
            vel_jitter = np.mean(vel_deltas**2)
        else:
            vel_jitter = 0.0

        # Energy usage
        energy_usage = np.mean(np.sum(actions**2, axis=1)) if len(actions) > 0 else 0.0

        # Pumping frequency (FFT)
        if len(leg_positions) > 100:
            fft = np.fft.fft(leg_positions - np.mean(leg_positions))
            freqs = np.fft.fftfreq(len(leg_positions), 0.01)
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            peak_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC
            pump_freq = positive_freqs[peak_idx]
        else:
            pump_freq = 0.0

        reason = "timeout" if not terminated else ("breach" if base_env.state.z >= 0.2 else "touchdown")

        metrics = {
            'duration': t,
            'reason': reason,
            'action_jitter': action_jitter,
            'vel_jitter': vel_jitter,
            'energy': energy_usage,
            'pump_freq': pump_freq,
        }
        all_metrics.append(metrics)

        print(f"Ep {ep+1}: {t:.1f}s ({reason}) | freq={pump_freq:.2f}Hz | "
              f"action_jitter={action_jitter:.4f} | vel_jitter={vel_jitter:.4f}")

    # Summary
    print(f"\n--- Summary ---")
    avg_duration = np.mean([m['duration'] for m in all_metrics])
    avg_jitter = np.mean([m['action_jitter'] for m in all_metrics])
    avg_vel_jitter = np.mean([m['vel_jitter'] for m in all_metrics])
    avg_energy = np.mean([m['energy'] for m in all_metrics])
    avg_freq = np.mean([m['pump_freq'] for m in all_metrics])
    timeouts = sum(1 for m in all_metrics if m['reason'] == 'timeout')

    print(f"Avg duration: {avg_duration:.1f}s | Timeouts: {timeouts}/{n_episodes}")
    print(f"Avg pump freq: {avg_freq:.2f}Hz")
    print(f"Motion quality:")
    print(f"  Action jitter: {avg_jitter:.4f} (lower = smoother)")
    print(f"  Velocity jitter: {avg_vel_jitter:.4f} (lower = smoother)")
    print(f"  Energy usage: {avg_energy:.4f} (lower = more efficient)")

    return all_metrics


def compare_models(baseline_path: str, stylized_path: str, n_episodes: int = 10):
    """Compare baseline vs stylized model motion quality."""
    print(f"\n{'='*60}")
    print(f"COMPARING BASELINE vs STYLIZED")
    print(f"{'='*60}")

    print(f"\n--- BASELINE ---")
    baseline_metrics = evaluate_stylized(baseline_path, n_episodes)

    print(f"\n--- STYLIZED ---")
    stylized_metrics = evaluate_stylized(stylized_path, n_episodes)

    # Comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")

    def avg(metrics, key):
        return np.mean([m[key] for m in metrics])

    print(f"{'Metric':<20} {'Baseline':>12} {'Stylized':>12} {'Change':>12}")
    print("-" * 56)

    for key in ['duration', 'action_jitter', 'vel_jitter', 'energy', 'pump_freq']:
        b = avg(baseline_metrics, key)
        s = avg(stylized_metrics, key)
        if b > 0:
            change = ((s - b) / b) * 100
            print(f"{key:<20} {b:>12.4f} {s:>12.4f} {change:>+11.1f}%")
        else:
            print(f"{key:<20} {b:>12.4f} {s:>12.4f} {'N/A':>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
    parser.add_argument("--output", "-o", type=str, default="checkpoints/stylized")
    parser.add_argument("--smoothness", type=float, default=0.05)  # Lower default
    parser.add_argument("--energy", type=float, default=0.005)     # Lower default
    parser.add_argument("--jerk", type=float, default=0.02)        # Lower default
    parser.add_argument("--symmetry", type=float, default=0.01)    # Lower default
    parser.add_argument("--finetune", "-f", type=str, default=None, help="Fine-tune from checkpoint")
    parser.add_argument("--eval", "-e", type=str, default=None, help="Evaluate checkpoint")
    parser.add_argument("--compare", "-c", nargs=2, metavar=('BASELINE', 'STYLIZED'),
                        help="Compare two checkpoints")
    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare[0], args.compare[1])
    elif args.eval:
        evaluate_stylized(args.eval)
    else:
        train_stylized(
            args.timesteps, args.output,
            args.smoothness, args.energy, args.jerk, args.symmetry,
            args.finetune
        )

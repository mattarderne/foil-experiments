#!/usr/bin/env python3
"""
Train pump foil RL with corrected physics for sustained flight.

Key physics changes:
- Cd0: 0.008 (reduced from 0.013 for modern clean foils)
- Cd_mast: 0.06 (reduced from 0.12 for streamlined masts)
- pump_thrust_efficiency: 0.20 (properly models pump thrust conversion)

These parameters allow sustained flight (60s+) with proper pumping.

Usage:
    python train_sustained.py -t 1000000 -o checkpoints/sustained_v1
"""

import os
import sys
import argparse
import numpy as np
sys.path.insert(0, '.')

from foil_env.config import TrainingConfig, BIG_FOIL, LimbConfig
from foil_env.checkpoint_utils import save_checkpoint

CHECKPOINT_PERCENTAGES = [1, 10, 25, 50, 75, 100]


def create_sustained_config():
    """Create training config with sustained-flight physics."""
    return TrainingConfig(
        foil=BIG_FOIL,  # Uses corrected physics parameters
        limbs=LimbConfig(
            arm_start_position=np.pi/2,
            arm_centering_spring=2.0,
        ),
        curriculum_phase=2,  # Full agent control
        target_velocity=4.5,
        pump_frequency=2.0,
    )


def train_sustained(
    total_timesteps: int = 1_000_000,
    output_dir: str = "checkpoints/sustained_v1",
):
    """Train with corrected sustained-flight physics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    config = create_sustained_config()
    os.makedirs(output_dir, exist_ok=True)

    # Extended environment for longer flights
    class SustainedEnv(PumpFoilEnvCurriculum):
        MAX_STEPS = 6000   # 60s max episode
        MAX_ENERGY = 50000  # More energy for long flights
        TARGET_ALTITUDE = -0.15  # Start lower to avoid immediate breach

    class CheckpointCallback(BaseCallback):
        def __init__(self, total_timesteps, save_path, percentages, config):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.save_path = save_path
            self.percentages = percentages
            self.config = config
            self.saved = set()

        def _on_step(self):
            pct = (self.num_timesteps / self.total_timesteps) * 100
            for p in self.percentages:
                if p not in self.saved and pct >= p:
                    path = os.path.join(self.save_path, f"model_{p}pct")
                    save_checkpoint(self.model, self.config, path)
                    print(f"\n>>> Saved: {path}.zip ({p}%)")
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = SustainedEnv(
                config=config.foil.to_dict(),
                limb_config=config.limbs,
                curriculum_phase=config.curriculum_phase,
            )
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print("=" * 60)
    print("SUSTAINED FLIGHT TRAINING")
    print("=" * 60)
    foil_dict = config.foil.to_dict()
    print(f"\nPhysics parameters:")
    print(f"  Foil: {foil_dict['S']*10000:.0f} cm²")
    print(f"  Cd0: {foil_dict['Cd0']} (reduced for modern foils)")
    print(f"  Cd_mast: {foil_dict['Cd_mast']} (reduced for carbon masts)")
    print(f"  pump_thrust_efficiency: {foil_dict['pump_thrust_efficiency']}")
    print(f"\nGoal: Learn to sustain flight for 60s+ through pumping")
    print(f"Training: {total_timesteps:,} steps")
    print()

    n_envs = 4
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,    # Longer rollouts for long episodes
        batch_size=256,
        n_epochs=10,
        gamma=0.995,     # Higher discount for long-horizon
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callback = CheckpointCallback(
        total_timesteps=total_timesteps,
        save_path=output_dir,
        percentages=CHECKPOINT_PERCENTAGES,
        config=config,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    print(f"\nDone! Checkpoints in {output_dir}/")
    return output_dir


def evaluate(checkpoint_path: str, n_episodes: int = 5):
    """Evaluate trained model."""
    from stable_baselines3 import PPO
    from foil_env.checkpoint_utils import load_checkpoint
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
    from scipy.fft import fft, fftfreq

    model, config = load_checkpoint(checkpoint_path)

    class EvalEnv(PumpFoilEnvCurriculum):
        MAX_STEPS = 6000
        MAX_ENERGY = 50000
        TARGET_ALTITUDE = -0.15

    env = EvalEnv(
        config=config.foil.to_dict(),
        limb_config=config.limbs,
        curriculum_phase=config.curriculum_phase,
    )

    print(f"\n=== Evaluating {checkpoint_path} ===")
    foil = config.foil.to_dict()
    print(f"Foil: {foil['S']*10000:.0f}cm², pump_eff={foil['pump_thrust_efficiency']}")

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        leg_positions = []
        t = 0.0
        dt = 0.01

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            leg_positions.append((env.left_leg_pos + env.right_leg_pos) / 2)
            t += dt

        # Analyze pumping
        leg_arr = np.array(leg_positions)
        if len(leg_arr) > 100:
            yf = fft(leg_arr - leg_arr.mean())
            xf = fftfreq(len(leg_arr), dt)
            idx = np.argmax(np.abs(yf[1:len(yf)//2])) + 1
            freq = abs(xf[idx])
            amp = (leg_arr.max() - leg_arr.min()) / 2 / env.MAX_LEG_EXTENSION * 100
        else:
            freq, amp = 0, 0

        # Termination reason
        reason = "timeout"
        if terminated:
            if env.state.z >= 0.2:
                reason = "breach"
            elif env.state.z <= -0.5:
                reason = "touchdown"
            elif env.state.vx < 1.5:
                reason = "stall"

        results.append({'duration': t, 'freq': freq, 'amp': amp, 'reason': reason})
        print(f"Ep {ep+1}: {t:.1f}s ({reason}), {freq:.2f}Hz, {amp:.0f}%")

    print(f"\n--- Summary ---")
    print(f"Avg duration: {np.mean([r['duration'] for r in results]):.1f}s")
    print(f"Max duration: {max([r['duration'] for r in results]):.1f}s")
    print(f"60s timeouts: {sum(1 for r in results if r['reason']=='timeout')}/{n_episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
    parser.add_argument("--output", "-o", default="checkpoints/sustained_v1")
    parser.add_argument("--eval", "-e", type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train_sustained(args.timesteps, args.output)

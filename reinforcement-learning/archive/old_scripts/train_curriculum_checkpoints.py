#!/usr/bin/env python3
"""
Train curriculum model with checkpoints at 1%, 25%, 50%, 75%, 100%.
For creating training evolution video.
"""

import os
import sys
sys.path.insert(0, '.')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

# Training foil config (more stable)
FOIL_CONFIG = {'S_stab': 0.035, 'stab_angle': -4.0, 'S': 0.18, 'AR': 8}

TOTAL_TIMESTEPS = 500_000
CHECKPOINT_DIR = "checkpoints/curriculum_evolution"


class PercentageCheckpointCallback(BaseCallback):
    """Save checkpoints at specific percentages of training."""

    def __init__(self, total_timesteps, save_path, percentages=[1, 25, 50, 75, 100]):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.save_path = save_path
        self.percentages = percentages
        self.checkpoints_saved = set()

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        current_pct = (self.num_timesteps / self.total_timesteps) * 100

        for pct in self.percentages:
            if pct not in self.checkpoints_saved and current_pct >= pct:
                path = os.path.join(self.save_path, f"model_{pct}pct")
                self.model.save(path)
                print(f"\n>>> Saved checkpoint at {pct}%: {path}.zip")
                self.checkpoints_saved.add(pct)

        return True


def make_env(rank: int = 0):
    def _init():
        env = PumpFoilEnvCurriculum(config=FOIL_CONFIG, curriculum_phase=2)
        env = Monitor(env)
        env.reset(seed=42 + rank)
        return env
    return _init


def train():
    print(f"Training curriculum model with checkpoints")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Foil config: {FOIL_CONFIG}")
    print(f"  Checkpoints at: 1%, 25%, 50%, 75%, 100%")
    print()

    # Create vectorized env (4 parallel for faster local training)
    env = DummyVecEnv([make_env(i) for i in range(4)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    # Checkpoint callback
    checkpoint_callback = PercentageCheckpointCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        save_path=CHECKPOINT_DIR,
        percentages=[1, 25, 50, 75, 100]
    )

    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )

    print(f"\nTraining complete!")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}/")


if __name__ == '__main__':
    train()

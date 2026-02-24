#!/usr/bin/env python3
"""
Standard training script for pump foil RL model.

Saves checkpoints at 1%, 25%, 50%, 75%, 100% of training for evolution videos.
Each checkpoint includes a config JSON file for reproducibility.

Usage:
    python train.py                          # Local training (500k steps)
    python train.py --timesteps 1000000      # Custom timesteps
    python train.py --modal                  # Run on Modal GPU (faster)
"""

import os
import sys
import argparse
import numpy as np
sys.path.insert(0, '.')

from foil_env.config import TrainingConfig, FoilConfig, LimbConfig, TRAINING_FOIL, DEFAULT_LIMBS
from foil_env.checkpoint_utils import save_checkpoint

# Default training configuration
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    foil=TRAINING_FOIL,
    limbs=DEFAULT_LIMBS,
    curriculum_phase=2,  # Full agent control
)

CHECKPOINT_PERCENTAGES = [1, 25, 50, 75, 100]


def train_local(
    total_timesteps: int = 500_000,
    output_dir: str = "checkpoints/run",
    config: TrainingConfig = None,
    finetune_from: str = None,
):
    """Train locally with checkpoint saving."""
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    if config is None:
        config = DEFAULT_TRAINING_CONFIG

    os.makedirs(output_dir, exist_ok=True)

    class PercentageCheckpointCallback(BaseCallback):
        """Save checkpoints at specific percentages with config."""
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
                    print(f"\n>>> Checkpoint saved: {path}.zip + config ({p}%)")
                    self.saved.add(p)
            return True

    def make_env(rank):
        def _init():
            env = PumpFoilEnvCurriculum(
                config=config.foil.to_dict(),
                limb_config=config.limbs,
                curriculum_phase=config.curriculum_phase,
            )
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"=== PUMP FOIL TRAINING ===")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Foil: S={config.foil.S}m², S_stab={config.foil.S_stab}m², AR={config.foil.AR}")
    print(f"Arms: start={np.degrees(config.limbs.arm_start_position):.0f}°, spring={config.limbs.arm_centering_spring}")
    if finetune_from:
        print(f"Fine-tuning from: {finetune_from}")
    print(f"Output: {output_dir}/")
    print(f"Checkpoints at: {CHECKPOINT_PERCENTAGES}%")
    print()

    n_envs = 2 if finetune_from else 4  # Fewer envs for fine-tuning to reduce memory
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    if finetune_from:
        # Manual weight copy workaround for Python 3.14 compatibility
        # PPO.load() has issues, so we create new model and copy weights
        print(f"Loading baseline model (manual weight copy)...")

        # Ensure path doesn't have double .zip
        if not finetune_from.endswith('.zip'):
            finetune_from += '.zip'

        # Create new model with same architecture
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower LR for fine-tuning
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

        # Load old model weights (CPU to avoid issues)
        import torch
        import zipfile
        import io

        with zipfile.ZipFile(finetune_from, 'r') as zf:
            with zf.open('policy.pth') as f:
                old_state = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)

        model.policy.load_state_dict(old_state)
        print(f"Loaded weights from {finetune_from}")
    else:
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
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    callback = PercentageCheckpointCallback(
        total_timesteps=total_timesteps,
        save_path=output_dir,
        percentages=CHECKPOINT_PERCENTAGES,
        config=config,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    print(f"\nTraining complete! Checkpoints in {output_dir}/")
    print()
    print("=" * 50)
    print(">>> REMINDER: Create evolution video with:")
    print(f"    python create_video.py --checkpoint-dir {output_dir} -o evolution.mp4")
    print("=" * 50)
    return output_dir


def train_modal(
    total_timesteps: int = 500_000,
    output_dir: str = "checkpoints/run",
    config: TrainingConfig = None,
):
    """Train on Modal GPU with checkpoint saving."""
    import modal
    import json

    if config is None:
        config = DEFAULT_TRAINING_CONFIG

    app = modal.App("pump-foil-train")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "gymnasium==0.29.1",
            "stable-baselines3[extra]==2.1.0",
            "torch",
            "numpy",
            "scipy",  # For metrics
        )
        .add_local_dir("foil_env", "/app/foil_env")
    )

    @app.function(image=image, gpu="A10G", timeout=7200, serialized=True)
    def train_remote(timesteps, config_dict, percentages):
        import os
        import sys
        sys.path.insert(0, "/app")

        import numpy as np
        import json
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
        from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum
        from foil_env.config import TrainingConfig, FoilConfig, LimbConfig

        # Reconstruct config from dict
        foil = FoilConfig(**config_dict['foil'])
        limbs = LimbConfig(**config_dict['limbs'])
        config = TrainingConfig(
            foil=foil,
            limbs=limbs,
            curriculum_phase=config_dict.get('curriculum_phase', 2),
        )

        save_path = "/tmp/checkpoints"
        os.makedirs(save_path, exist_ok=True)

        class PercentageCheckpointCallback(BaseCallback):
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
                        self.model.save(path)
                        # Save config alongside
                        self.config.save(f"{path}_config.json")
                        print(f"\n>>> Checkpoint: {p}%")
                        self.saved.add(p)
                return True

        def make_env(rank):
            def _init():
                env = PumpFoilEnvCurriculum(
                    config=config.foil.to_dict(),
                    limb_config=config.limbs,
                    curriculum_phase=config.curriculum_phase,
                )
                env = Monitor(env)
                env.reset(seed=42 + rank)
                return env
            return _init

        print(f"Training {timesteps:,} steps on GPU...")
        env = DummyVecEnv([make_env(i) for i in range(8)])

        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=1, device="cuda",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

        callback = PercentageCheckpointCallback(timesteps, save_path, percentages, config)
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

        # Return all checkpoint files AND config files
        checkpoints = {}
        for p in percentages:
            model_path = f"{save_path}/model_{p}pct.zip"
            config_path = f"{save_path}/model_{p}pct_config.json"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    checkpoints[f"{p}_model"] = f.read()
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    checkpoints[f"{p}_config"] = f.read()
        return checkpoints

    # Run on Modal
    print("Launching training on Modal GPU...")
    print(f"Foil: S={config.foil.S}m², S_stab={config.foil.S_stab}m², AR={config.foil.AR}")
    print(f"Arms: start={np.degrees(config.limbs.arm_start_position):.0f}°, spring={config.limbs.arm_centering_spring}")
    os.makedirs(output_dir, exist_ok=True)

    # Serialize config to pass to Modal
    config_dict = config.to_dict()

    with app.run():
        checkpoints = train_remote.remote(total_timesteps, config_dict, CHECKPOINT_PERCENTAGES)

    # Save locally - both model and config files
    for pct in CHECKPOINT_PERCENTAGES:
        model_key = f"{pct}_model"
        config_key = f"{pct}_config"

        if model_key in checkpoints:
            model_path = f"{output_dir}/model_{pct}pct.zip"
            with open(model_path, "wb") as f:
                f.write(checkpoints[model_key])
            print(f"Saved: {model_path}")

        if config_key in checkpoints:
            config_path = f"{output_dir}/model_{pct}pct_config.json"
            with open(config_path, "w") as f:
                f.write(checkpoints[config_key])
            print(f"Saved: {config_path}")

    print(f"\nTraining complete! Checkpoints in {output_dir}/")
    print()
    print("=" * 50)
    print(">>> REMINDER: Create evolution video with:")
    print(f"    python create_video.py --checkpoint-dir {output_dir} -o evolution.mp4")
    print("=" * 50)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pump foil RL model")
    parser.add_argument("--timesteps", "-t", type=int, default=500_000,
                       help="Total training timesteps (default: 500000)")
    parser.add_argument("--output", "-o", default="checkpoints/run",
                       help="Output directory for checkpoints")
    parser.add_argument("--modal", action="store_true",
                       help="Run on Modal GPU instead of local")
    parser.add_argument("--finetune", "-f", type=str, default=None,
                       help="Path to checkpoint to fine-tune from (e.g., checkpoints/ent005_2M/model_100pct.zip)")
    args = parser.parse_args()

    if args.modal:
        train_modal(args.timesteps, args.output)
    else:
        train_local(args.timesteps, args.output, finetune_from=args.finetune)

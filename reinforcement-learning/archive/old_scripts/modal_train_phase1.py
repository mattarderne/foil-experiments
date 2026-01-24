"""
Modal GPU training with checkpoint saving and resume support.

Phase 1: Steady glide with pitch control (arms, legs, waist).

Run with:
  modal run modal_train_phase1.py
  modal run modal_train_phase1.py --resume checkpoints/phase1_500k.zip
"""

import modal

app = modal.App("pump-foil-phase1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium==0.29.1",
        "stable-baselines3[extra]==2.1.0",
        "torch",
        "numpy",
        "tensorboard",
        "tqdm",
        "rich",
    )
    .add_local_dir("foil_env", "/app/foil_env")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
)
def train_phase1(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    checkpoint_interval: int = 100_000,
    resume_model_bytes: bytes = None,
):
    """Train Phase 1: steady glide with pitch control."""
    import os
    import sys
    import io
    sys.path.insert(0, "/app")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    from foil_env.pump_foil_env_4limb_phase1 import PumpFoilEnv4LimbPhase1

    def make_env(rank: int = 0, seed: int = 42):
        def _init():
            env = PumpFoilEnv4LimbPhase1()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    print(f"Training Phase 1: Steady Glide with Pitch Control")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Checkpoint interval: {checkpoint_interval:,}")
    print(f"  Resume from checkpoint: {resume_model_bytes is not None}")

    env = DummyVecEnv([make_env(i, 42) for i in range(n_envs)])

    # Load or create model
    if resume_model_bytes is not None:
        print("\nLoading model from checkpoint...")
        # Save bytes to temp file
        with open("/tmp/resume_model.zip", "wb") as f:
            f.write(resume_model_bytes)
        model = PPO.load("/tmp/resume_model.zip", env=env, device="cuda")
        print("Resumed from checkpoint!")
    else:
        print("\nCreating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device="cuda",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
        )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval // n_envs,  # Adjust for n_envs
        save_path="/tmp/checkpoints",
        name_prefix="phase1",
    )

    # Custom callback to track progress
    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_mean_reward = -float('inf')

        def _on_step(self):
            if self.n_calls % 10000 == 0:
                # Log some stats
                if len(self.model.ep_info_buffer) > 0:
                    mean_reward = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                    mean_length = sum(ep['l'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                    print(f"  Step {self.n_calls}: mean_reward={mean_reward:.1f}, mean_length={mean_length:.0f}")
            return True

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, ProgressCallback()],
        progress_bar=True,
    )

    # Save final model
    model.save("/tmp/phase1_final")
    print(f"\nFinal model saved")

    # Collect all checkpoints
    checkpoints = {}
    checkpoint_dir = "/tmp/checkpoints"
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".zip"):
                with open(os.path.join(checkpoint_dir, f), "rb") as file:
                    checkpoints[f] = file.read()

    # Add final model
    with open("/tmp/phase1_final.zip", "rb") as f:
        checkpoints["phase1_final.zip"] = f.read()

    return checkpoints


@app.local_entrypoint()
def main(
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    resume: str = None,
):
    """Run Phase 1 training on Modal GPU."""
    import os

    print(f"Launching Modal GPU training (Phase 1)...")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Environments: {n_envs}")

    # Load resume checkpoint if provided
    resume_bytes = None
    if resume and os.path.exists(resume):
        print(f"  Resume from: {resume}")
        with open(resume, "rb") as f:
            resume_bytes = f.read()

    # Run training
    checkpoints = train_phase1.remote(
        total_timesteps=timesteps,
        n_envs=n_envs,
        resume_model_bytes=resume_bytes,
    )

    # Save all checkpoints locally
    os.makedirs("checkpoints/phase1", exist_ok=True)
    for name, data in checkpoints.items():
        path = f"checkpoints/phase1/{name}"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")

    print(f"\nAll checkpoints saved to checkpoints/phase1/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    main(timesteps=args.timesteps, n_envs=args.n_envs, resume=args.resume)

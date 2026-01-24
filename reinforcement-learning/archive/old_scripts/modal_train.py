"""
Modal GPU training for 4-Limb Pump Foil RL.

Run with: modal run modal_train.py
"""

import modal

app = modal.App("pump-foil-rl-4limb")

# Create image with all dependencies
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
    gpu="A10G",  # Or "T4" for cheaper, "A100" for faster
    timeout=7200,  # 2 hours max
)
def train_4limb(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_steps: int = 1024,
):
    """Train 4-limb pump foil agent on GPU."""
    import os
    import sys
    sys.path.insert(0, "/app")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback

    from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb

    def make_env(rank: int = 0, seed: int = 42):
        def _init():
            env = PumpFoilEnv4Limb()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    print(f"Training 4-Limb PPO agent on GPU")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")

    # Create vectorized environment (DummyVecEnv for Modal compatibility)
    env = DummyVecEnv([make_env(i, 42) for i in range(n_envs)])

    # Create model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
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

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    # Save model
    model_path = "/tmp/pump_foil_4limb"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Return model bytes
    with open(f"{model_path}.zip", "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    timesteps: int = 2_000_000,
    n_envs: int = 8,
):
    """Run training on Modal GPU."""
    import os

    print(f"Launching Modal GPU training...")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Environments: {n_envs}")

    # Run training
    model_bytes = train_4limb.remote(
        total_timesteps=timesteps,
        n_envs=n_envs,
    )

    # Save model locally
    os.makedirs("checkpoints", exist_ok=True)
    output_path = "checkpoints/model_4limb_modal.zip"
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    args = parser.parse_args()

    main(timesteps=args.timesteps, n_envs=args.n_envs)

"""
Modal GPU training with checkpoint saving at quartiles.
Saves models at 1%, 25%, 50%, 75%, 100% of training.
"""

import modal

app = modal.App("pump-foil-rl-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium==0.29.1",
        "stable-baselines3[extra]==2.1.0",
        "torch",
        "numpy",
        "tensorboard",
    )
    .add_local_dir("foil_env", "/app/foil_env")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
)
def train_with_checkpoints(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
):
    """Train and save checkpoints at quartiles."""
    import os
    import sys
    sys.path.insert(0, "/app")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb

    # Checkpoints at these percentages
    checkpoint_pcts = [0.01, 0.25, 0.50, 0.75, 1.0]
    checkpoint_steps = [int(p * total_timesteps) for p in checkpoint_pcts]
    saved_checkpoints = {}

    class CheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_checkpoint_idx = 0

        def _on_step(self) -> bool:
            if self.next_checkpoint_idx < len(checkpoint_steps):
                if self.num_timesteps >= checkpoint_steps[self.next_checkpoint_idx]:
                    pct = checkpoint_pcts[self.next_checkpoint_idx]
                    path = f"/tmp/checkpoint_{int(pct*100)}pct"
                    self.model.save(path)
                    print(f"Saved checkpoint at {pct*100:.0f}% ({self.num_timesteps} steps)")
                    saved_checkpoints[pct] = path
                    self.next_checkpoint_idx += 1
            return True

    def make_env(rank: int = 0, seed: int = 42):
        def _init():
            env = PumpFoilEnv4Limb()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    print(f"Training with checkpoint saving")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Checkpoints at: {checkpoint_pcts}")

    env = DummyVecEnv([make_env(i, 42) for i in range(n_envs)])

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
        device="cuda",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    callback = CheckpointCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Return all checkpoint files
    result = {}
    for pct, path in saved_checkpoints.items():
        with open(f"{path}.zip", "rb") as f:
            result[f"{int(pct*100)}pct"] = f.read()

    return result


@app.local_entrypoint()
def main(timesteps: int = 1_000_000):
    """Run training and save all checkpoints locally."""
    import os

    print(f"Launching Modal training with checkpoints...")
    checkpoints = train_with_checkpoints.remote(total_timesteps=timesteps)

    os.makedirs("checkpoints/training_stages", exist_ok=True)
    for name, data in checkpoints.items():
        path = f"checkpoints/training_stages/model_{name}.zip"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

"""
Modal training for curriculum pump foil with frequency reward.
Uses training foil (more stable) + pumping incentive rewards.
"""

import modal

app = modal.App("pump-foil-curriculum")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium==0.29.1",
        "stable-baselines3[extra]==2.1.0",
        "torch",
        "numpy",
    )
    .add_local_dir("foil_env", "/app/foil_env")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
)
def train_curriculum(total_timesteps: int = 1_000_000):
    """Train with stable foil and pumping reward."""
    import sys
    sys.path.insert(0, "/app")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    # Training foil config (more stable)
    config = {'S_stab': 0.035, 'stab_angle': -4.0, 'S': 0.18, 'AR': 8}

    def make_env(rank: int = 0):
        def _init():
            env = PumpFoilEnvCurriculum(config=config, curriculum_phase=2)
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env
        return _init

    print(f"Training curriculum env with pumping reward")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Foil: Training (stable)")

    env = DummyVecEnv([make_env(i) for i in range(8)])

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

    print("\nStarting training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model_path = "/tmp/pump_foil_curriculum"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    with open(f"{model_path}.zip", "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(timesteps: int = 1_000_000):
    """Run training on Modal GPU."""
    import os

    print(f"Launching Modal training...")
    model_bytes = train_curriculum.remote(total_timesteps=timesteps)

    os.makedirs("checkpoints", exist_ok=True)
    output_path = "checkpoints/model_curriculum_pump.zip"
    with open(output_path, "wb") as f:
        f.write(model_bytes)
    print(f"\nSaved: {output_path}")

"""
Training script for the Pump Foil RL environment.

Uses Stable Baselines3 with PPO or SAC algorithms.
"""

import argparse
import os
from datetime import datetime

import numpy as np

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Run: pip install stable-baselines3")

from foil_env import PumpFoilEnv


def make_env(rank: int = 0, seed: int = 0, use_rk4: bool = False):
    """
    Create a wrapped environment.

    Args:
        rank: Environment index for parallel training
        seed: Random seed base
        use_rk4: Use RK4 integration

    Returns:
        Callable that creates the environment
    """
    def _init():
        env = PumpFoilEnv(use_rk4=use_rk4)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    use_rk4: bool = False,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    seed: int = 42,
):
    """
    Train the pump foil agent.

    Args:
        algorithm: "PPO" or "SAC"
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        batch_size: Batch size for updates
        n_steps: Steps per environment before update (PPO only)
        use_rk4: Use RK4 integration
        log_dir: Directory for tensorboard logs
        model_dir: Directory for saved models
        seed: Random seed
    """
    if not HAS_SB3:
        print("Error: stable-baselines3 is required for training")
        return None

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pump_foil_{algorithm.lower()}_{timestamp}"
    log_path = os.path.join(log_dir, run_name)
    model_path = os.path.join(model_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    print(f"Training {algorithm} agent")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Log directory: {log_path}")
    print(f"  Model directory: {model_path}")

    # Create vectorized environment
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, seed, use_rk4) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, seed, use_rk4)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, seed + 1000, use_rk4)])

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=max(10000 // n_envs, 1000),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 5000),
        save_path=model_path,
        name_prefix="checkpoint",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Create model
    if algorithm.upper() == "PPO":
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
            ent_coef=0.01,  # Entropy coefficient for exploration
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=100_000,
            learning_starts=1000,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = os.path.join(model_path, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Pump Foil RL agent")
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs", "-n",
        type=int,
        default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per environment before update (PPO only)"
    )
    parser.add_argument(
        "--use-rk4",
        action="store_true",
        help="Use RK4 integration instead of Euler"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Tensorboard log directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Model save directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    train(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_rk4=args.use_rk4,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

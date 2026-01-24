"""
Train stance-only agent to learn optimal board position.
Quick smoke test to verify stance physics is learnable.
"""

import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from foil_env.stance_env import StanceEnv


class EvalCallback(BaseCallback):
    """Evaluate and log progress during training."""

    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Quick evaluation
            env = StanceEnv()
            rewards = []
            durations = []
            final_stances = []

            for _ in range(5):
                obs, _ = env.reset(seed=42)
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

                rewards.append(total_reward)
                durations.append(info['step_count'] * 0.01)
                final_stances.append(info['stance_offset'])

            mean_reward = np.mean(rewards)
            mean_duration = np.mean(durations)
            mean_stance = np.mean(final_stances)

            print(f"\n[{self.n_calls}] Eval: reward={mean_reward:.1f}, "
                  f"duration={mean_duration:.2f}s, stance={mean_stance:+.3f}m")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"  -> New best!")

        return True


def make_env():
    return StanceEnv()


def main():
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/stance_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training stance-only agent...")
    print(f"Checkpoints: {checkpoint_dir}")

    # Parallel environments
    n_envs = 4
    env = SubprocVecEnv([make_env for _ in range(n_envs)])

    # PPO with small network (simple task)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000 // n_envs,  # Per-env frequency
        save_path=checkpoint_dir,
        name_prefix="stance",
    )
    eval_callback = EvalCallback(eval_freq=5000)

    # Train
    total_timesteps = 50000  # Quick smoke test
    print(f"Training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{checkpoint_dir}/stance_final.zip"
    model.save(final_path)
    print(f"\nSaved final model: {final_path}")

    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)

    env_eval = StanceEnv()
    for seed in [42, 123, 456]:
        obs, _ = env_eval.reset(seed=seed)
        done = False
        total_reward = 0
        stance_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            total_reward += reward
            stance_history.append(info['stance_offset'])
            done = terminated or truncated

        print(f"Seed {seed}: duration={info['step_count']*0.01:.2f}s, "
              f"reward={total_reward:.1f}, "
              f"final_stance={info['stance_offset']:+.3f}m, "
              f"reason={info.get('termination_reason', 'timeout')}")

    print(f"\nCheckpoint dir: {checkpoint_dir}")


if __name__ == "__main__":
    main()

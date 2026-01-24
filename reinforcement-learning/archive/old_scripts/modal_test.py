"""
Modal.com script to test the Pump Foil RL environment.

Usage:
    modal run modal_test.py::test_environment
    modal run modal_test.py::test_training
"""

import modal

# Define the Modal image with dependencies and copy local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium>=0.29.0",
        "stable-baselines3[extra]>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir("foil_env", remote_path="/root/foil_env")
)

app = modal.App("pump-foil-test", image=image)


@app.function(timeout=300)
def test_environment():
    """Test the physics engine and Gymnasium environment."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    from foil_env import PumpFoilEnv, FoilPhysics, CONFIG
    from foil_env.foil_physics import FoilState, compute_trim_angle

    print("=" * 60)
    print("PUMP FOIL RL ENVIRONMENT - MODAL TEST")
    print("=" * 60)

    # Test 1: Physics Engine
    print("\n=== Test 1: Physics Engine ===")
    physics = FoilPhysics()
    state = FoilState()

    print(f"Initial state: z={state.z}m, vx={state.vx}m/s, theta={np.degrees(state.theta):.2f}deg")
    print(f"Trim angle at {state.vx} m/s: {np.degrees(compute_trim_angle(state.vx)):.2f} deg")

    forces = physics.calculate_forces(state)
    print(f"Lift: {forces.L:.1f} N")
    print(f"Drag: {forces.D:.1f} N")
    print(f"AoA: {np.degrees(forces.alpha):.2f} deg")
    print(f"Stalled: {forces.is_stalled}")

    # Sanity checks
    assert forces.L > 500, f"Lift too low: {forces.L}"
    assert forces.D > 0, f"Drag should be positive: {forces.D}"
    assert not forces.is_stalled, "Should not be stalled at trim"
    print("Physics engine tests passed!")

    # Test 2: Physics Integration
    print("\n=== Test 2: Physics Integration ===")
    state = FoilState()
    for i in range(100):
        state = physics.step(state, leg_force=0, pitch_torque=0, dt=0.01)

    print(f"After 1 second (neutral): z={state.z:.3f}m, vx={state.vx:.2f}m/s")
    # With no input, foil should sink slightly and slow down
    assert state.z < 0.2, "Should sink without pumping"
    print("Physics integration tests passed!")

    # Test 3: Gymnasium Environment
    print("\n=== Test 3: Gymnasium Environment ===")
    env = PumpFoilEnv()
    obs, info = env.reset(seed=42)

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial: z={info['z']:.3f}m, vx={info['vx']:.2f}m/s")

    assert obs.shape == (11,), f"Wrong obs shape: {obs.shape}"
    assert env.action_space.shape == (2,), f"Wrong action shape: {env.action_space.shape}"
    print("Environment initialization tests passed!")

    # Test 4: Random Policy Episode
    print("\n=== Test 4: Random Policy Episode ===")
    obs, info = env.reset(seed=42)
    total_reward = 0
    steps = 0

    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"Steps: {steps}")
    print(f"Final: z={info['z']:.3f}m, vx={info['vx']:.2f}m/s")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Termination: {info.get('termination_reason', 'none')}")
    print("Random policy episode completed!")

    # Test 5: Neutral Actions (Glide Test)
    print("\n=== Test 5: Neutral Actions (Glide) ===")
    obs, info = env.reset(seed=42)
    total_reward = 0

    for i in range(200):
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Glide ended at step {i+1}: {info.get('termination_reason', 'truncated')}")
            break

    print(f"Final: z={info['z']:.3f}m, vx={info['vx']:.2f}m/s")
    print(f"Total reward: {total_reward:.1f}")
    print("Glide test completed!")

    # Test 6: Extreme Actions
    print("\n=== Test 6: Extreme Actions ===")
    obs, info = env.reset(seed=42)

    # Full leg push
    for i in range(50):
        action = np.array([1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    print(f"After full leg push: z={info['z']:.3f}m")

    # Full pitch up
    obs, info = env.reset(seed=42)
    for i in range(50):
        action = np.array([0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    print(f"After full pitch up: theta={info['theta_deg']:.1f}deg")
    print("Extreme actions test completed!")

    env.close()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    return {"status": "success", "message": "All environment tests passed"}


@app.function(timeout=600)
def test_training():
    """Run a short training session to verify SB3 integration."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    from foil_env import PumpFoilEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    print("=" * 60)
    print("PUMP FOIL RL - TRAINING TEST")
    print("=" * 60)

    # Create environment
    def make_env():
        return PumpFoilEnv()

    env = DummyVecEnv([make_env for _ in range(4)])
    eval_env = DummyVecEnv([make_env])

    print("\n=== Creating PPO Model ===")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
    )

    # Evaluate before training
    print("\n=== Evaluating Random Policy ===")
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"Before training: {mean_reward_before:.1f} +/- {std_reward_before:.1f}")

    # Train for a short period
    print("\n=== Training (10k steps) ===")
    model.learn(total_timesteps=10_000, progress_bar=True)

    # Evaluate after training
    print("\n=== Evaluating Trained Policy ===")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"After training: {mean_reward_after:.1f} +/- {std_reward_after:.1f}")

    # Check for improvement
    improved = mean_reward_after > mean_reward_before
    print(f"\nImproved: {improved}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("TRAINING TEST COMPLETE!")
    print("=" * 60)

    return {
        "status": "success",
        "before": mean_reward_before,
        "after": mean_reward_after,
        "improved": improved,
    }


@app.local_entrypoint()
def main():
    """Run all tests."""
    print("Running environment tests...")
    result1 = test_environment.remote()
    print(f"\nEnvironment test result: {result1}")

    print("\n" + "-" * 60)
    print("Running training test...")
    result2 = test_training.remote()
    print(f"\nTraining test result: {result2}")

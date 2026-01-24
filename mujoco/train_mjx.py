"""
GPU-accelerated training using MJX and Brax PPO.

This script trains a pump foil policy using:
- MJX for physics simulation (JAX-compiled MuJoCo)
- Brax PPO for policy optimization
- Batched rollouts for high throughput
"""

import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
from brax import envs
from brax.training.agents.ppo import train as ppo_module
from brax.training.agents.ppo import networks as ppo_networks

ppo_train = ppo_module.train
from brax.io import model as brax_model
import functools
from pathlib import Path
from typing import Any, Dict, Tuple
import time
import argparse


class PumpFoilBraxEnv(envs.Env):
    """
    Brax-compatible pump foil environment using MJX.

    This wraps our MJX physics in Brax's environment interface
    so we can use Brax's PPO training.
    """

    def __init__(self, **kwargs):
        # Load MuJoCo model
        model_path = Path(__file__).parent / "rider.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.sys = mjx.put_model(self.mj_model)

        # Cache IDs
        self.board_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "board"
        )

        # Physics constants
        self.rho = 1000.0
        self.g = 9.81
        self.total_mass = sum(self.mj_model.body_mass)
        self.S = 0.18
        self.S_stab = 0.035
        self.AR = 8.0
        self.Cd0 = 0.008
        self.pump_efficiency = 0.20
        self.lift_slope = (2 * jnp.pi) / (1 + 2 / self.AR)
        self.stall_rad = jnp.radians(13.0)
        self.stab_angle_rad = jnp.radians(-4.0)
        self.k_induced = 1 / (jnp.pi * self.AR)

        # Config
        self.initial_height = 0.05
        self.initial_velocity = 4.5
        self.breach_height = 0.4
        self.touchdown_depth = -0.3
        self.stall_velocity = 1.5

    @property
    def observation_size(self) -> int:
        return 17  # Simplified observation

    @property
    def action_size(self) -> int:
        return 3  # leg, arm, waist

    @property
    def backend(self) -> str:
        return 'mjx'

    def reset(self, rng: jnp.ndarray) -> envs.State:
        """Reset to initial state."""
        mjx_data = mjx.make_data(self.sys)

        # Set initial state
        qpos = mjx_data.qpos.at[2].set(self.initial_height)
        qvel = mjx_data.qvel.at[0].set(self.initial_velocity)

        # Initial pitch
        pitch = 0.08
        qpos = qpos.at[3].set(jnp.cos(pitch / 2))
        qpos = qpos.at[5].set(jnp.sin(pitch / 2))

        # Initial joint positions
        qpos = qpos.at[7].set(0.2)   # left_knee
        qpos = qpos.at[8].set(0.2)   # right_knee
        qpos = qpos.at[9].set(0.1)   # left_hip
        qpos = qpos.at[10].set(0.1)  # right_hip
        qpos = qpos.at[11].set(0.1)  # waist

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(self.sys, mjx_data)

        obs = self._get_obs(mjx_data)
        reward = jnp.array(0.0)
        done = jnp.array(0.0)  # Must be float for Brax

        return envs.State(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info={'rng': rng},
        )

    def step(self, state: envs.State, action: jnp.ndarray) -> envs.State:
        """Execute one step."""
        mjx_data = state.pipeline_state
        action = jnp.clip(action, -1.0, 1.0)

        # Apply control
        ctrl = self._action_to_ctrl(action)
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # Compute and apply foil forces
        foil_forces = self._compute_foil_forces(mjx_data)
        xfrc = jnp.zeros((self.mj_model.nbody, 6))
        xfrc = xfrc.at[self.board_id, 0].set(foil_forces[0])
        xfrc = xfrc.at[self.board_id, 2].set(foil_forces[1])
        xfrc = xfrc.at[self.board_id, 4].set(foil_forces[2])
        mjx_data = mjx_data.replace(xfrc_applied=xfrc)

        # Step simulation
        mjx_data = mjx.step(self.sys, mjx_data)

        # Check termination
        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]
        done = jnp.where(
            jnp.logical_or(
                jnp.logical_or(z >= self.breach_height, z <= self.touchdown_depth),
                vx < self.stall_velocity
            ),
            1.0, 0.0
        )

        # Compute reward
        reward = self._compute_reward(mjx_data, action)

        # Get observation
        obs = self._get_obs(mjx_data)

        return state.replace(
            pipeline_state=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _action_to_ctrl(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert action to control."""
        leg = action[0] * 15.0
        arm = action[1] * 10.0
        waist = action[2] * 5.0

        ctrl = jnp.zeros(9)
        ctrl = ctrl.at[0].set(leg / 150)
        ctrl = ctrl.at[1].set(leg / 150)
        ctrl = ctrl.at[2].set(-leg * 0.5 / 200)
        ctrl = ctrl.at[3].set(-leg * 0.5 / 200)
        ctrl = ctrl.at[4].set(waist / 250)
        ctrl = ctrl.at[5].set(arm / 50)
        ctrl = ctrl.at[6].set(arm / 50)
        return ctrl

    def _compute_foil_forces(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Compute foil hydrodynamic forces."""
        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]
        vz = mjx_data.qvel[2]
        qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)

        gamma = jnp.arctan2(vz, vx)
        alpha = theta - gamma
        v_sq = vx * vx + vz * vz

        # Lift/drag
        is_stalled = jnp.abs(alpha) >= self.stall_rad
        Cl = jnp.where(is_stalled,
                       self.lift_slope * self.stall_rad * jnp.sign(alpha),
                       self.lift_slope * alpha)
        Cd = jnp.where(is_stalled,
                       0.05 + (jnp.abs(alpha) - self.stall_rad) * 2.0,
                       self.Cd0 + self.k_induced * Cl * Cl)

        q = 0.5 * self.rho * v_sq
        L = q * self.S * Cl
        D = q * self.S * Cd

        Lz = L * jnp.cos(gamma)
        Dx = -D * jnp.cos(gamma)

        # Stabilizer
        L_stab_z = -q * self.S_stab * self.lift_slope * (theta + self.stab_angle_rad - gamma) * 0.5
        M_stab = -L_stab_z * 0.6

        # Leg force (MuJoCo handles gravity)
        leg_vel = (mjx_data.qvel[6] + mjx_data.qvel[7]) / 2
        leg_force = leg_vel * 20

        Fx = Dx + jnp.where(jnp.abs(vz) > 0.1,
                           self.pump_efficiency * jnp.abs(vz) * jnp.abs(L), 0.0)
        Fz = Lz - leg_force + L_stab_z  # No W - MuJoCo adds gravity

        waist_pos = mjx_data.qpos[11]
        My = M_stab + waist_pos * 100

        return jnp.array([Fx, Fz, My])

    def _get_obs(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Get observation."""
        z = mjx_data.qpos[2]
        vz = mjx_data.qvel[2]
        qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)
        omega = mjx_data.qvel[4]
        vx = mjx_data.qvel[0]

        return jnp.concatenate([
            jnp.array([z, vz, theta, omega, vx]),
            mjx_data.qpos[7:14],  # Joint positions
            mjx_data.qvel[6:11],  # Joint velocities
        ])

    def _compute_reward(self, mjx_data: mjx.Data, action: jnp.ndarray) -> jnp.ndarray:
        """Compute reward."""
        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]
        qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)

        reward = 0.1  # Survival
        reward -= jnp.abs(z) * 0.5  # Altitude
        reward -= jnp.abs(vx - 4.5) * 0.2  # Velocity
        reward -= jnp.maximum(0, jnp.abs(jnp.degrees(theta)) - 15) * 0.1  # Pitch

        return reward


def train(
    num_timesteps: int = 1_000_000,
    episode_length: int = 1000,
    num_envs: int = 256,
    learning_rate: float = 3e-4,
    output_dir: str = "checkpoints/mjx_ppo",
    seed: int = 0,
):
    """Train using Brax PPO."""
    print(f"Training MJX Pump Foil with Brax PPO")
    print(f"  Timesteps: {num_timesteps:,}")
    print(f"  Envs: {num_envs}")
    print(f"  Episode length: {episode_length}")
    print("=" * 50)

    # Create environment
    env = PumpFoilBraxEnv()

    # Train
    start_time = time.time()

    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_envs=num_envs,
        learning_rate=learning_rate,
        seed=seed,
        normalize_observations=True,
        reward_scaling=0.1,
        entropy_cost=0.01,
        discounting=0.99,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=4,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Steps/sec: {num_timesteps / elapsed:.0f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.pkl"
    brax_model.save_params(str(model_path), params)
    print(f"  Saved: {model_path}")

    return make_inference_fn, params, metrics


def evaluate(model_path: str, num_episodes: int = 10):
    """Evaluate a trained model."""
    env = PumpFoilBraxEnv()

    # Load model
    params = brax_model.load_params(model_path)

    # Create inference function
    make_inference_fn, _, _ = ppo_train(
        environment=env,
        num_timesteps=1,  # Just to get the network structure
        episode_length=1,
        num_envs=1,
    )
    inference_fn = make_inference_fn(params)

    print(f"Evaluating {model_path}")
    print("=" * 50)

    rng = random.PRNGKey(0)

    for ep in range(num_episodes):
        rng, ep_rng = random.split(rng)
        state = env.reset(ep_rng)

        total_reward = 0
        steps = 0

        while not state.done and steps < 6000:
            rng, act_rng = random.split(rng)
            action, _ = inference_fn(state.obs, act_rng)
            state = env.step(state, action)
            total_reward += float(state.reward)
            steps += 1

        z = float(state.pipeline_state.qpos[2])
        vx = float(state.pipeline_state.qvel[0])
        duration = steps * 0.01

        print(f"Ep {ep+1}: {duration:.1f}s | z={z:.2f}m | vx={vx:.1f}m/s | r={total_reward:.1f}")


def main():
    parser = argparse.ArgumentParser(description="MJX PPO Training")
    parser.add_argument("-t", "--timesteps", type=int, default=1_000_000,
                        help="Training timesteps")
    parser.add_argument("-n", "--num-envs", type=int, default=256,
                        help="Number of parallel environments")
    parser.add_argument("-o", "--output", default="checkpoints/mjx_ppo",
                        help="Output directory")
    parser.add_argument("--eval", type=str, default=None,
                        help="Evaluate model instead of training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(
            num_timesteps=args.timesteps,
            num_envs=args.num_envs,
            output_dir=args.output,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

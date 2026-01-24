"""
MJX-based Pump Foil Environment for GPU-accelerated training.

Uses JAX for vectorized simulation across many parallel environments.
Key differences from foil_env_mujoco.py:
- All computation in JAX (JIT-compiled, GPU-compatible)
- Batched environments for parallel rollouts
- Custom foil physics ported to JAX
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import mujoco
from mujoco import mjx
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, NamedTuple
from functools import partial


class EnvState(NamedTuple):
    """Environment state for JAX compatibility."""
    mjx_data: mjx.Data
    step_count: jnp.ndarray
    prev_action: jnp.ndarray
    total_energy: jnp.ndarray
    rng: jnp.ndarray


class EnvConfig(NamedTuple):
    """Environment configuration."""
    # Foil physics
    S: float = 0.18              # Wing area (m²)
    S_stab: float = 0.035        # Stabilizer area
    AR: float = 8.0              # Aspect ratio
    stab_angle: float = -4.0     # Stabilizer angle (deg)
    Cd0: float = 0.008           # Parasitic drag
    pump_thrust_efficiency: float = 0.20

    # Environment
    max_steps: int = 6000        # 60s at 100Hz
    initial_velocity: float = 4.5
    initial_height: float = 0.05

    # Termination
    breach_height: float = 0.4
    touchdown_depth: float = -0.3
    stall_velocity: float = 1.5

    # Rewards
    altitude_target: float = 0.0
    velocity_target: float = 4.5


class PumpFoilEnvMJX:
    """
    MJX-based pump foil environment for GPU-accelerated training.

    Supports batched simulation for parallel rollouts.
    """

    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()

        # Load MuJoCo model
        model_path = Path(__file__).parent / "rider.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))

        # Convert to MJX model
        self.mjx_model = mjx.put_model(self.mj_model)

        # Cache body IDs
        self.board_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "board"
        )

        # Precompute physics constants
        self._precompute_physics()

        # Observation and action dimensions
        self.obs_dim = 20
        self.action_dim = 3

    def _precompute_physics(self):
        """Precompute foil physics constants."""
        cfg = self.config

        # Lift slope (Prandtl lifting line)
        self.lift_slope = (2 * jnp.pi) / (1 + 2 / cfg.AR)

        # Stall angle
        self.stall_rad = jnp.radians(13.0)
        self.stab_angle_rad = jnp.radians(cfg.stab_angle)

        # Induced drag factor
        self.k_induced = 1 / (jnp.pi * cfg.AR)

        # Water density
        self.rho = 1000.0

        # Total mass (from model)
        self.total_mass = sum(self.mj_model.body_mass)
        self.g = 9.81

    @partial(jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """Reset environment to initial state."""
        cfg = self.config

        # Create fresh MJX data
        mjx_data = mjx.make_data(self.mjx_model)

        # Set initial position
        qpos = mjx_data.qpos.at[2].set(cfg.initial_height)  # z

        # Set initial velocity
        qvel = mjx_data.qvel.at[0].set(cfg.initial_velocity)  # vx

        # Set initial pitch (quaternion rotation about Y)
        pitch = 0.08
        qpos = qpos.at[3].set(jnp.cos(pitch / 2))  # w
        qpos = qpos.at[5].set(jnp.sin(pitch / 2))  # y

        # Set initial joint positions (slight bend)
        # Indices: 7=left_knee, 8=right_knee, 9=left_hip, 10=right_hip, 11=waist
        qpos = qpos.at[7].set(0.2)   # left_knee
        qpos = qpos.at[8].set(0.2)   # right_knee
        qpos = qpos.at[9].set(0.1)   # left_hip
        qpos = qpos.at[10].set(0.1)  # right_hip
        qpos = qpos.at[11].set(0.1)  # waist

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        # Forward kinematics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.array(0),
            prev_action=jnp.zeros(self.action_dim),
            total_energy=jnp.array(0.0),
            rng=rng,
        )

        obs = self._get_obs(state)
        return state, obs

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: EnvState, action: jnp.ndarray
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one environment step."""
        action = jnp.clip(action, -1.0, 1.0)

        # Apply control
        ctrl = self._action_to_ctrl(action)
        mjx_data = state.mjx_data.replace(ctrl=ctrl)

        # Compute foil forces
        foil_forces = self._compute_foil_forces(mjx_data)

        # Apply external forces to board
        xfrc = jnp.zeros((self.mj_model.nbody, 6))
        xfrc = xfrc.at[self.board_id, 0].set(foil_forces[0])  # Fx
        xfrc = xfrc.at[self.board_id, 2].set(foil_forces[1])  # Fz
        xfrc = xfrc.at[self.board_id, 4].set(foil_forces[2])  # My
        mjx_data = mjx_data.replace(xfrc_applied=xfrc)

        # Step simulation
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        # Update state
        new_step = state.step_count + 1
        new_energy = state.total_energy + jnp.sum(jnp.abs(action)) * 10

        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=new_step,
            prev_action=action,
            total_energy=new_energy,
            rng=state.rng,
        )

        # Get observation
        obs = self._get_obs(new_state)

        # Check termination
        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]

        terminated = jnp.logical_or(
            jnp.logical_or(
                z >= self.config.breach_height,
                z <= self.config.touchdown_depth
            ),
            vx < self.config.stall_velocity
        )

        truncated = new_step >= self.config.max_steps

        # Compute reward
        reward = self._compute_reward(new_state, action)

        info = {
            'z': z,
            'vx': vx,
            'energy': new_energy,
        }

        return new_state, obs, reward, terminated, info

    def _action_to_ctrl(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert action to MuJoCo control signals."""
        leg_cmd = action[0] * 15.0
        arm_cmd = action[1] * 10.0
        waist_cmd = action[2] * 5.0

        ctrl = jnp.zeros(9)
        ctrl = ctrl.at[0].set(leg_cmd / 150)      # left_knee
        ctrl = ctrl.at[1].set(leg_cmd / 150)      # right_knee
        ctrl = ctrl.at[2].set(-leg_cmd * 0.5 / 200)  # left_hip
        ctrl = ctrl.at[3].set(-leg_cmd * 0.5 / 200)  # right_hip
        ctrl = ctrl.at[4].set(waist_cmd / 250)    # waist
        ctrl = ctrl.at[5].set(arm_cmd / 50)       # left_shoulder
        ctrl = ctrl.at[6].set(arm_cmd / 50)       # right_shoulder
        ctrl = ctrl.at[7].set(0)                  # left_elbow
        ctrl = ctrl.at[8].set(0)                  # right_elbow

        return ctrl

    def _compute_foil_forces(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Compute foil hydrodynamic forces in JAX."""
        cfg = self.config

        # Extract state
        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]
        vz = mjx_data.qvel[2]

        # Pitch from quaternion
        qw = mjx_data.qpos[3]
        qy = mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)

        omega = mjx_data.qvel[4]

        # Flight path angle and angle of attack
        gamma = jnp.arctan2(vz, vx)
        alpha = theta - gamma
        v_sq = vx * vx + vz * vz

        # Lift coefficient (with stall)
        alpha_abs = jnp.abs(alpha)
        is_stalled = alpha_abs >= self.stall_rad

        Cl_linear = self.lift_slope * alpha
        Cl_stalled = self.lift_slope * self.stall_rad * jnp.sign(alpha)
        Cl = jnp.where(is_stalled, Cl_stalled, Cl_linear)

        # Drag coefficient
        Cd_linear = cfg.Cd0 + self.k_induced * Cl * Cl
        Cd_stalled = 0.05 + (alpha_abs - self.stall_rad) * 2.0
        Cd = jnp.where(is_stalled, Cd_stalled, Cd_linear)

        # Dynamic pressure
        q = 0.5 * self.rho * v_sq

        # Lift and drag magnitudes
        L_mag = q * cfg.S * Cl
        D_mag = q * cfg.S * Cd

        # Resolve into x-z components
        Lx = -L_mag * jnp.sin(gamma)
        Lz = L_mag * jnp.cos(gamma)
        Dx = -D_mag * jnp.cos(gamma)
        Dz = -D_mag * jnp.sin(gamma)

        # Stabilizer forces (simplified)
        L_stab_z = -q * cfg.S_stab * self.lift_slope * (theta + self.stab_angle_rad - gamma) * 0.5
        fuselage_len = 0.6
        M_stab = -L_stab_z * fuselage_len

        # NOTE: MuJoCo handles gravity - don't add W here

        # Leg force from knee velocity (reaction force)
        left_knee_vel = mjx_data.qvel[6]  # Approximate index
        right_knee_vel = mjx_data.qvel[7]
        avg_leg_vel = (left_knee_vel + right_knee_vel) / 2
        leg_force = avg_leg_vel * 20  # Reduced scaling

        # Net forces (hydro only - MuJoCo adds gravity)
        Fx = Lx + Dx
        Fz = Lz + Dz - leg_force + L_stab_z

        # Pump thrust
        pump_thrust = jnp.where(
            jnp.abs(vz) > 0.1,
            cfg.pump_thrust_efficiency * jnp.abs(vz) * jnp.abs(L_mag),
            0.0
        )
        Fx = Fx + pump_thrust

        # Pitch moment
        waist_pos = mjx_data.qpos[11]
        rider_moment = waist_pos * 100
        My = M_stab + rider_moment

        return jnp.array([Fx, Fz, My])

    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        """Build observation vector."""
        mjx_data = state.mjx_data

        # Board state
        z = mjx_data.qpos[2]
        vz = mjx_data.qvel[2]
        qw = mjx_data.qpos[3]
        qy = mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)
        omega = mjx_data.qvel[4]
        vx = mjx_data.qvel[0]

        board_obs = jnp.array([z, vz, theta, omega, vx])

        # Joint positions (indices 7-13)
        joint_pos = mjx_data.qpos[7:14]

        # Joint velocities (indices 6-10)
        joint_vel = mjx_data.qvel[6:11]

        # Previous action
        prev_action = state.prev_action

        obs = jnp.concatenate([board_obs, joint_pos, joint_vel, prev_action])
        return obs

    def _compute_reward(self, state: EnvState, action: jnp.ndarray) -> jnp.ndarray:
        """Compute reward."""
        cfg = self.config
        mjx_data = state.mjx_data

        z = mjx_data.qpos[2]
        vx = mjx_data.qvel[0]
        qw = mjx_data.qpos[3]
        qy = mjx_data.qpos[5]
        theta = 2 * jnp.arctan2(qy, qw)

        # Survival bonus
        reward = 0.1

        # Altitude penalty
        alt_error = jnp.abs(z - cfg.altitude_target)
        reward = reward - alt_error * 0.5

        # Velocity penalty
        vel_error = jnp.abs(vx - cfg.velocity_target)
        reward = reward - vel_error * 0.2

        # Pitch stability
        pitch_deg = jnp.abs(jnp.degrees(theta))
        pitch_penalty = jnp.maximum(0, pitch_deg - 15) * 0.1
        reward = reward - pitch_penalty

        # Action smoothness
        action_delta = jnp.sum(jnp.abs(action - state.prev_action))
        reward = reward - action_delta * 0.05

        return reward


def create_batched_env(config: EnvConfig = None, num_envs: int = 256):
    """Create a batched environment for parallel rollouts."""
    env = PumpFoilEnvMJX(config)

    # Vectorize reset and step
    batched_reset = vmap(env.reset)
    batched_step = vmap(env.step)

    return env, batched_reset, batched_step


def test_mjx_env():
    """Test the MJX environment."""
    print("Testing MJX Pump Foil Environment")
    print("=" * 50)

    env = PumpFoilEnvMJX()

    # Test single env
    rng = random.PRNGKey(0)
    state, obs = env.reset(rng)

    print(f"Observation shape: {obs.shape}")
    print(f"Initial obs: z={float(obs[0]):.2f}, vx={float(obs[4]):.2f}")

    # Run a few steps
    total_reward = 0
    for i in range(100):
        action = jnp.array([
            jnp.sin(i * 0.1),
            -jnp.sin(i * 0.1) * 0.5,
            jnp.sin(i * 0.1) * 0.3
        ])
        state, obs, reward, terminated, info = env.step(state, action)
        total_reward += float(reward)

        if terminated:
            print(f"Terminated at step {i}")
            break

    print(f"\nAfter 100 steps:")
    print(f"  z = {float(info['z']):.3f} m")
    print(f"  vx = {float(info['vx']):.2f} m/s")
    print(f"  total reward = {total_reward:.2f}")

    # Test batched environment
    print("\n" + "=" * 50)
    print("Testing batched environment")

    num_envs = 64
    env, batched_reset, batched_step = create_batched_env(num_envs=num_envs)

    rngs = random.split(random.PRNGKey(0), num_envs)
    states, obs_batch = batched_reset(rngs)

    print(f"Batch size: {num_envs}")
    print(f"Batched obs shape: {obs_batch.shape}")

    # Run batched steps
    actions = jnp.zeros((num_envs, 3))
    for i in range(10):
        actions = actions.at[:, 0].set(jnp.sin(i * 0.1))
        states, obs_batch, rewards, terminated, info = batched_step(states, actions)

    print(f"Batched step complete")
    print(f"  Mean z: {float(jnp.mean(info['z'])):.3f}")
    print(f"  Mean vx: {float(jnp.mean(info['vx'])):.2f}")

    # Benchmark
    print("\n" + "=" * 50)
    print("Benchmarking...")

    import time

    # Compile
    _ = batched_step(states, actions)

    # Time 1000 steps
    start = time.time()
    for i in range(1000):
        actions = actions.at[:, 0].set(jnp.sin(i * 0.01))
        states, obs_batch, rewards, terminated, info = batched_step(states, actions)
    jax.block_until_ready(obs_batch)
    elapsed = time.time() - start

    steps_per_sec = (1000 * num_envs) / elapsed
    print(f"  {num_envs} envs × 1000 steps = {1000 * num_envs} total steps")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {steps_per_sec:.0f} steps/sec")

    print("\n[PASS] MJX environment test complete")


if __name__ == "__main__":
    test_mjx_env()

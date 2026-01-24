"""
Modal-based GPU training for MJX Pump Foil environment.

Uses Modal's GPU infrastructure for fast training.
Run with: modal run mujoco/modal_train_mjx.py
"""

import modal

# Create Modal app
app = modal.App("pump-foil-mjx")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mujoco>=3.0.0",
        "mujoco-mjx>=3.0.0",
        "jax[cuda12]>=0.4.20",
        "brax>=0.10.0",
        "numpy>=1.24.0",
    )
)

# Volume for saving checkpoints
volume = modal.Volume.from_name("pump-foil-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # or "T4", "A100" for more power
    timeout=3600,  # 1 hour max
    volumes={"/checkpoints": volume},
)
def train_on_gpu(
    num_timesteps: int = 10_000_000,
    num_envs: int = 2048,
    episode_length: int = 1000,
    seed: int = 0,
):
    """Train the pump foil policy on GPU."""
    import jax
    import jax.numpy as jnp
    from jax import random
    import mujoco
    from mujoco import mjx
    from brax import envs
    from brax.training.agents.ppo import train as ppo_module
    from brax.io import model as brax_model
    import time
    import os

    ppo_train = ppo_module.train

    print(f"JAX devices: {jax.devices()}")
    print(f"Training with {num_envs} parallel environments")
    print(f"Target: {num_timesteps:,} timesteps")
    print("=" * 50)

    # Inline environment definition (can't import from local files in Modal)
    class PumpFoilBraxEnv(envs.Env):
        def __init__(self, **kwargs):
            # Create MJCF model inline
            mjcf = """
            <mujoco model="pump_foil_rider">
              <compiler angle="radian" autolimits="true"/>
              <option timestep="0.01" gravity="0 0 -9.81"/>
              <visual>
                <global offwidth="1920" offheight="1080"/>
              </visual>
              <default>
                <joint damping="5" armature="0.1"/>
                <geom contype="1" conaffinity="1"/>
                <motor ctrllimited="true" ctrlrange="-1 1"/>
              </default>
              <worldbody>
                <body name="board" pos="0 0 0.2">
                  <freejoint name="board_joint"/>
                  <geom name="board_geom" type="box" size="0.7 0.25 0.025" mass="6"/>
                  <body name="left_shin" pos="-0.05 0.15 0.05">
                    <geom type="capsule" fromto="0 0 0  0 0 0.45" size="0.04" mass="5"/>
                    <body name="left_thigh" pos="0 0 0.45">
                      <joint name="left_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                      <geom type="capsule" fromto="0 0 0  0 0 0.45" size="0.05" mass="10"/>
                    </body>
                  </body>
                  <body name="right_shin" pos="-0.05 -0.15 0.05">
                    <geom type="capsule" fromto="0 0 0  0 0 0.45" size="0.04" mass="5"/>
                    <body name="right_thigh" pos="0 0 0.45">
                      <joint name="right_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                      <geom type="capsule" fromto="0 0 0  0 0 0.45" size="0.05" mass="10"/>
                    </body>
                  </body>
                  <body name="pelvis" pos="-0.05 0 0.95">
                    <joint name="left_hip" type="hinge" axis="0 1 0" range="-0.5 1.5"/>
                    <joint name="right_hip" type="hinge" axis="0 1 0" range="-0.5 1.5"/>
                    <geom type="box" size="0.08 0.18 0.08" mass="5"/>
                    <body name="torso" pos="0 0 0.08">
                      <joint name="waist" type="hinge" axis="0 1 0" range="-0.3 0.5"/>
                      <geom type="capsule" fromto="0 0 0  0 0 0.50" size="0.10" mass="30"/>
                      <body name="shoulders" pos="0 0 0.45">
                        <body name="head" pos="0 0 0.17">
                          <geom type="sphere" size="0.10" mass="5"/>
                        </body>
                        <body name="left_arm" pos="0 0.20 0">
                          <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-1.0 2.5"/>
                          <geom type="capsule" fromto="0 0 0  0 0 -0.30" size="0.03" mass="3"/>
                          <body pos="0 0 -0.30">
                            <joint name="left_elbow" type="hinge" axis="0 1 0" range="0 2.5"/>
                            <geom type="capsule" fromto="0 0 0  0 0 -0.25" size="0.025" mass="2"/>
                          </body>
                        </body>
                        <body name="right_arm" pos="0 -0.20 0">
                          <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-1.0 2.5"/>
                          <geom type="capsule" fromto="0 0 0  0 0 -0.30" size="0.03" mass="3"/>
                          <body pos="0 0 -0.30">
                            <joint name="right_elbow" type="hinge" axis="0 1 0" range="0 2.5"/>
                            <geom type="capsule" fromto="0 0 0  0 0 -0.25" size="0.025" mass="2"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </worldbody>
              <actuator>
                <motor name="left_knee_motor" joint="left_knee" gear="150"/>
                <motor name="right_knee_motor" joint="right_knee" gear="150"/>
                <motor name="left_hip_motor" joint="left_hip" gear="200"/>
                <motor name="right_hip_motor" joint="right_hip" gear="200"/>
                <motor name="waist_motor" joint="waist" gear="250"/>
                <motor name="left_shoulder_motor" joint="left_shoulder" gear="50"/>
                <motor name="right_shoulder_motor" joint="right_shoulder" gear="50"/>
                <motor name="left_elbow_motor" joint="left_elbow" gear="30"/>
                <motor name="right_elbow_motor" joint="right_elbow" gear="30"/>
              </actuator>
            </mujoco>
            """

            self.mj_model = mujoco.MjModel.from_xml_string(mjcf)
            self.sys = mjx.put_model(self.mj_model)
            self.board_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "board")

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

            self.initial_height = 0.05
            self.initial_velocity = 4.5
            self.breach_height = 0.4
            self.touchdown_depth = -0.3
            self.stall_velocity = 1.5

        @property
        def observation_size(self):
            return 17

        @property
        def action_size(self):
            return 3

        @property
        def backend(self):
            return 'mjx'

        def reset(self, rng):
            mjx_data = mjx.make_data(self.sys)
            qpos = mjx_data.qpos.at[2].set(self.initial_height)
            qvel = mjx_data.qvel.at[0].set(self.initial_velocity)
            pitch = 0.08
            qpos = qpos.at[3].set(jnp.cos(pitch / 2))
            qpos = qpos.at[5].set(jnp.sin(pitch / 2))
            qpos = qpos.at[7].set(0.2)
            qpos = qpos.at[8].set(0.2)
            qpos = qpos.at[9].set(0.1)
            qpos = qpos.at[10].set(0.1)
            qpos = qpos.at[11].set(0.1)
            mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
            mjx_data = mjx.forward(self.sys, mjx_data)
            obs = self._get_obs(mjx_data)
            return envs.State(
                pipeline_state=mjx_data,
                obs=obs,
                reward=jnp.array(0.0),
                done=jnp.array(0.0),
                metrics={},
                info={'rng': rng},
            )

        def step(self, state, action):
            mjx_data = state.pipeline_state
            action = jnp.clip(action, -1.0, 1.0)

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
            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Foil forces
            z = mjx_data.qpos[2]
            vx = mjx_data.qvel[0]
            vz = mjx_data.qvel[2]
            qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
            theta = 2 * jnp.arctan2(qy, qw)
            gamma = jnp.arctan2(vz, vx)
            alpha = theta - gamma
            v_sq = vx * vx + vz * vz

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
            L_stab_z = -q * self.S_stab * self.lift_slope * (theta + self.stab_angle_rad - gamma) * 0.5
            M_stab = -L_stab_z * 0.6
            W = -self.total_mass * self.g
            leg_vel = (mjx_data.qvel[6] + mjx_data.qvel[7]) / 2
            leg_force = leg_vel * 50

            Fx = Dx + jnp.where(jnp.abs(vz) > 0.1,
                               self.pump_efficiency * jnp.abs(vz) * jnp.abs(L), 0.0)
            Fz = Lz + W - leg_force + L_stab_z
            waist_pos = mjx_data.qpos[11]
            My = M_stab + waist_pos * 100

            xfrc = jnp.zeros((self.mj_model.nbody, 6))
            xfrc = xfrc.at[self.board_id, 0].set(Fx)
            xfrc = xfrc.at[self.board_id, 2].set(Fz)
            xfrc = xfrc.at[self.board_id, 4].set(My)
            mjx_data = mjx_data.replace(xfrc_applied=xfrc)
            mjx_data = mjx.step(self.sys, mjx_data)

            z = mjx_data.qpos[2]
            vx = mjx_data.qvel[0]
            done = jnp.where(
                jnp.logical_or(
                    jnp.logical_or(z >= self.breach_height, z <= self.touchdown_depth),
                    vx < self.stall_velocity
                ), 1.0, 0.0)

            qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
            theta = 2 * jnp.arctan2(qy, qw)
            reward = 0.1
            reward -= jnp.abs(z) * 0.5
            reward -= jnp.abs(vx - 4.5) * 0.2
            reward -= jnp.maximum(0, jnp.abs(jnp.degrees(theta)) - 15) * 0.1

            obs = self._get_obs(mjx_data)
            return state.replace(pipeline_state=mjx_data, obs=obs, reward=reward, done=done)

        def _get_obs(self, mjx_data):
            z = mjx_data.qpos[2]
            vz = mjx_data.qvel[2]
            qw, qy = mjx_data.qpos[3], mjx_data.qpos[5]
            theta = 2 * jnp.arctan2(qy, qw)
            omega = mjx_data.qvel[4]
            vx = mjx_data.qvel[0]
            return jnp.concatenate([
                jnp.array([z, vz, theta, omega, vx]),
                mjx_data.qpos[7:14],
                mjx_data.qvel[6:11],
            ])

    # Create environment and train
    env = PumpFoilBraxEnv()

    start_time = time.time()
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_envs=num_envs,
        learning_rate=3e-4,
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

    print(f"\nTraining complete!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Steps/sec: {num_timesteps / elapsed:.0f}")

    # Save model
    os.makedirs("/checkpoints/mjx", exist_ok=True)
    model_path = f"/checkpoints/mjx/model_{num_timesteps}.pkl"
    brax_model.save_params(model_path, params)
    print(f"  Saved: {model_path}")

    volume.commit()

    return {
        "elapsed": elapsed,
        "steps_per_sec": num_timesteps / elapsed,
        "model_path": model_path,
    }


@app.local_entrypoint()
def main(
    timesteps: int = 10_000_000,
    num_envs: int = 2048,
):
    """Entry point for modal run."""
    print(f"Starting GPU training: {timesteps:,} steps, {num_envs} envs")
    result = train_on_gpu.remote(
        num_timesteps=timesteps,
        num_envs=num_envs,
    )
    print(f"\nResults: {result}")

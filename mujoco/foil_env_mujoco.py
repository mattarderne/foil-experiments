"""
MuJoCo-based Pump Foil Environment

Hybrid physics approach:
1. MuJoCo handles rider body dynamics (joints, masses, collisions)
2. Custom physics computes foil hydrodynamics (lift, drag, thrust)
3. Foil forces applied to board via xfrc_applied

This replaces pump_foil_env_curriculum.py with MuJoCo-based simulation.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# Import the custom foil physics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from foil_env.foil_physics import FoilPhysics, FoilState, CONFIG


@dataclass
class MuJoCoFoilConfig:
    """Configuration for the MuJoCo foil environment."""
    # Foil physics (sized for 86kg MuJoCo model)
    S: float = 0.22              # Wing area (m²) - larger for heavier model
    S_stab: float = 0.044        # Stabilizer area (m²)
    AR: float = 8.0              # Aspect ratio
    stab_angle: float = -1.0     # Stabilizer angle (deg)
    Cd0: float = 0.006           # Parasitic drag (reduced for modern foils)
    Cd_mast: float = 0.04        # Mast drag (reduced for carbon)
    pump_thrust_efficiency: float = 0.30  # Pump thrust conversion (tuned for sustained flight)

    # Environment
    max_episode_steps: int = 6000  # 60s at 100Hz
    initial_velocity: float = 4.5  # m/s
    initial_height: float = 0.0    # m (at water level)
    initial_pitch: float = 0.0635  # ~3.64° (lift-equilibrium pitch)

    # Termination conditions
    breach_height: float = 0.3     # Foil breach threshold
    touchdown_depth: float = -0.5  # Board touchdown threshold
    stall_velocity: float = 1.5    # Minimum velocity

    # Pitch control gains (strong PD for stability)
    pitch_Kp: float = 500.0        # Position gain
    pitch_Kd: float = 200.0        # Velocity gain

    # Rewards
    altitude_target: float = 0.0   # Target altitude
    velocity_target: float = 4.5   # Target velocity


class PumpFoilEnvMuJoCo(gym.Env):
    """
    Pump Foil environment using MuJoCo for body dynamics.

    Observation space (20D):
    - Board state: z, vz, pitch, pitch_rate (4)
    - Velocity: vx (1)
    - Joint positions: knee*2, hip*2, waist, shoulder*2 (7)
    - Joint velocities: knee*2, hip*2, waist (5)
    - Previous action (3)

    Action space (5D):
    - Leg extension (symmetric knee+hip) (1)
    - Waist lean (1)
    - Arm swing (symmetric shoulders) (1)
    - Arm frequency factor (1)
    - Energy allocation (1)

    Actually simplified to (3D) for now to match original env:
    - Leg extension velocity command (1)
    - Arm swing velocity command (1)
    - Waist lean velocity command (1)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 100}

    def __init__(
        self,
        config: Optional[MuJoCoFoilConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config or MuJoCoFoilConfig()
        self.render_mode = render_mode

        # Load MuJoCo model
        model_path = Path(__file__).parent / "rider.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Get body/joint IDs
        self._cache_ids()

        # Initialize foil physics
        foil_config = {
            **CONFIG,
            'S': self.config.S,
            'S_stab': self.config.S_stab,
            'AR': self.config.AR,
            'stab_angle': self.config.stab_angle,
            'Cd0': self.config.Cd0,
            'Cd_mast': self.config.Cd_mast,
            'pump_thrust_efficiency': self.config.pump_thrust_efficiency,
        }
        self.foil_physics = FoilPhysics(foil_config)

        # Action space: leg, arm, waist velocity commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # Rendering
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        # Episode state
        self.steps = 0
        self.prev_action = np.zeros(3)
        self.total_energy = 0.0

    def _cache_ids(self):
        """Cache body and joint IDs for fast lookup."""
        self.board_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "board"
        )

        # Joint indices in qpos/qvel
        # Freejoint: 7 qpos (3 pos + 4 quat), 6 qvel (3 vel + 3 angvel)
        # Then hinge joints add 1 each
        self.joint_ids = {}
        for name in ["left_knee", "right_knee", "left_hip", "right_hip",
                     "waist", "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow"]:
            jnt_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            qpos_adr = self.mj_model.jnt_qposadr[jnt_id]
            qvel_adr = self.mj_model.jnt_dofadr[jnt_id]
            self.joint_ids[name] = (jnt_id, qpos_adr, qvel_adr)

        # Actuator indices
        self.actuator_ids = {}
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            self.actuator_ids[name] = i

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Set initial board position and velocity
        # qpos[0:3] = position, qpos[3:7] = quaternion
        self.mj_data.qpos[2] = self.config.initial_height  # z position

        # Set initial forward velocity
        self.mj_data.qvel[0] = self.config.initial_velocity  # vx

        # Set initial pitch (equilibrium angle)
        # Convert pitch angle to quaternion (rotation about Y axis)
        pitch = self.config.initial_pitch  # ~3.64° for lift equilibrium
        self.mj_data.qpos[3] = np.cos(pitch / 2)  # w
        self.mj_data.qpos[5] = np.sin(pitch / 2)  # y component

        # Reset joints to neutral standing pose
        self._set_joint("left_knee", 0.2)
        self._set_joint("right_knee", 0.2)
        self._set_joint("left_hip", 0.1)
        self._set_joint("right_hip", 0.1)
        self._set_joint("waist", 0.1)
        self._set_joint("left_shoulder", 0.0)
        self._set_joint("right_shoulder", 0.0)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.steps = 0
        self.prev_action = np.zeros(3)
        self.total_energy = 0.0

        return self._get_obs(), {}

    def _set_joint(self, name: str, value: float):
        """Set joint position."""
        _, qpos_adr, _ = self.joint_ids[name]
        self.mj_data.qpos[qpos_adr] = value

    def _get_joint(self, name: str) -> Tuple[float, float]:
        """Get joint position and velocity."""
        _, qpos_adr, qvel_adr = self.joint_ids[name]
        return self.mj_data.qpos[qpos_adr], self.mj_data.qvel[qvel_adr]

    def step(self, action):
        """Execute one environment step."""
        action = np.clip(action, -1.0, 1.0)

        # Apply control via actuators
        # Action: [leg_vel, arm_vel, waist_vel]
        leg_cmd = action[0] * 15.0    # Scale to rad/s
        arm_cmd = action[1] * 10.0
        waist_cmd = action[2] * 5.0

        # Symmetric leg control (both knees and hips move together)
        self.mj_data.ctrl[self.actuator_ids["left_knee_motor"]] = leg_cmd / 150
        self.mj_data.ctrl[self.actuator_ids["right_knee_motor"]] = leg_cmd / 150
        self.mj_data.ctrl[self.actuator_ids["left_hip_motor"]] = -leg_cmd * 0.5 / 200
        self.mj_data.ctrl[self.actuator_ids["right_hip_motor"]] = -leg_cmd * 0.5 / 200

        # Waist control
        self.mj_data.ctrl[self.actuator_ids["waist_motor"]] = waist_cmd / 250

        # Arm control (symmetric)
        self.mj_data.ctrl[self.actuator_ids["left_shoulder_motor"]] = arm_cmd / 50
        self.mj_data.ctrl[self.actuator_ids["right_shoulder_motor"]] = arm_cmd / 50

        # Compute foil forces before MuJoCo step
        foil_state = self._get_foil_state()
        foil_forces = self._compute_foil_forces(foil_state)

        # Apply foil forces to board
        # xfrc_applied is [fx, fy, fz, tx, ty, tz] per body
        self.mj_data.xfrc_applied[self.board_id] = [
            foil_forces['Fx'], 0, foil_forces['Fz'],
            0, foil_forces['My'], 0
        ]

        # Step MuJoCo simulation
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.steps += 1

        # Track energy usage
        self.total_energy += np.sum(np.abs(action)) * 10  # Simplified energy model

        # Get observation
        obs = self._get_obs()

        # Check termination
        terminated, term_reason = self._check_termination(foil_state)

        # Compute reward
        reward = self._compute_reward(foil_state, action)

        # Truncated if max steps reached
        truncated = self.steps >= self.config.max_episode_steps

        # Info
        info = {
            'z': foil_state.z,
            'vx': foil_state.vx,
            'vz': foil_state.vz,
            'pitch': np.degrees(foil_state.theta),
            'energy': self.total_energy,
            'termination': term_reason if terminated else None,
        }

        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def _get_foil_state(self) -> FoilState:
        """Extract foil state from MuJoCo data."""
        # Board position and velocity
        x = self.mj_data.qpos[0]
        z = self.mj_data.qpos[2]
        vx = self.mj_data.qvel[0]
        vz = self.mj_data.qvel[2]

        # Convert quaternion to pitch angle (rotation about Y axis)
        qw = self.mj_data.qpos[3]
        qy = self.mj_data.qpos[5]
        theta = 2 * np.arctan2(qy, qw)

        # Angular velocity about Y axis
        omega = self.mj_data.qvel[4]  # Angular velocity Y

        return FoilState(x=x, z=z, vx=vx, vz=vz, theta=theta, omega=omega)

    def _compute_foil_forces(self, state: FoilState) -> Dict[str, float]:
        """Compute hydrodynamic forces using custom physics.

        NOTE: MuJoCo already applies gravity, so we only add hydrodynamic forces.

        Uses pure PD pitch control (not M_stab) for stability with articulated rider.
        """
        forces = self.foil_physics.calculate_forces(state)

        # Compute leg force from rider motion (vertical reaction)
        # This comes from the center of mass acceleration
        # Simplified: use knee velocity as proxy
        left_knee_vel = self._get_joint("left_knee")[1]
        right_knee_vel = self._get_joint("right_knee")[1]
        avg_leg_vel = (left_knee_vel + right_knee_vel) / 2

        # Leg extension pushes board down (reaction force)
        leg_force = avg_leg_vel * 20  # Reduced scaling

        # Net forces (MuJoCo handles gravity, we add hydro forces only)
        Fx = forces.F_hydro_x
        Fz = forces.F_hydro_z - leg_force  # No W - MuJoCo adds gravity

        # Add pump thrust from vertical motion
        # Only apply when altitude is near target (actual pumping, not just rising/falling)
        pump_efficiency = self.config.pump_thrust_efficiency
        altitude_in_range = abs(state.z - self.config.altitude_target) < 0.15
        if abs(state.vz) > 0.1 and altitude_in_range:
            # Cap pump thrust to prevent runaway acceleration
            pump_thrust = pump_efficiency * abs(state.vz) * abs(forces.L)
            pump_thrust = min(pump_thrust, 50.0)  # Cap at 50N
            Fx += pump_thrust

        # Pure PD pitch control (don't use M_stab - it causes overcorrection)
        # Target pitch is the equilibrium angle where lift = weight
        target_pitch = self.config.initial_pitch
        pitch_error = state.theta - target_pitch

        # Strong PD control to overcome rider body dynamics
        Kp = self.config.pitch_Kp
        Kd = self.config.pitch_Kd
        My = -Kp * pitch_error - Kd * state.omega

        return {'Fx': Fx, 'Fz': Fz, 'My': My}

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        state = self._get_foil_state()

        # Board state
        board_obs = [state.z, state.vz, state.theta, state.omega, state.vx]

        # Joint positions
        joint_pos = [
            self._get_joint("left_knee")[0],
            self._get_joint("right_knee")[0],
            self._get_joint("left_hip")[0],
            self._get_joint("right_hip")[0],
            self._get_joint("waist")[0],
            self._get_joint("left_shoulder")[0],
            self._get_joint("right_shoulder")[0],
        ]

        # Joint velocities
        joint_vel = [
            self._get_joint("left_knee")[1],
            self._get_joint("right_knee")[1],
            self._get_joint("left_hip")[1],
            self._get_joint("right_hip")[1],
            self._get_joint("waist")[1],
        ]

        # Previous action
        prev_action = self.prev_action.tolist()

        obs = board_obs + joint_pos + joint_vel + prev_action
        return np.array(obs, dtype=np.float32)

    def _check_termination(self, state: FoilState) -> Tuple[bool, Optional[str]]:
        """Check if episode should terminate."""
        # Foil breach
        if state.z >= self.config.breach_height:
            return True, "foil_breached"

        # Board touchdown
        if state.z <= self.config.touchdown_depth:
            return True, "board_touchdown"

        # Stalled velocity
        if state.vx < self.config.stall_velocity:
            return True, "stalled_velocity"

        return False, None

    def _compute_reward(self, state: FoilState, action: np.ndarray) -> float:
        """Compute reward for current state."""
        reward = 0.0

        # Survival bonus
        reward += 0.1

        # Altitude reward (penalize deviation from target)
        alt_error = abs(state.z - self.config.altitude_target)
        reward -= alt_error * 0.5

        # Velocity reward
        vel_error = abs(state.vx - self.config.velocity_target)
        reward -= vel_error * 0.2

        # Pitch stability
        pitch_deg = abs(np.degrees(state.theta))
        if pitch_deg > 15:
            reward -= (pitch_deg - 15) * 0.1

        # Action smoothness (penalize rapid changes)
        action_delta = np.sum(np.abs(action - self.prev_action))
        reward -= action_delta * 0.05

        return reward

    def render(self):
        """Render the environment."""
        if self.renderer is None:
            if self.render_mode == "rgb_array":
                self.renderer = mujoco.Renderer(
                    self.mj_model, height=480, width=640
                )
            else:
                return None

        self.renderer.update_scene(self.mj_data)
        return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def test_env():
    """Test the MuJoCo environment."""
    print("Testing MuJoCo Pump Foil Environment")
    print("=" * 50)

    env = PumpFoilEnvMuJoCo(render_mode="rgb_array")
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: z={obs[0]:.2f}, vx={obs[4]:.2f}")

    # Run for 100 steps
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample() * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Terminated at step {i}: {info['termination']}")
            break

    print(f"\nAfter 100 steps:")
    print(f"  z = {info['z']:.3f} m")
    print(f"  vx = {info['vx']:.2f} m/s")
    print(f"  pitch = {info['pitch']:.1f}°")
    print(f"  total reward = {total_reward:.2f}")

    # Render a frame
    frame = env.render()
    print(f"\nRender shape: {frame.shape}")

    # Save frame
    try:
        from PIL import Image
        img = Image.fromarray(frame)
        img.save("mujoco_env_test.png")
        print("Saved: mujoco_env_test.png")
    except ImportError:
        pass

    env.close()
    print("\n[PASS] Environment test complete")


if __name__ == "__main__":
    test_env()

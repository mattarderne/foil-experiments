#!/usr/bin/env python3
"""
Hybrid Pump Foil Environment

MuJoCo simulates the rider body joints.
Python foil physics controls the board trajectory.
Board position is set directly (no force application).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import sys
sys.path.insert(0, '/Users/mattbook-air/claude-stuff/motion/foilpump')
from foil_env.foil_physics import FoilPhysics, FoilState, CONFIG


@dataclass
class HybridConfig:
    """Configuration for hybrid environment."""
    # Foil parameters - tuned for equilibrium
    foil_area: float = 0.20          # m² (2000 cm²)
    stab_area: float = 0.040         # m²
    stab_angle: float = -4.0         # degrees

    # Physics - tuned for sustained flight
    pump_efficiency: float = 0.26    # Higher pump-to-thrust
    drag_coef: float = 0.005         # Low parasitic drag
    mast_drag: float = 0.035         # Low mast drag

    # Velocity limits
    vx_max: float = 5.2              # Max velocity (balanced)

    # Body force amplification (small body motion → meaningful force)
    force_amplification: float = 60.0  # rel_vel 0.1 m/s → ~400N

    # Episode
    max_time: float = 60.0           # seconds
    dt: float = 0.01                 # timestep

    # Initial conditions - low start with room to rise
    init_vx: float = 4.5             # m/s
    init_z: float = -0.15            # m (lower, more margin from breach)
    init_pitch: float = 0.085        # rad (~4.9°)

    # Termination
    z_min: float = -0.5              # touchdown
    z_max: float = 0.30              # breach (higher margin)
    vx_min: float = 1.5              # stall

    # Reward weights
    alive_bonus: float = 1.0
    velocity_weight: float = 0.5
    altitude_penalty: float = 1.0
    energy_penalty: float = 0.001


class HybridPumpFoilEnv(gym.Env):
    """
    Hybrid environment: MuJoCo body + Python foil physics.

    Observation (17):
        - Board: z, vx, vz, pitch, omega (5)
        - Joints: hip_L, hip_R, knee_L, knee_R, waist positions (5)
        - Joints: hip_L, hip_R, knee_L, knee_R, waist velocities (5)
        - Time remaining (normalized) (1)
        - Altitude error (1)

    Action (5):
        - Target positions for: hip_L, hip_R, knee_L, knee_R, waist
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[HybridConfig] = None, render_mode: str = None):
        super().__init__()

        self.config = config or HybridConfig()
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path('rider_hybrid.xml')
        self.data = mujoco.MjData(self.model)

        # Get IDs
        self.board_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'board')
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        # Rider mass (everything except board)
        self.board_mass = self.model.body_mass[self.board_id]
        self.total_mass = sum(self.model.body_mass)
        self.rider_mass = self.total_mass - self.board_mass

        # Sensor indices
        self._setup_sensors()

        # Initialize foil physics
        self.foil = FoilPhysics({
            **CONFIG,
            'S': self.config.foil_area,
            'S_stab': self.config.stab_area,
            'stab_angle': self.config.stab_angle,
            'Cd0': self.config.drag_coef,
            'Cd_mast': self.config.mast_drag,
            'pump_thrust_efficiency': self.config.pump_efficiency,
            'mass': self.rider_mass,
            'board_mass': self.board_mass,
        })

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # Action: joint position targets (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Joint limits for action scaling (surf stance)
        self.joint_ranges = {
            'left_hip': (-0.5, 1.2),    # front leg
            'right_hip': (-0.5, 1.2),   # back leg
            'left_knee': (-2.4, -0.1),  # front knee
            'right_knee': (-2.4, -0.1), # back knee
            'waist': (-0.4, 0.4),
        }

        # State tracking
        self.prev_rider_com = None
        self.foil_state = None
        self.time = 0.0
        self.energy_used = 0.0

        # Renderer
        self.renderer = None

    def _setup_sensors(self):
        """Get sensor data indices."""
        # Helper to get sensor address
        def get_sensor_adr(name):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return self.model.sensor_adr[sid]

        self.sensor_adr = {
            'board_pos': get_sensor_adr('board_pos'),
            'board_quat': get_sensor_adr('board_quat'),
            'board_vel': get_sensor_adr('board_vel'),
            'board_angvel': get_sensor_adr('board_angvel'),
            'rider_com': get_sensor_adr('rider_com'),
            'rider_vel': get_sensor_adr('rider_vel'),
            'left_hip_pos': get_sensor_adr('left_hip_pos'),
            'right_hip_pos': get_sensor_adr('right_hip_pos'),
            'left_knee_pos': get_sensor_adr('left_knee_pos'),
            'right_knee_pos': get_sensor_adr('right_knee_pos'),
            'waist_pos': get_sensor_adr('waist_pos'),
            'left_hip_vel': get_sensor_adr('left_hip_vel'),
            'right_hip_vel': get_sensor_adr('right_hip_vel'),
            'left_knee_vel': get_sensor_adr('left_knee_vel'),
            'right_knee_vel': get_sensor_adr('right_knee_vel'),
            'waist_vel': get_sensor_adr('waist_vel'),
        }

    def _get_sensor(self, name: str, size: int = 1) -> np.ndarray:
        """Read sensor data."""
        adr = self.sensor_adr[name]
        return self.data.sensordata[adr:adr+size].copy()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Set initial board state
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = self.config.init_z  # z

        # Quaternion for pitch
        pitch = self.config.init_pitch
        self.data.qpos[3] = np.cos(pitch / 2)  # qw
        self.data.qpos[4] = 0.0  # qx
        self.data.qpos[5] = np.sin(pitch / 2)  # qy
        self.data.qpos[6] = 0.0  # qz

        # Initial velocities
        self.data.qvel[0] = self.config.init_vx  # vx
        self.data.qvel[1] = 0.0  # vy
        self.data.qvel[2] = 0.0  # vz

        # Set joints to standing position
        self._set_standing_pose()

        # Forward to compute positions
        mujoco.mj_forward(self.model, self.data)

        # Initialize foil state
        self.foil_state = FoilState(
            x=0.0,
            z=self.config.init_z,
            vx=self.config.init_vx,
            vz=0.0,
            theta=self.config.init_pitch,
            omega=0.0,
        )

        # Initialize CoM tracking
        self.prev_rider_com = self._get_sensor('rider_com', 3).copy()
        rider_com = self._get_sensor('rider_com', 3)
        board_pos = self._get_sensor('board_pos', 3)
        self.prev_rider_rel_z = rider_com[2] - board_pos[2]

        # Reset counters
        self.time = 0.0
        self.energy_used = 0.0

        return self._get_obs(), {}

    def _set_standing_pose(self):
        """Set joints to neutral surf stance position."""
        # Joint qpos addresses (after 7 DOF freejoint)
        joint_qpos = {
            'left_hip': 7,
            'left_knee': 8,
            'right_hip': 9,
            'right_knee': 10,
            'waist': 11,
        }

        # Surf stance: knees moderately bent, hips neutral
        self.data.qpos[joint_qpos['left_hip']] = 0.1    # front leg slightly forward
        self.data.qpos[joint_qpos['right_hip']] = 0.1   # back leg slightly forward
        self.data.qpos[joint_qpos['left_knee']] = -0.8  # front knee bent
        self.data.qpos[joint_qpos['right_knee']] = -0.8 # back knee bent
        self.data.qpos[joint_qpos['waist']] = 0.0

        # Set actuator targets to match
        self.data.ctrl[0] = 0.1    # left_hip
        self.data.ctrl[1] = 0.1    # right_hip
        self.data.ctrl[2] = -0.8   # left_knee
        self.data.ctrl[3] = -0.8   # right_knee
        self.data.ctrl[4] = 0.0    # waist

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1,1] to joint ranges."""
        ranges = [
            self.joint_ranges['left_hip'],
            self.joint_ranges['right_hip'],
            self.joint_ranges['left_knee'],
            self.joint_ranges['right_knee'],
            self.joint_ranges['waist'],
        ]

        scaled = np.zeros(5)
        for i, (low, high) in enumerate(ranges):
            # Map [-1, 1] to [low, high]
            scaled[i] = low + (action[i] + 1) * 0.5 * (high - low)

        return scaled

    def step(self, action: np.ndarray):
        # Scale and apply action
        joint_targets = self._scale_action(np.clip(action, -1, 1))
        self.data.ctrl[:] = joint_targets

        # Track energy (sum of |torque * velocity|)
        joint_vels = np.array([
            self._get_sensor('left_hip_vel')[0],
            self._get_sensor('right_hip_vel')[0],
            self._get_sensor('left_knee_vel')[0],
            self._get_sensor('right_knee_vel')[0],
            self._get_sensor('waist_vel')[0],
        ])
        step_energy = np.sum(np.abs(self.data.actuator_force * joint_vels)) * self.config.dt
        self.energy_used += step_energy
        self.step_energy = step_energy  # For per-step reward calculation

        # Step MuJoCo (body simulation only)
        mujoco.mj_step(self.model, self.data)

        # Get rider CoM in world frame
        rider_com = self._get_sensor('rider_com', 3)

        # Get board position in world frame
        board_pos = self._get_sensor('board_pos', 3)

        # Compute rider position RELATIVE TO BOARD (this is what matters)
        rider_rel_z = rider_com[2] - board_pos[2]

        # Compute relative velocity by differentiating relative position
        dt = self.config.dt
        if not hasattr(self, 'prev_rider_rel_z'):
            self.prev_rider_rel_z = rider_rel_z

        relative_vel_z = (rider_rel_z - self.prev_rider_rel_z) / dt
        relative_vel_z = np.clip(relative_vel_z, -2.0, 2.0)
        self.prev_rider_rel_z = rider_rel_z

        # Body-driven leg force with amplification
        # The body motion provides the timing/phase, we amplify to meaningful force
        # This lets the RL agent learn pump timing even if body can't generate full force
        # With amplification=60, rel_vel of 0.1 m/s → ~400N force
        force_amplification = self.config.force_amplification
        raw_force = self.rider_mass * relative_vel_z * force_amplification
        leg_force = np.clip(raw_force, -400, 400)

        # Pitch torque from waist lean
        waist_angle = self._get_sensor('waist_pos')[0]
        pitch_torque = -30 * waist_angle
        new_foil_state = self.foil.step(
            self.foil_state,
            leg_force=leg_force,
            pitch_torque=pitch_torque,
            dt=dt
        )

        # Cap velocity to prevent runaway
        capped_vx = np.clip(new_foil_state.vx, self.config.vx_min, self.config.vx_max)
        if capped_vx != new_foil_state.vx:
            new_foil_state = FoilState(
                x=new_foil_state.x, z=new_foil_state.z,
                vx=capped_vx, vz=new_foil_state.vz,
                theta=new_foil_state.theta, omega=new_foil_state.omega
            )

        # Update foil state
        self.foil_state = new_foil_state

        # SET board position directly from foil physics (the key insight!)
        self.data.qpos[0] = new_foil_state.x
        self.data.qpos[2] = new_foil_state.z

        # Set pitch via quaternion
        pitch = new_foil_state.theta
        self.data.qpos[3] = np.cos(pitch / 2)
        self.data.qpos[5] = np.sin(pitch / 2)

        # Set velocities
        self.data.qvel[0] = new_foil_state.vx
        self.data.qvel[2] = new_foil_state.vz
        self.data.qvel[4] = new_foil_state.omega  # pitch rate

        # Update CoM history
        self.prev_rider_com = rider_com.copy()

        # Update time
        self.time += dt

        # Check termination
        terminated = False
        truncated = False
        termination_reason = None

        z = new_foil_state.z
        vx = new_foil_state.vx

        if z < self.config.z_min:
            terminated = True
            termination_reason = "touchdown"
        elif z > self.config.z_max:
            terminated = True
            termination_reason = "breach"
        elif vx < self.config.vx_min:
            terminated = True
            termination_reason = "stall"
        elif self.time >= self.config.max_time:
            truncated = True
            termination_reason = "timeout"

        # Compute reward
        reward = self._compute_reward(terminated, truncated)

        # Get observation
        obs = self._get_obs()

        info = {
            'time': self.time,
            'z': z,
            'vx': vx,
            'energy': self.energy_used,
            'termination': termination_reason,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, terminated: bool, truncated: bool) -> float:
        """Compute reward for current step."""
        cfg = self.config
        reward = 0.0

        # Alive bonus
        if not terminated:
            reward += cfg.alive_bonus

        # Velocity reward (maintain target speed)
        target_vx = cfg.init_vx
        vx_error = abs(self.foil_state.vx - target_vx)
        reward += cfg.velocity_weight * np.exp(-vx_error)

        # Altitude penalty (prefer middle of range)
        z = self.foil_state.z
        z_target = (cfg.z_min + cfg.z_max) / 2
        z_error = abs(z - z_target)
        reward -= cfg.altitude_penalty * z_error

        # Energy penalty (per-step, not cumulative)
        step_energy = getattr(self, 'step_energy', 0.0)
        reward -= cfg.energy_penalty * step_energy

        # Terminal penalties
        if terminated:
            reward -= 10.0

        return reward

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        cfg = self.config

        # Board state
        z = self.foil_state.z
        vx = self.foil_state.vx
        vz = self.foil_state.vz
        pitch = self.foil_state.theta
        omega = self.foil_state.omega

        # Joint positions
        joint_pos = np.array([
            self._get_sensor('left_hip_pos')[0],
            self._get_sensor('right_hip_pos')[0],
            self._get_sensor('left_knee_pos')[0],
            self._get_sensor('right_knee_pos')[0],
            self._get_sensor('waist_pos')[0],
        ])

        # Joint velocities
        joint_vel = np.array([
            self._get_sensor('left_hip_vel')[0],
            self._get_sensor('right_hip_vel')[0],
            self._get_sensor('left_knee_vel')[0],
            self._get_sensor('right_knee_vel')[0],
            self._get_sensor('waist_vel')[0],
        ])

        # Time remaining (normalized)
        time_remaining = 1.0 - (self.time / cfg.max_time)

        # Altitude error from target
        z_target = (cfg.z_min + cfg.z_max) / 2
        alt_error = z - z_target

        obs = np.concatenate([
            [z, vx, vz, pitch, omega],  # 5
            joint_pos,                   # 5
            joint_vel,                   # 5
            [time_remaining],            # 1
            [alt_error],                 # 1
        ]).astype(np.float32)

        return obs

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.renderer.update_scene(self.data, camera="side")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Quick test
if __name__ == "__main__":
    import imageio

    env = HybridPumpFoilEnv(render_mode="rgb_array")
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Rider mass: {env.rider_mass:.1f} kg")
    print(f"Total mass: {env.total_mass:.1f} kg")

    frames = []
    total_reward = 0

    print("\nRunning episode with sinusoidal leg pumping...")

    for step in range(6000):  # 60 seconds
        t = step * 0.01

        # Simple sinusoidal pump action
        pump_phase = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz

        # Altitude-responsive pumping (skilled rider behavior)
        z = env.foil_state.z
        target_z = -0.10  # Target altitude (middle of range)
        alt_error = z - target_z
        # When too low, pump harder; when too high, pump gentler
        pump_gain = np.clip(0.6 - 1.0 * alt_error, 0.4, 0.9)

        # Squat pump action - both knees bend together (surf stance)
        # Knee action: negative = bent, positive = straight
        # When knees straighten (action +), legs extend, pushing board down
        # When knees bend (action -), legs compress, body drops relative to board
        #
        # For pumping, we want:
        # - Quick extension (push down) to accelerate board down
        # - Slower flexion (absorb) as board rises
        knee_action = 0.7 * pump_phase       # Both knees in sync: -0.7 to +0.7

        # Hips stay relatively neutral - slight counter to keep torso upright
        hip_action = -0.1 * pump_phase       # Small counter-rotation

        # Waist helps shift weight fore/aft during pump
        waist_action = 0.1 * pump_phase

        action = np.array([
            hip_action,                    # front hip (left)
            hip_action,                    # back hip (right)
            knee_action,                   # front knee (left)
            knee_action,                   # back knee (right)
            waist_action,                  # waist
        ])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if step % 500 == 0:  # Every 5s
            vz = env.foil_state.vz
            print(f"t={info['time']:.0f}s: z={info['z']:+.2f}m, vx={info['vx']:.2f}m/s")

        if terminated or truncated:
            print(f"Episode ended: {info['termination']} at t={info['time']:.1f}s")
            break

    if frames:
        imageio.mimsave('hybrid_test.mp4', frames, fps=20)
        print(f"\nSaved hybrid_test.mp4 ({len(frames)} frames)")

    print(f"Total reward: {total_reward:.1f}")
    env.close()

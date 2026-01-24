"""
Pump Foil Gymnasium Environment (v2 - Realistic Pumping Physics)

A reinforcement learning environment where an agent learns to control
a pump foil rider to maintain flight and generate forward thrust.

Key physics constraints (v2):
- Leg force must oscillate (zero mean over time window) - can't apply constant force
- Metabolic energy cost includes isometric (holding) component
- Finite leg range of motion enforces pumping pattern
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from foil_env.foil_physics import FoilPhysics, FoilState, ForceOutput, CONFIG


class PumpFoilEnv(gym.Env):
    """
    Gymnasium environment for pump foiling simulation.

    v2 Changes:
    - Leg extension is now a state variable with finite range (±15cm)
    - Force comes from accelerating rider mass, not free vertical force
    - Energy cost includes isometric (holding) component
    - This naturally enforces pumping: legs must oscillate within range

    Observation Space (13 dimensions):
        [z, vx, vz, theta, omega, leg_pos, leg_vel,
         sin(phase), cos(phase), altitude_error, velocity_error,
         energy_rate, is_stalled]

    Action Space (2 dimensions):
        [leg_velocity_cmd, pitch_torque_normalized] in [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Physical limits for human rider
    RIDER_MASS = 70.0          # kg - rider mass that moves relative to board
    BOARD_MASS = 6.0           # kg - board + foil
    MAX_LEG_EXTENSION = 0.15   # m - leg range of motion (±15cm = 30cm total)
    MAX_LEG_VELOCITY = 1.0     # m/s - max leg extension rate
    MAX_PITCH_TORQUE = 50.0    # Nm - maximum torque from ankle/core
    MAX_POWER = 300.0          # W - sustainable mechanical power output (trained athlete)
    METABOLIC_EFFICIENCY = 0.25  # Muscles are ~25% efficient

    # Target operating conditions
    TARGET_ALTITUDE = 0.15     # m - wing depth below surface
    TARGET_VELOCITY = 4.5      # m/s - target forward velocity (~16 km/h)

    # Geometric constraints (derived from CONFIG)
    # z coordinate: z=0.2 means wing at surface, z<0.2 means submerged
    # Ceiling: wing breaches surface when z >= riding_depth (0.2m)
    # Floor: board touches water when z <= riding_depth - mast_length

    # Episode termination thresholds
    MIN_VELOCITY = 1.5         # m/s - stalled out (raised - can't pump below this)
    MAX_STEPS = 3000           # ~30 seconds at 100 Hz
    MAX_ENERGY = 4500.0        # J - energy budget (300W × 15s = 4.5kJ)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dt: float = 0.01,
        render_mode: Optional[str] = None,
        use_rk4: bool = False,
    ):
        """
        Initialize the pump foil environment.

        Args:
            config: Physics configuration override
            dt: Simulation timestep (default 10ms = 100Hz)
            render_mode: Rendering mode ("human", "rgb_array", or None)
            use_rk4: Use RK4 integration instead of Euler
        """
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode
        self.use_rk4 = use_rk4

        # Initialize physics engine
        self.physics = FoilPhysics(config)

        # Action space: leg velocity command and pitch torque
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: state + leg state + phase + errors + energy + stall
        high = np.array([
            10.0,    # z (altitude)
            20.0,    # vx (forward velocity)
            10.0,    # vz (vertical velocity)
            1.0,     # theta (pitch angle, rad)
            5.0,     # omega (pitch rate)
            1.0,     # leg_position (normalized)
            1.0,     # leg_velocity (normalized)
            1.0,     # sin(phase)
            1.0,     # cos(phase)
            2.0,     # altitude_error
            10.0,    # velocity_error
            1.0,     # energy_rate (normalized)
            1.0,     # is_stalled
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )

        # State
        self.state: Optional[FoilState] = None
        self.forces: Optional[ForceOutput] = None
        self.step_count = 0
        self.phase = 0.0

        # Leg extension state (rider moving relative to board)
        self.leg_position = 0.0  # m, relative to neutral (-0.15 to +0.15)
        self.leg_velocity = 0.0  # m/s

        # Leg position history for computing oscillation amplitude
        self.leg_position_history = []
        self.LEG_HISTORY_LENGTH = 50  # ~0.5s at 100Hz

        # Energy tracking
        self.cumulative_energy = 0.0
        self.last_power = 0.0
        self.metabolic_power = 0.0

        # For rendering
        self._renderer = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        options = options or {}
        initial_velocity = options.get("initial_velocity", self.TARGET_VELOCITY)

        # Compute trim angle for level flight
        from foil_env.foil_physics import compute_trim_angle
        trim_theta = compute_trim_angle(initial_velocity, self.physics.config)

        # Small perturbations
        z_noise = self.np_random.uniform(-0.01, 0.01) if seed is not None else 0
        vx_noise = self.np_random.uniform(-0.1, 0.1) if seed is not None else 0

        actual_vx = initial_velocity + vx_noise
        trim_theta = compute_trim_angle(actual_vx, self.physics.config)

        # Initialize state at equilibrium
        self.state = FoilState(
            x=0.0,
            z=self.TARGET_ALTITUDE + z_noise,
            vx=actual_vx,
            vz=0.0,
            theta=trim_theta,
            omega=0.0,
        )

        # Reset leg state
        self.leg_position = 0.0
        self.leg_velocity = 0.0
        self.leg_position_history = []

        self.step_count = 0
        self.phase = 0.0
        self.cumulative_energy = 0.0
        self.last_power = 0.0
        self.metabolic_power = 0.0

        self.forces = self.physics.calculate_forces(self.state)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep in the environment."""
        assert self.state is not None, "Must call reset() before step()"

        # Parse actions
        leg_velocity_cmd = float(action[0]) * self.MAX_LEG_VELOCITY  # m/s
        pitch_torque = float(action[1]) * self.MAX_PITCH_TORQUE

        # Store previous state
        prev_vx = self.state.vx
        prev_z = self.state.z
        prev_leg_pos = self.leg_position
        prev_leg_vel = self.leg_velocity

        # === LEG DYNAMICS ===
        # Leg moves toward commanded velocity with acceleration limit
        MAX_LEG_ACCEL = 15.0  # m/s^2 - reasonable human leg acceleration

        # Compute desired acceleration
        leg_accel_cmd = (leg_velocity_cmd - self.leg_velocity) / self.dt
        leg_accel = np.clip(leg_accel_cmd, -MAX_LEG_ACCEL, MAX_LEG_ACCEL)

        # Predict next position
        next_velocity = self.leg_velocity + leg_accel * self.dt
        next_position = self.leg_position + next_velocity * self.dt

        # Soft limit: if approaching limit, reduce velocity smoothly
        # This prevents impact forces when hitting the stops
        LIMIT_MARGIN = 0.02  # Start slowing 2cm before limit

        if next_position > self.MAX_LEG_EXTENSION - LIMIT_MARGIN:
            # Approaching upper limit - reduce positive velocity
            overshoot = next_position - (self.MAX_LEG_EXTENSION - LIMIT_MARGIN)
            damping = max(0, 1 - overshoot / LIMIT_MARGIN)
            if next_velocity > 0:
                next_velocity *= damping
            if next_position > self.MAX_LEG_EXTENSION:
                next_position = self.MAX_LEG_EXTENSION
                next_velocity = min(0, next_velocity)

        elif next_position < -self.MAX_LEG_EXTENSION + LIMIT_MARGIN:
            # Approaching lower limit - reduce negative velocity
            overshoot = (-self.MAX_LEG_EXTENSION + LIMIT_MARGIN) - next_position
            damping = max(0, 1 - overshoot / LIMIT_MARGIN)
            if next_velocity < 0:
                next_velocity *= damping
            if next_position < -self.MAX_LEG_EXTENSION:
                next_position = -self.MAX_LEG_EXTENSION
                next_velocity = max(0, next_velocity)

        # Compute actual acceleration (limited by soft stops)
        actual_leg_accel = (next_velocity - self.leg_velocity) / self.dt

        # Update state
        self.leg_velocity = next_velocity
        self.leg_position = next_position

        # Track leg position history for oscillation amplitude
        self.leg_position_history.append(self.leg_position)
        if len(self.leg_position_history) > self.LEG_HISTORY_LENGTH:
            self.leg_position_history.pop(0)

        # === FORCE FROM LEG MOTION ===
        # Force on board = -m_rider * a_leg (reaction force)
        # Positive leg accel = rider accelerating upward relative to board
        # This pushes the board DOWN
        leg_force_raw = -self.RIDER_MASS * actual_leg_accel

        # Limit force to realistic human capability
        MAX_LEG_FORCE = 800.0  # N - max force human legs can apply
        leg_force = np.clip(leg_force_raw, -MAX_LEG_FORCE, MAX_LEG_FORCE)

        # === PHYSICS STEP ===
        if self.use_rk4:
            self.state = self.physics.step_rk4(
                self.state, leg_force, pitch_torque, self.dt
            )
        else:
            self.state = self.physics.step(
                self.state, leg_force, pitch_torque, self.dt
            )

        self.forces = self.physics.calculate_forces(self.state)

        # Update phase
        self.phase += self.dt * 2 * np.pi * 1.5
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # === ENERGY CALCULATION ===
        # Mechanical power: force × velocity
        mechanical_power = abs(leg_force * self.leg_velocity) + abs(pitch_torque * self.state.omega)

        # Simple energy model: mechanical power + small isometric overhead
        # This represents actual work output (~200-300W sustainable for trained athlete)
        isometric_power = abs(leg_force) * 0.05  # Small holding cost
        self.metabolic_power = mechanical_power + isometric_power

        self.last_power = self.metabolic_power
        self.cumulative_energy += self.metabolic_power * self.dt

        self.step_count += 1

        # Check termination
        terminated, termination_reason = self._check_termination()
        truncated = self.step_count >= self.MAX_STEPS

        # Compute reward
        reward = self._compute_reward(
            prev_vx, prev_z, leg_force, pitch_torque, terminated, termination_reason
        )

        obs = self._get_observation()
        info = self._get_info()
        info["termination_reason"] = termination_reason
        info["leg_force"] = leg_force
        info["leg_position"] = self.leg_position
        info["leg_velocity"] = self.leg_velocity
        info["metabolic_power"] = self.metabolic_power

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation array from current state."""
        s = self.state
        f = self.forces

        altitude_error = s.z - self.TARGET_ALTITUDE
        velocity_error = s.vx - self.TARGET_VELOCITY
        energy_rate = self.metabolic_power / self.MAX_POWER

        obs = np.array([
            s.z,
            s.vx,
            s.vz,
            s.theta,
            s.omega,
            self.leg_position / self.MAX_LEG_EXTENSION,  # Normalized
            self.leg_velocity / self.MAX_LEG_VELOCITY,   # Normalized
            np.sin(self.phase),
            np.cos(self.phase),
            altitude_error,
            velocity_error,
            min(1.0, energy_rate),
            float(f.is_stalled),
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        return {
            "x": self.state.x,
            "z": self.state.z,
            "vx": self.state.vx,
            "vz": self.state.vz,
            "theta_deg": np.degrees(self.state.theta),
            "alpha_deg": np.degrees(self.forces.alpha),
            "lift": self.forces.L,
            "drag": self.forces.D,
            "thrust": self.forces.F_hydro_x,
            "power": self.last_power,
            "cumulative_energy": self.cumulative_energy,
            "is_stalled": self.forces.is_stalled,
            "is_breached": self.forces.is_breached,
            "step_count": self.step_count,
        }

    def _check_termination(self) -> Tuple[bool, str]:
        """Check if episode should terminate based on physical constraints."""
        s = self.state
        cfg = self.physics.config

        # Geometric limits from foil configuration
        riding_depth = cfg['riding_depth']  # 0.20m - wing at surface level
        mast_length = cfg['mast_length']    # 0.70m

        # CEILING: Wing breaches surface (no lift possible)
        if s.z >= riding_depth:
            return True, "foil_breached"

        # FLOOR: Board touches water (touchdown crash)
        board_touchdown_z = riding_depth - mast_length  # -0.5m for 0.7m mast
        if s.z <= board_touchdown_z:
            return True, "board_touchdown"

        # Velocity too low to sustain flight
        if s.vx < self.MIN_VELOCITY:
            return True, "stalled_velocity"

        # Excessive pitch angle (loss of control)
        if abs(s.theta) > np.radians(30):  # Reduced from 45 - more realistic
            return True, "excessive_pitch"

        # Energy budget exhausted
        if self.cumulative_energy > self.MAX_ENERGY:
            return True, "energy_exhausted"

        return False, ""

    def _compute_reward(
        self,
        prev_vx: float,
        prev_z: float,
        leg_force: float,
        pitch_torque: float,
        terminated: bool,
        termination_reason: str,
    ) -> float:
        """
        Compute reward for the current step.

        Primary: Stay flying at target altitude AND maintain velocity
        Secondary: Encourage pumping (oscillation), minimize energy
        """
        s = self.state
        f = self.forces

        # 1. Survival bonus
        survival_bonus = 1.0

        # 2. Altitude maintenance
        altitude_error = abs(s.z - self.TARGET_ALTITUDE)
        altitude_reward = np.exp(-altitude_error * 20.0) * 2.0

        # Altitude DRIFT penalty - penalize moving away from target altitude
        # This prevents the "sink while pumping" strategy
        altitude_drift = abs(s.z - self.TARGET_ALTITUDE) - abs(prev_z - self.TARGET_ALTITUDE)
        altitude_drift_penalty = -max(0, altitude_drift) * 100.0  # Strong penalty for drifting away

        # 3. Velocity maintenance - CRITICAL for pumping
        # Must maintain velocity, not just stay airborne
        velocity_error = abs(s.vx - self.TARGET_VELOCITY)
        velocity_reward = np.exp(-velocity_error * 0.2) * 5.0  # Very strong velocity reward

        # Velocity LOSS penalty - heavily penalize velocity decay
        # This is key: survival strategies that let velocity decay should be punished
        velocity_loss = prev_vx - s.vx  # Positive if velocity decreased
        velocity_loss_penalty = -max(0, velocity_loss) * 100.0  # Very strong penalty for losing speed

        # Thrust reward - reward forward force generation (the key to pumping!)
        thrust = f.F_hydro_x
        thrust_reward = max(0, thrust) * 0.05  # Increased: reward positive thrust more

        # 4. Pumping reward - reward leg MOVEMENT (velocity magnitude)
        # This directly encourages the oscillatory motion of pumping
        leg_velocity_magnitude = abs(self.leg_velocity) / self.MAX_LEG_VELOCITY
        pumping_reward = leg_velocity_magnitude * 1.0  # Reward for moving legs

        # Reward oscillation AMPLITUDE - we want large oscillations around zero
        # Compute amplitude as (max - min) / 2 over recent history
        if len(self.leg_position_history) >= 20:  # Need some history
            leg_max = max(self.leg_position_history)
            leg_min = min(self.leg_position_history)
            leg_amplitude = (leg_max - leg_min) / 2
            leg_mean = (leg_max + leg_min) / 2

            # Reward amplitude (target ~0.1m = 0.7 * 0.15m)
            target_amplitude = 0.7 * self.MAX_LEG_EXTENSION
            amplitude_reward = min(1.0, leg_amplitude / target_amplitude) * 2.0

            # Penalize mean bias (should oscillate around zero)
            bias_penalty = -abs(leg_mean) / self.MAX_LEG_EXTENSION * 1.0
        else:
            amplitude_reward = 0.0
            bias_penalty = 0.0

        # Combined leg penalty
        leg_limit_ratio = abs(self.leg_position) / self.MAX_LEG_EXTENSION
        leg_limit_penalty = -1.0 if leg_limit_ratio > 0.95 else 0.0  # Only penalize hard limit

        # 5. Energy efficiency
        power_normalized = self.metabolic_power / self.MAX_POWER
        energy_penalty = -power_normalized * 0.3

        # 6. Stall penalty
        stall_penalty = -2.0 if f.is_stalled else 0.0

        # 7. Crash/exhaustion penalty
        crash_penalty = 0.0
        if terminated:
            if termination_reason == "energy_exhausted":
                crash_penalty = -20.0
            else:
                crash_penalty = -50.0

        reward = (
            survival_bonus +
            altitude_reward +
            altitude_drift_penalty +
            velocity_reward +
            velocity_loss_penalty +
            thrust_reward +
            pumping_reward +
            amplitude_reward +
            bias_penalty +
            leg_limit_penalty +
            energy_penalty +
            stall_penalty +
            crash_penalty
        )

        return reward

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from foil_env.renderer import FoilRenderer
            self._renderer = FoilRenderer(self.render_mode)

        return self._renderer.render(self.state, self.forces, self._get_info())

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment with Gymnasium
gym.register(
    id="PumpFoil-v1",
    entry_point="foil_env.pump_foil_env:PumpFoilEnv",
    max_episode_steps=3000,
)

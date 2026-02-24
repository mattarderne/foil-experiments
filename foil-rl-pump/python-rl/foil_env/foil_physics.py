"""
Foil Physics Engine - Ported from foilphysics/index.html

This module implements the hydrodynamic physics model for a pump foil,
including lift/drag forces from the main wing, stabilizer, and mast.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any

# Default configuration matching foilphysics
CONFIG = {
    'mass': 70.0,              # Rider mass (kg)
    'board_mass': 6.0,         # Board + foil + wet gear (kg)
    'rho': 1000.0,             # Water density (kg/m^3)
    'S': 0.135,                # Wing area (m^2) - e.g., 1350 cm^2
    'AR': 13.9,                # Aspect ratio (Span^2 / Area)
    'Cd0': 0.013,              # Parasitic drag coefficient
    'g': 9.81,                 # Gravity (m/s^2)

    # Stabilizer parameters
    'S_stab': 0.012,           # Stabilizer area (m^2)
    'AR_stab': 8.0,            # Stabilizer aspect ratio
    'fuselage_len': 0.60,      # Mast to stabilizer distance (m)
    'stab_angle': -1.0,        # Stabilizer incidence angle (deg)

    # Operating conditions
    'riding_depth': 0.20,      # Average depth of wing below surface (m)
    'stall_angle': 13.0,       # Stall angle (degrees)

    # Mast geometry
    'mast_length': 0.70,       # Mast length (m) - board to wing distance
    'mast_thickness': 0.015,   # Mast chord (m)
    'Cd_mast': 0.12,           # Mast drag coefficient

    # Rider stance geometry
    # CoM offset from center of lift (CoL). Positive = forward, negative = aft.
    # CoL is approximately at the front wing, which is at the bottom of the mast.
    # Neutral stance: rider CoM roughly above CoL (offset ~0)
    'rider_com_offset': 0.0,   # m - rider center of mass offset from CoL
}


@dataclass
class FoilState:
    """State of the foil system."""
    x: float = 0.0             # Horizontal position (m)
    z: float = 0.2             # Vertical position (m) - positive = above water surface level
    vx: float = 4.5            # Forward velocity (m/s)
    vz: float = 0.0            # Vertical velocity (m/s)
    theta: float = 0.08        # Pitch angle (rad) - ~5 degrees
    omega: float = 0.0         # Pitch angular velocity (rad/s)


@dataclass
class ForceOutput:
    """Output forces and derived quantities from physics calculation."""
    # Lift and drag forces (N)
    L: float = 0.0             # Lift magnitude
    D: float = 0.0             # Drag magnitude
    Lx: float = 0.0            # Lift x-component
    Lz: float = 0.0            # Lift z-component
    Dx: float = 0.0            # Drag x-component
    Dz: float = 0.0            # Drag z-component

    # Stabilizer forces
    L_stab: float = 0.0
    L_stab_x: float = 0.0
    L_stab_z: float = 0.0
    D_stab: float = 0.0
    D_stab_x: float = 0.0
    D_stab_z: float = 0.0

    # Mast drag
    D_mast: float = 0.0
    D_mast_x: float = 0.0
    D_mast_z: float = 0.0

    # Total hydrodynamic forces
    F_hydro_x: float = 0.0
    F_hydro_z: float = 0.0

    # Angles
    alpha: float = 0.0         # Angle of attack (rad)
    gamma: float = 0.0         # Flight path angle (rad)

    # Coefficients
    Cl: float = 0.0
    Cd: float = 0.0

    # State flags
    is_stalled: bool = False
    is_breached: bool = False

    # Moments about center of lift
    M_stab: float = 0.0        # Stabilizer moment
    M_weight: float = 0.0      # Rider weight moment (from CoM offset)


class FoilPhysics:
    """
    Physics engine for pump foiling simulation.

    This class calculates the hydrodynamic forces acting on a foil
    given the current state and control inputs.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize physics engine with configuration.

        Args:
            config: Dictionary of physics parameters. Uses defaults if not provided.
        """
        self.config = {**CONFIG, **(config or {})}
        self._precompute()

    def update_config(self, new_config: Dict[str, Any]):
        """Update config and recompute derived quantities."""
        self.config.update(new_config)
        self._precompute()

    def _precompute(self):
        """Precompute derived quantities from config."""
        cfg = self.config

        # Total mass
        self.m_total = cfg['mass'] + cfg['board_mass']

        # Lift slope for finite wing (Prandtl lifting line theory)
        self.lift_slope = (2 * np.pi) / (1 + 2 / cfg['AR'])
        self.lift_slope_stab = (2 * np.pi) / (1 + 2 / cfg['AR_stab'])

        # Stall angle in radians
        self.stall_rad = np.radians(cfg['stall_angle'])
        self.stab_angle_rad = np.radians(cfg['stab_angle'])

        # Induced drag factor
        self.k_induced = 1 / (np.pi * cfg['AR'])
        self.k_induced_stab = 1 / (np.pi * cfg['AR_stab'])

    def calculate_forces(self, state: FoilState) -> ForceOutput:
        """
        Calculate all hydrodynamic forces given current state.

        Args:
            state: Current foil state (position, velocity, attitude)

        Returns:
            ForceOutput with all calculated forces and derived quantities
        """
        cfg = self.config
        output = ForceOutput()

        vx = state.vx
        vz = state.vz
        theta = state.theta
        z = state.z
        omega = state.omega

        # Flight path angle and angle of attack
        gamma = np.arctan2(vz, vx)
        alpha = theta - gamma
        v_sq = vx * vx + vz * vz

        output.gamma = gamma
        output.alpha = alpha

        # --- MAIN WING AERODYNAMICS ---
        alpha_abs = abs(alpha)
        sign_alpha = np.sign(alpha)

        if alpha_abs < self.stall_rad:
            # Linear region
            Cl = self.lift_slope * alpha
            Cd = cfg['Cd0'] + self.k_induced * Cl * Cl
            output.is_stalled = False
        else:
            # Stall region - Cl saturates, Cd increases dramatically
            max_Cl = self.lift_slope * self.stall_rad
            Cl = max_Cl * sign_alpha
            excess_rad = alpha_abs - self.stall_rad
            Cd = 0.05 + excess_rad * 2.0
            output.is_stalled = True

        output.Cl = Cl
        output.Cd = Cd

        # Lift and drag magnitudes
        q = 0.5 * cfg['rho'] * v_sq  # Dynamic pressure
        L_mag = q * cfg['S'] * Cl
        D_mag = q * cfg['S'] * Cd

        output.L = L_mag
        output.D = D_mag

        # Resolve into x-z components
        # Lift perpendicular to velocity, Drag parallel
        Lx = -L_mag * np.sin(gamma)
        Lz = L_mag * np.cos(gamma)
        Dx = -D_mag * np.cos(gamma)
        Dz = -D_mag * np.sin(gamma)

        # --- STABILIZER AERODYNAMICS ---
        # Downwash angle from main wing
        epsilon = (2 * Cl) / (np.pi * cfg['AR'])

        # Vertical velocity at stabilizer (accounting for pitch rotation)
        vz_stab = vz - (omega * cfg['fuselage_len'])
        gamma_stab = np.arctan2(vz_stab, vx)

        # Stabilizer angle of attack
        alpha_stab = theta + self.stab_angle_rad - gamma_stab - epsilon

        # Stabilizer lift coefficient (assuming no stall for small stabilizer)
        Cl_stab = self.lift_slope_stab * alpha_stab

        # Stabilizer forces
        v_sq_stab = vx * vx + vz_stab * vz_stab
        q_stab = 0.5 * cfg['rho'] * v_sq_stab
        L_stab_mag = q_stab * cfg['S_stab'] * Cl_stab

        # Stabilizer drag (profile + induced)
        Cd_stab = 0.02 + self.k_induced_stab * Cl_stab * Cl_stab
        D_stab_mag = q_stab * cfg['S_stab'] * Cd_stab

        # Resolve stabilizer forces
        L_stab_x = -L_stab_mag * np.sin(gamma_stab)
        L_stab_z = L_stab_mag * np.cos(gamma_stab)
        D_stab_x = -D_stab_mag * np.cos(gamma_stab)
        D_stab_z = -D_stab_mag * np.sin(gamma_stab)

        output.L_stab = L_stab_mag
        output.L_stab_x = L_stab_x
        output.L_stab_z = L_stab_z
        output.D_stab = D_stab_mag
        output.D_stab_x = D_stab_x
        output.D_stab_z = D_stab_z

        # --- MAST DRAG & SURFACE BREACH ---
        depth = cfg['riding_depth'] - z

        if depth < 0:
            # Wing breached surface - no hydrodynamic forces
            Lx = Lz = Dx = Dz = 0.0
            L_stab_x = L_stab_z = D_stab_x = D_stab_z = 0.0
            depth = 0.0
            output.is_breached = True
        else:
            output.is_breached = False

        # Mast drag (proportional to submerged depth)
        area_mast = depth * cfg['mast_thickness']
        D_mast_mag = 0.5 * cfg['rho'] * area_mast * cfg['Cd_mast'] * v_sq
        D_mast_x = -D_mast_mag * np.cos(gamma)
        D_mast_z = -D_mast_mag * np.sin(gamma)

        output.D_mast = D_mast_mag
        output.D_mast_x = D_mast_x
        output.D_mast_z = D_mast_z

        # Store resolved forces (after breach check)
        output.Lx = Lx
        output.Lz = Lz
        output.Dx = Dx
        output.Dz = Dz

        # --- TOTAL HYDRODYNAMIC FORCES ---
        output.F_hydro_x = Lx + Dx + L_stab_x + D_stab_x + D_mast_x
        output.F_hydro_z = Lz + Dz + L_stab_z + D_stab_z + D_mast_z

        # Stabilizer moment (about center of lift)
        # Torque = Force * Distance. Nose up = positive.
        # Stabilizer downforce (negative Lz) creates nose-up moment
        output.M_stab = -(L_stab_z * cfg['fuselage_len'])

        # Rider weight moment (about center of lift)
        # If rider CoM is forward of CoL (positive offset), weight creates nose-down moment
        # M = -W * x_offset (W is positive, forward offset gives negative/nose-down moment)
        rider_weight = cfg['mass'] * cfg['g']
        output.M_weight = -rider_weight * cfg['rider_com_offset']

        return output

    def step(self, state: FoilState, leg_force: float, pitch_torque: float,
             dt: float = 0.01) -> FoilState:
        """
        Advance physics by one timestep using Euler integration.

        Pumping is like jumping - the rider pushes down with legs, creating
        a vertical force that moves the board/foil up and down. This vertical
        motion combined with forward velocity creates changing angle of attack,
        generating net thrust.

        Args:
            state: Current foil state
            leg_force: Vertical force from rider's legs (N) - positive = push down
            pitch_torque: Torque applied by rider (Nm) - positive = pitch up
            dt: Timestep (s)

        Returns:
            New state after integration
        """
        cfg = self.config

        # Calculate hydrodynamic forces
        forces = self.calculate_forces(state)

        # Weight
        W = -self.m_total * cfg['g']

        # Net forces (leg force acts vertically - pushing board down)
        Fx = forces.F_hydro_x
        Fz = forces.F_hydro_z + W - leg_force

        # Pump thrust: efficient pumping generates forward thrust
        # Real pump foiling converts vertical motion into forward thrust through
        # the foil's lift vector tilting. This is partially captured in Lx via
        # flight path angle, but we add explicit pump thrust to model the
        # efficiency of coordinated pumping motion.
        #
        # For sustained flight at ~4.5 m/s, need ~20-30N thrust to overcome drag.
        # With efficiency=0.15, vz=0.5, L=800: thrust = 0.15 * 0.5 * 800 = 60N peak
        # Average over cycle (sin wave) ≈ 60 * 2/π ≈ 38N - sufficient for sustained flight.
        pump_efficiency = cfg.get('pump_thrust_efficiency', 0.15)
        pump_thrust = 0.0
        if abs(state.vz) > 0.1:  # Significant vertical motion
            # Thrust proportional to vertical speed and lift
            pump_thrust = pump_efficiency * abs(state.vz) * abs(forces.L)
            Fx += pump_thrust  # Always positive (forward)

        # Accelerations
        ax = Fx / self.m_total
        az = Fz / self.m_total

        # Moment of inertia estimate (rider + board rotating about CoL)
        I = self.m_total * 0.25  # kg*m^2

        # Angular acceleration from all moments:
        # - Stabilizer moment (automatic pitch stability)
        # - Rider weight moment (from CoM offset - static trim)
        # - Rider pitch torque (from leg asymmetry, arm swing, lean)
        M_total = forces.M_stab + forces.M_weight + pitch_torque
        alpha_dot = M_total / I

        # Euler integration
        new_state = FoilState(
            x=state.x + state.vx * dt,
            z=state.z + state.vz * dt,
            vx=state.vx + ax * dt,
            vz=state.vz + az * dt,
            theta=state.theta + state.omega * dt,
            omega=state.omega + alpha_dot * dt,
        )

        return new_state

    def step_rk4(self, state: FoilState, leg_force: float, pitch_torque: float,
                 dt: float = 0.01) -> FoilState:
        """
        Advance physics by one timestep using RK4 integration.

        More accurate than Euler, especially for oscillatory dynamics.
        """
        def derivatives(s: FoilState):
            forces = self.calculate_forces(s)
            W = -self.m_total * self.config['g']

            Fx = forces.F_hydro_x
            Fz = forces.F_hydro_z + W - leg_force

            ax = Fx / self.m_total
            az = Fz / self.m_total

            I = self.m_total * 0.25
            M_total = forces.M_stab + forces.M_weight + pitch_torque
            alpha_dot = M_total / I

            return np.array([s.vx, s.vz, ax, az, s.omega, alpha_dot])

        def state_to_array(s: FoilState):
            return np.array([s.x, s.z, s.vx, s.vz, s.theta, s.omega])

        def array_to_state(arr: np.ndarray) -> FoilState:
            return FoilState(x=arr[0], z=arr[1], vx=arr[2], vz=arr[3],
                           theta=arr[4], omega=arr[5])

        y = state_to_array(state)

        k1 = derivatives(state)
        k2 = derivatives(array_to_state(y + 0.5 * dt * k1))
        k3 = derivatives(array_to_state(y + 0.5 * dt * k2))
        k4 = derivatives(array_to_state(y + dt * k3))

        y_new = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return array_to_state(y_new)


def compute_trim_angle(vx: float, config: Dict[str, Any] = None) -> float:
    """
    Compute the trim pitch angle for level flight at given speed.
    Uses iterative search to account for all forces (wing + stabilizer + mast).

    Args:
        vx: Forward velocity (m/s)
        config: Physics configuration

    Returns:
        Trim angle in radians
    """
    cfg = {**CONFIG, **(config or {})}
    physics = FoilPhysics(cfg)
    m_total = cfg['mass'] + cfg['board_mass']
    W = m_total * cfg['g']

    # Binary search for equilibrium angle
    theta_low = np.radians(2.0)
    theta_high = np.radians(12.0)

    for _ in range(20):  # Binary search iterations
        theta_mid = (theta_low + theta_high) / 2
        state = FoilState(z=cfg['riding_depth'], vx=vx, vz=0, theta=theta_mid, omega=0)
        forces = physics.calculate_forces(state)
        net_fz = forces.F_hydro_z - W

        if net_fz > 0:  # Too much lift, reduce angle
            theta_high = theta_mid
        else:  # Not enough lift, increase angle
            theta_low = theta_mid

        if abs(net_fz) < 1.0:  # Within 1N of equilibrium
            break

    return theta_mid


if __name__ == "__main__":
    # Quick test
    physics = FoilPhysics()
    state = FoilState()

    print("Initial state:", state)
    print(f"Trim angle at {state.vx} m/s: {np.degrees(compute_trim_angle(state.vx)):.2f} deg")

    forces = physics.calculate_forces(state)
    print(f"\nForces at initial state:")
    print(f"  Lift: {forces.L:.1f} N")
    print(f"  Drag: {forces.D:.1f} N")
    print(f"  AoA: {np.degrees(forces.alpha):.2f} deg")
    print(f"  Stalled: {forces.is_stalled}")

    # Simulate a few steps
    print("\nSimulating 10 steps with no control input...")
    for i in range(10):
        state = physics.step(state, leg_force=0, pitch_torque=0, dt=0.01)
        print(f"  Step {i+1}: z={state.z:.3f}m, vx={state.vx:.2f}m/s, theta={np.degrees(state.theta):.2f}deg")

"""
Central configuration for foil physics and testing.

ALL test files and environments should import from here.
DO NOT define physics parameters inline in test files.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import numpy as np


# =============================================================================
# PHYSICS CONSTANTS (not configurable - these are physical laws)
# =============================================================================

class PhysicsConstants:
    """Physical constants that should never change."""
    RHO_WATER = 1000.0      # kg/m³ - water density
    GRAVITY = 9.81          # m/s² - gravitational acceleration

    # Aerodynamic model constants
    STALL_DRAG_BASE = 0.05      # Base Cd when stalled
    STALL_DRAG_SLOPE = 2.0      # Cd increase per radian above stall
    STAB_DRAG_PROFILE = 0.02    # Stabilizer profile drag coefficient

    # Integration constants
    MOMENT_INERTIA_SCALE = 0.25  # I = m_total * scale (simplified)
    PUMP_VZ_THRESHOLD = 0.02     # Minimum vz to activate pump thrust


# =============================================================================
# TERMINATION THRESHOLDS
# =============================================================================

@dataclass
class TerminationThresholds:
    """Episode termination conditions."""
    z_max: float = 0.25      # Breach ceiling (m)
    z_min: float = -0.50     # Touchdown floor (m)
    vx_min: float = 1.5      # Stall velocity (m/s)
    vx_max: float = 10.0     # Maximum velocity (m/s)
    pitch_max: float = 0.8   # Maximum pitch (rad, ~45°)
    energy_max: float = 50000  # Maximum energy budget (J)


# =============================================================================
# FOIL GEOMETRY PRESETS
# =============================================================================

@dataclass
class FoilGeometry:
    """Foil physical dimensions and drag characteristics."""
    name: str
    S: float                 # Wing area (m²)
    S_stab: float           # Stabilizer area (m²)
    AR: float = 8.0         # Aspect ratio
    AR_stab: float = 8.0    # Stabilizer aspect ratio
    stab_angle: float = -4.0  # Stabilizer incidence (deg)
    Cd0: float = 0.008      # Parasitic drag coefficient
    Cd_mast: float = 0.06   # Mast drag coefficient
    stall_angle: float = 13.0  # Stall angle (deg)
    mast_length: float = 0.70  # Mast length (m)
    mast_thickness: float = 0.015  # Mast chord (m)
    fuselage_len: float = 0.60  # Fuselage length (m)
    riding_depth: float = 0.20  # Nominal riding depth (m)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for FoilPhysics."""
        return asdict(self)


# Predefined foil sizes
FOIL_BEGINNER = FoilGeometry(
    name="beginner",
    S=0.22,           # 2200 cm² - large, forgiving
    S_stab=0.044,     # 20% of main
)

FOIL_TRAINING = FoilGeometry(
    name="training",
    S=0.20,           # 2000 cm² - standard training
    S_stab=0.040,
)

FOIL_INTERMEDIATE = FoilGeometry(
    name="intermediate",
    S=0.18,           # 1800 cm² - moderate difficulty
    S_stab=0.036,
)

FOIL_ADVANCED = FoilGeometry(
    name="advanced",
    S=0.15,           # 1500 cm² - challenging
    S_stab=0.030,
)

FOIL_REALISTIC = FoilGeometry(
    name="realistic",
    S=0.135,          # 1350 cm² - matches original foilphysics
    S_stab=0.027,
    AR=13.9,          # Original AR
    Cd0=0.013,        # Original Cd0
    Cd_mast=0.12,     # Original mast drag
    stab_angle=-1.0,  # Original stab angle
)


# =============================================================================
# RIDER CONFIGURATION
# =============================================================================

@dataclass
class RiderConfig:
    """Rider physical properties."""
    mass: float = 75.0       # Rider mass (kg)
    board_mass: float = 6.0  # Board + foil mass (kg)

    @property
    def total_mass(self) -> float:
        return self.mass + self.board_mass

    @property
    def weight(self) -> float:
        return self.total_mass * PhysicsConstants.GRAVITY


# =============================================================================
# PUMP PHYSICS CONFIGURATION
# =============================================================================

@dataclass
class PumpConfig:
    """Pump thrust and efficiency settings.

    pump_thrust_efficiency represents unsteady aero effects not explicitly modeled:
    - 0.00: Pure steady-state aero (incomplete physics)
    - 0.08: Conservative estimate (recommended minimum)
    - 0.10: Balanced (recommended for training)
    - 0.15: Upper bound of physical justification
    - 0.18+: Overtuned (training wheels)

    See docs/pump-physics-explained.md for derivation.
    """
    pump_thrust_efficiency: float = 0.10  # Recommended default

    # Pump motion parameters (for scripted tests)
    frequency: float = 2.0       # Hz
    vz_amplitude: float = 0.8    # m/s
    pitch_modulation: float = 0.0  # rad


# Presets for different training phases
PUMP_REALISTIC = PumpConfig(pump_thrust_efficiency=0.08)
PUMP_TRAINING = PumpConfig(pump_thrust_efficiency=0.10)
PUMP_EASY = PumpConfig(pump_thrust_efficiency=0.15)
PUMP_NO_ASSIST = PumpConfig(pump_thrust_efficiency=0.0)


# =============================================================================
# INITIAL STATE CONFIGURATION
# =============================================================================

@dataclass
class InitialState:
    """Initial conditions for simulation."""
    x: float = 0.0
    z: float = -0.10         # Start slightly below neutral
    vx: float = 4.0          # Forward velocity (m/s)
    vz: float = 0.0
    theta: float = 0.08      # ~5° pitch
    omega: float = 0.0


# =============================================================================
# COMBINED SIMULATION CONFIG
# =============================================================================

@dataclass
class SimConfig:
    """Complete simulation configuration."""
    foil: FoilGeometry = field(default_factory=lambda: FOIL_TRAINING)
    rider: RiderConfig = field(default_factory=RiderConfig)
    pump: PumpConfig = field(default_factory=lambda: PUMP_TRAINING)
    termination: TerminationThresholds = field(default_factory=TerminationThresholds)
    initial: InitialState = field(default_factory=InitialState)
    dt: float = 0.01

    def to_physics_config(self) -> Dict[str, Any]:
        """Generate config dict for FoilPhysics."""
        config = self.foil.to_dict()
        config['mass'] = self.rider.mass
        config['board_mass'] = self.rider.board_mass
        config['rho'] = PhysicsConstants.RHO_WATER
        config['g'] = PhysicsConstants.GRAVITY
        config['pump_thrust_efficiency'] = self.pump.pump_thrust_efficiency
        return config


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Standard test configuration (used by most tests)
TEST_CONFIG = SimConfig(
    foil=FOIL_TRAINING,
    pump=PUMP_TRAINING,
)

# Validation against original foilphysics
VALIDATION_CONFIG = SimConfig(
    foil=FOIL_REALISTIC,
    pump=PUMP_NO_ASSIST,
    rider=RiderConfig(mass=70.0),
)

# Easy mode for initial learning
EASY_CONFIG = SimConfig(
    foil=FOIL_BEGINNER,
    pump=PUMP_EASY,
)

# Realistic/hard mode
HARD_CONFIG = SimConfig(
    foil=FOIL_ADVANCED,
    pump=PUMP_REALISTIC,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config(name: str) -> SimConfig:
    """Get a named configuration."""
    configs = {
        'test': TEST_CONFIG,
        'validation': VALIDATION_CONFIG,
        'easy': EASY_CONFIG,
        'hard': HARD_CONFIG,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    return configs[name]


def list_foils() -> Dict[str, FoilGeometry]:
    """List all available foil presets."""
    return {
        'beginner': FOIL_BEGINNER,
        'training': FOIL_TRAINING,
        'intermediate': FOIL_INTERMEDIATE,
        'advanced': FOIL_ADVANCED,
        'realistic': FOIL_REALISTIC,
    }


def list_pump_configs() -> Dict[str, PumpConfig]:
    """List all pump configuration presets."""
    return {
        'realistic': PUMP_REALISTIC,
        'training': PUMP_TRAINING,
        'easy': PUMP_EASY,
        'no_assist': PUMP_NO_ASSIST,
    }


# =============================================================================
# DOCUMENTATION
# =============================================================================

CONFIG_DOCS = """
FOIL CONFIGURATION GUIDE
========================

Foil Sizes (by difficulty):
  beginner (2200 cm²)    - Large, very forgiving, hard to stall
  training (2000 cm²)    - Standard for RL training
  intermediate (1800 cm²) - Moderate challenge
  advanced (1500 cm²)    - Requires good technique
  realistic (1350 cm²)   - Matches real pump foils

Pump Thrust Efficiency:
  0.00 - No assist (pure lift-vector physics, crashes quickly)
  0.08 - Conservative (physically justified minimum)
  0.10 - Training (recommended for RL)
  0.15 - Easy mode (training wheels)

Usage:
  from foil_config import TEST_CONFIG, FOIL_TRAINING

  # Use preset
  physics = FoilPhysics(TEST_CONFIG.to_physics_config())

  # Customize
  config = SimConfig(foil=FOIL_ADVANCED, pump=PUMP_REALISTIC)
"""

if __name__ == "__main__":
    print(CONFIG_DOCS)
    print("\nAvailable foils:")
    for name, foil in list_foils().items():
        print(f"  {name}: {foil.S*10000:.0f} cm²")
    print("\nPump configs:")
    for name, pump in list_pump_configs().items():
        print(f"  {name}: efficiency={pump.pump_thrust_efficiency}")

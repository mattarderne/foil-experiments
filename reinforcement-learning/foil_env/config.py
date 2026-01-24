"""
Central configuration for pump foil RL environment.

Single source of truth for all configurations.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class FoilConfig:
    """Foil hydrodynamic configuration."""
    S: float = 0.18            # Wing area (m²)
    S_stab: float = 0.035      # Stabilizer area (m²)
    stab_angle: float = -4.0   # Stabilizer angle (deg)
    AR: float = 8              # Aspect ratio

    # Additional physics params (with defaults from foil_physics.py)
    mast_length: float = 0.70  # m
    riding_depth: float = 0.20 # m

    # Drag coefficients
    Cd0: float = 0.013         # Parasitic drag coefficient
    Cd_mast: float = 0.12      # Mast drag coefficient

    # Pump thrust efficiency - how well pumping converts to forward thrust
    # Real pump foiling generates thrust through coordinated pitch-heave motion
    # Higher values allow sustained flight with less perfect coordination
    pump_thrust_efficiency: float = 0.20  # Tuned for sustained flight capability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for FoilPhysics compatibility."""
        return asdict(self)


@dataclass
class LimbConfig:
    """Limb physics configuration."""
    # Arm configuration
    arm_start_position: float = np.pi / 2  # π/2 = DOWN (not 0!)
    max_arm_swing: float = 2.0             # rad from center
    arm_centering_spring: float = 2.0      # Spring constant pulling to neutral
    max_arm_velocity: float = 15.0         # rad/s
    max_arm_accel: float = 30.0            # rad/s²

    # Leg configuration
    max_leg_extension: float = 0.15        # m
    max_leg_velocity: float = 2.0          # m/s
    max_leg_accel: float = 15.0            # m/s²

    # Waist configuration
    max_waist_angle: float = 0.3           # rad
    max_waist_velocity: float = 2.0        # rad/s
    max_waist_accel: float = 15.0          # rad/s²
    waist_start_position: float = 0.1      # rad (slight forward lean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    foil: FoilConfig = field(default_factory=FoilConfig)
    limbs: LimbConfig = field(default_factory=LimbConfig)

    # Training parameters
    curriculum_phase: int = 2        # 0=forced pump, 1=blended, 2=free
    agent_blend: float = 0.0         # How much agent controls legs
    target_velocity: float = 4.5     # m/s
    target_altitude: float = 0.10    # m (10cm)
    pump_frequency: float = 2.0      # Hz

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            'foil': self.foil.to_dict(),
            'limbs': self.limbs.to_dict(),
            'curriculum_phase': self.curriculum_phase,
            'agent_blend': self.agent_blend,
            'target_velocity': self.target_velocity,
            'target_altitude': self.target_altitude,
            'pump_frequency': self.pump_frequency,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        foil = FoilConfig(**data.get('foil', {}))
        limbs = LimbConfig(**data.get('limbs', {}))

        return cls(
            foil=foil,
            limbs=limbs,
            curriculum_phase=data.get('curriculum_phase', 2),
            agent_blend=data.get('agent_blend', 0.0),
            target_velocity=data.get('target_velocity', 4.5),
            target_altitude=data.get('target_altitude', 0.10),
            pump_frequency=data.get('pump_frequency', 2.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict."""
        return {
            'foil': self.foil.to_dict(),
            'limbs': self.limbs.to_dict(),
            'curriculum_phase': self.curriculum_phase,
            'agent_blend': self.agent_blend,
            'target_velocity': self.target_velocity,
            'target_altitude': self.target_altitude,
            'pump_frequency': self.pump_frequency,
        }


# =============================================================================
# Predefined configurations
# =============================================================================

# Training foil - larger stabilizer for stability during learning
# Updated physics for sustained pump flight
TRAINING_FOIL = FoilConfig(
    S=0.18,
    S_stab=0.035,
    stab_angle=-4.0,
    AR=8,
    Cd0=0.008,                    # Modern clean foil
    Cd_mast=0.06,                 # Streamlined carbon mast
    pump_thrust_efficiency=0.20,  # Efficient pumping
)

# Performance/"big" foil - larger wing area, more forgiving
# Good for learning pump coordination
BIG_FOIL = FoilConfig(
    S=0.22,           # 2200 cm² - more lift, more forgiving
    S_stab=0.044,     # Proportionally scaled
    stab_angle=-4.0,
    AR=8,
    Cd0=0.013,
    Cd_mast=0.12,
    pump_thrust_efficiency=0.20,
)

# Mocap-matched foil - based on real pump foiling data at 4.0 m/s
# Smaller foil requires better technique
MOCAP_FOIL = FoilConfig(
    S=0.12,           # 1200 cm² - matches real foil
    S_stab=0.024,     # Proportionally scaled stabilizer
    stab_angle=-4.0,
    AR=8,
    Cd0=0.013,
    Cd_mast=0.12,
    pump_thrust_efficiency=0.20,
)

# Default limb config with corrected arm start position
DEFAULT_LIMBS = LimbConfig(
    arm_start_position=np.pi / 2,  # DOWN position
    arm_centering_spring=2.0,
)

# Default training config
DEFAULT_CONFIG = TrainingConfig(
    foil=TRAINING_FOIL,
    limbs=DEFAULT_LIMBS,
)

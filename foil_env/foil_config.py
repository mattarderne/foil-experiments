"""
Foil Configuration - Maps to real foil terminology.

Real foil geometry:

    BOARD (top view)     SIDE VIEW
    ┌─────────────┐
    │   ○   ○     │      ══════════ Board
    └─────────────┘           │
          │                   │ Mast (70-90cm)
          │                   │
    ┌─────┴─────┐            ═╪═══════════ Fuselage
    │  FUSELAGE │         Front    Stab
    └───────────┘         Wing     Wing

Angles (all relative to fuselage centerline):
- front_wing_angle: Front wing incidence (shim), typically 0-2°
- stab_angle: Stabilizer incidence, typically -1° to -2° (toed down)
- Board is perpendicular to mast

Typical foil setups:
- Pump foil (beginner): 2000-2500cm² wing, 200-250cm² stab
- Pump foil (advanced): 1200-1800cm² wing, 150-200cm² stab
- Surf foil: 1200-1600cm² wing, 200-250cm² stab
- Wing foil: 1000-2000cm² wing, 200-300cm² stab

Stabilizer sizing rule of thumb (from user):
- 800cm² wing → ~150cm² stab (18.75% ratio)
- 2200cm² wing → ~200cm² stab (9% ratio)
- Larger wings need proportionally smaller stabs
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class FoilGeometry:
    """
    Physical foil geometry matching real foil specifications.

    All areas in cm², angles in degrees, lengths in cm.
    """
    # Front wing
    front_wing_area_cm2: float = 1800.0    # cm² (typical pump foil)
    front_wing_span_cm: float = 85.0       # cm (wingspan)
    front_wing_angle_deg: float = 1.0      # Incidence angle (shim), relative to fuse

    # Stabilizer
    stab_area_cm2: float = 180.0           # cm² (scaled for 1800cm² wing)
    stab_span_cm: float = 42.0             # cm
    stab_angle_deg: float = -1.5           # Incidence, relative to fuse (toed down)

    # Fuselage
    fuselage_length_cm: float = 60.0       # cm (mast to stab distance)

    # Mast
    mast_length_cm: float = 75.0           # cm (board to fuse)
    mast_chord_cm: float = 12.0            # cm (mast thickness/chord)

    @property
    def front_wing_AR(self) -> float:
        """Front wing aspect ratio."""
        return (self.front_wing_span_cm ** 2) / self.front_wing_area_cm2

    @property
    def stab_AR(self) -> float:
        """Stabilizer aspect ratio."""
        return (self.stab_span_cm ** 2) / self.stab_area_cm2

    @property
    def stab_ratio(self) -> float:
        """Stabilizer to front wing area ratio."""
        return self.stab_area_cm2 / self.front_wing_area_cm2

    def to_physics_config(self) -> Dict[str, Any]:
        """Convert to physics engine config (SI units)."""
        return {
            # Front wing
            'S': self.front_wing_area_cm2 / 10000,  # m²
            'AR': self.front_wing_AR,
            'front_wing_angle': self.front_wing_angle_deg,  # deg

            # Stabilizer
            'S_stab': self.stab_area_cm2 / 10000,  # m²
            'AR_stab': self.stab_AR,
            'stab_angle': self.stab_angle_deg,  # deg

            # Geometry
            'fuselage_len': self.fuselage_length_cm / 100,  # m
            'mast_length': self.mast_length_cm / 100,  # m
            'mast_thickness': self.mast_chord_cm / 100,  # m
        }

    def __str__(self) -> str:
        return (
            f"FoilGeometry:\n"
            f"  Front wing: {self.front_wing_area_cm2:.0f}cm² @ {self.front_wing_angle_deg:+.1f}° "
            f"(span={self.front_wing_span_cm:.0f}cm, AR={self.front_wing_AR:.1f})\n"
            f"  Stabilizer: {self.stab_area_cm2:.0f}cm² @ {self.stab_angle_deg:+.1f}° "
            f"(span={self.stab_span_cm:.0f}cm, AR={self.stab_AR:.1f})\n"
            f"  Stab ratio: {self.stab_ratio*100:.1f}%\n"
            f"  Fuselage: {self.fuselage_length_cm:.0f}cm\n"
            f"  Mast: {self.mast_length_cm:.0f}cm (chord={self.mast_chord_cm:.0f}cm)"
        )


# Predefined foil setups
# AR = span² / area, so span = sqrt(AR * area)
# Target AR ~8-10 for front wings, ~10 for stabs

PUMP_FOIL_BEGINNER = FoilGeometry(
    front_wing_area_cm2=2200,
    front_wing_span_cm=133,      # AR=8.0 (sqrt(8*2200))
    front_wing_angle_deg=1.0,
    stab_area_cm2=200,           # User guidance: 200cm² for 2200cm² wing (9% ratio)
    stab_span_cm=45,             # AR=10.1
    stab_angle_deg=-2.0,         # Tuned for stable glide at ~4.5m/s
    fuselage_length_cm=65,
    mast_length_cm=75,
    mast_chord_cm=12,
)

PUMP_FOIL_INTERMEDIATE = FoilGeometry(
    front_wing_area_cm2=1800,
    front_wing_span_cm=120,      # AR=8.0 (sqrt(8*1800))
    front_wing_angle_deg=1.0,
    stab_area_cm2=180,           # ~10% ratio
    stab_span_cm=42,             # AR=9.8
    stab_angle_deg=-1.5,
    fuselage_length_cm=60,
    mast_length_cm=75,
    mast_chord_cm=12,
)

PUMP_FOIL_ADVANCED = FoilGeometry(
    front_wing_area_cm2=1200,
    front_wing_span_cm=98,       # AR=8.0 (sqrt(8*1200))
    front_wing_angle_deg=0.5,
    stab_area_cm2=165,           # ~14% ratio (interpolated between 150 and 180)
    stab_span_cm=40,             # AR=9.7
    stab_angle_deg=-1.0,
    fuselage_length_cm=55,
    mast_length_cm=75,
    mast_chord_cm=10,
)

PUMP_FOIL_SMALL = FoilGeometry(
    front_wing_area_cm2=800,
    front_wing_span_cm=80,       # AR=8.0 (sqrt(8*800))
    front_wing_angle_deg=0.5,
    stab_area_cm2=150,           # User guidance: 150cm² for 800cm² wing (18.75% ratio)
    stab_span_cm=38,             # AR=9.6
    stab_angle_deg=-1.0,
    fuselage_length_cm=55,
    mast_length_cm=75,
    mast_chord_cm=10,
)


def print_current_vs_real():
    """Compare our current config to real foil specs."""
    print("=" * 70)
    print("CURRENT CONFIG vs REAL FOIL SPECS")
    print("=" * 70)

    # What we were using in MuJoCo tests
    print("\nWHAT WE WERE USING (MuJoCo tests):")
    print(f"  Front wing: 2200cm² (S=0.22m²)")
    print(f"  Stabilizer: 440cm² (S_stab=0.044m²)  <-- WAY TOO BIG!")
    print(f"  Stab ratio: 20%  <-- Should be ~9% for this wing size")
    print(f"  Stab angle: -1° to -3° (varied during testing)")
    print(f"  Front wing angle: NOT EXPLICIT (assumed 0°)")

    print("\n" + "-" * 70)
    print("\nCORRECT SIZING (per user guidance):")
    print(f"  800cm² wing  → 150cm² stab (18.75% ratio)")
    print(f"  2200cm² wing → 200cm² stab (9% ratio)")

    print("\n" + "-" * 70)
    print("\nPREDEFINED FOIL SETUPS:")

    for name, foil in [
        ("Beginner (2200cm²)", PUMP_FOIL_BEGINNER),
        ("Intermediate (1800cm²)", PUMP_FOIL_INTERMEDIATE),
        ("Advanced (1200cm²)", PUMP_FOIL_ADVANCED),
        ("Small (800cm²)", PUMP_FOIL_SMALL),
    ]:
        print(f"\n{name}:")
        print(f"  {foil}")


if __name__ == "__main__":
    print_current_vs_real()

    print("\n" + "=" * 70)
    print("PHYSICS CONFIG FOR BEGINNER FOIL:")
    print("=" * 70)
    config = PUMP_FOIL_BEGINNER.to_physics_config()
    for k, v in config.items():
        print(f"  {k}: {v}")

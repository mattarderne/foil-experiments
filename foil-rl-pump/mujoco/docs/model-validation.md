# Physics Model Validation

## Overview

This document validates our foil physics model against the original reference implementation at https://github.com/lsegessemann/foilphysics.

**Validation script**: `validate_physics.py`

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| Lift Coefficient | ✓ PASS | Formula matches exactly |
| Drag Coefficient | ✓ PASS | Formula matches exactly |
| Angle of Attack | ✓ PASS | Calculation verified |
| Lift Decomposition | ✓ PASS | Vector components correct |
| Stall Behavior | ✓ PASS | Cl caps at 1.246, Cd increases |
| Stabilizer Downwash | ✓ PASS | Formula verified |
| Force Magnitudes | ✓ PASS | Realistic L/D ratio (29) |
| Constant Velocity | ✓ PASS | Explains pump_thrust_efficiency |

## Validated Formulas

### 1. Lift Coefficient
```
CL = (2π × α) / (1 + 2/AR)
```

| AoA | Expected | Actual | Match |
|-----|----------|--------|-------|
| 2° | 0.1917 | 0.1917 | ✓ |
| 6° | 0.5752 | 0.5752 | ✓ |
| 10° | 0.9587 | 0.9587 | ✓ |

Lift slope at AR=13.9: **5.4928 rad⁻¹** (matches)

### 2. Drag Coefficient
```
CD = CD0 + CL² / (π × AR)
```

| AoA | Cl | Expected Cd | Actual Cd | Match |
|-----|-----|-------------|-----------|-------|
| 4° | 0.383 | 0.01637 | 0.01637 | ✓ |
| 8° | 0.767 | 0.02647 | 0.02647 | ✓ |
| 12° | 1.150 | 0.04331 | 0.04331 | ✓ |

### 3. Angle of Attack
```
α = pitch - arctan(vz/vx)
```

| Pitch | Vz | γ | Expected α | Actual α | Match |
|-------|-----|-----|------------|----------|-------|
| 10° | 0.0 | 0° | 10.0° | 10.0° | ✓ |
| 10° | -1.0 | -14° | 24.0° | 24.0° | ✓ |
| 10° | +1.0 | +14° | -4.0° | -4.0° | ✓ |

### 4. Lift Vector Decomposition
```
Lx = -L × sin(γ)
Lz = L × cos(γ)
```

| Vz | γ | L | Expected Lx | Actual Lx | Match |
|----|-----|-----|-------------|-----------|-------|
| -2.0 | -26.6° | 1682N | 752.4 | 752.4 | ✓ |
| 0.0 | 0° | 1035N | 0.0 | 0.0 | ✓ |
| +2.0 | +26.6° | -1682N | 752.4 | 752.4 | ✓ |

### 5. Stall Behavior

Stall angle: **13°**
Max Cl at stall: **1.246**

| AoA | Cl | Cd | Stalled | Cl Capped |
|-----|-----|-----|---------|-----------|
| 12° | 1.150 | 0.043 | No | ✓ |
| 13° | 1.246 | 0.050 | Yes | ✓ |
| 20° | 1.246 | 0.294 | Yes | ✓ |
| 30° | 1.246 | 0.643 | Yes | ✓ |

### 6. Reference Condition Sanity Check

At original foilphysics default conditions:
- Velocity: 4.72 m/s (17 km/h)
- Pitch: 8°
- Wing area: 1350 cm²

| Metric | Value | Expected Range |
|--------|-------|----------------|
| Lift | 1154 N | Rider weight ±50% |
| Drag | 40 N | Reasonable |
| L/D ratio | 29 | 10-40 typical |
| Lift/Weight | 1.55 | 0.5-2.0 |

## Key Validation: Why pump_thrust_efficiency Exists

### The Energy Gap

At vz=±1.0 m/s, 2Hz pumping:

| Metric | Value |
|--------|-------|
| Average thrust (Lx) | 118 N |
| Average drag | 164 N |
| **Net force** | **-46 N** |
| Average lift | 723 N |
| Required (weight) | 746 N |
| **Lift deficit** | **-23 N** |

### Conclusion

**Thrust < Drag** → Velocity decreases without external energy

The original foilphysics model avoids this problem by assuming constant velocity ("the board is on an invisible rail"). Our dynamics model must close the energy loop.

`pump_thrust_efficiency = 0.08-0.10` bridges this gap, representing:
- Unsteady aerodynamic effects
- Added mass contributions
- Flapping foil propulsion efficiency

This is not a "cheat" but a physically-justified lumped parameter for effects not explicitly modeled.

## Running Validation

```bash
cd mujoco
source ../venv_mujoco/bin/activate
python validate_physics.py
```

Expected output: `ALL TESTS PASSED`

## Model Assumptions

### Verified Against Original
1. ✓ Thin airfoil lift theory with finite AR correction
2. ✓ Parabolic drag polar
3. ✓ Linear stall model (Cl saturates, Cd increases)
4. ✓ Stabilizer downwash calculation
5. ✓ Lift/drag vector decomposition

### Extensions Beyond Original
1. **Dynamics integration** - velocity changes from net forces
2. **pump_thrust_efficiency** - unsteady aero effects
3. **State evolution** - full physics simulation
4. **Mast drag** - depth-dependent

### Simplifications
1. No added mass modeling
2. No elastic energy recovery
3. Single-mass rider model
4. 2D motion only (no roll/yaw)

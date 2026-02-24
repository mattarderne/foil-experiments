# Configuration Architecture

## Overview

All physics parameters and test configurations are centralized in `foil_config.py`. Test files should NEVER define inline CONFIG dicts.

## File Structure

```
mujoco/
├── foil_config.py          # CENTRAL CONFIG - import from here
├── validate_physics.py     # Uses VALIDATION_CONFIG
├── test_physics_refactored.py  # Example of proper usage
└── docs/
    └── config-architecture.md  # This file
```

## Configuration Hierarchy

```
SimConfig (complete simulation setup)
├── FoilGeometry (wing, stabilizer, drag)
├── RiderConfig (mass)
├── PumpConfig (thrust efficiency)
├── TerminationThresholds (z_min, z_max, vx_min)
└── InitialState (starting conditions)
```

## Predefined Configurations

### Foil Geometries

| Name | Area | Use Case |
|------|------|----------|
| `FOIL_BEGINNER` | 2200 cm² | Easy mode, very forgiving |
| `FOIL_TRAINING` | 2000 cm² | Standard RL training |
| `FOIL_INTERMEDIATE` | 1800 cm² | Moderate challenge |
| `FOIL_ADVANCED` | 1500 cm² | Requires good technique |
| `FOIL_REALISTIC` | 1350 cm² | Matches original foilphysics |

### Pump Efficiency Presets

| Name | Efficiency | Use Case |
|------|------------|----------|
| `PUMP_NO_ASSIST` | 0.00 | Validation only |
| `PUMP_REALISTIC` | 0.08 | Conservative estimate |
| `PUMP_TRAINING` | 0.10 | Recommended for RL |
| `PUMP_EASY` | 0.15 | Training wheels |

### Complete SimConfigs

| Name | Foil | Pump | Use Case |
|------|------|------|----------|
| `TEST_CONFIG` | training | training | Most tests |
| `VALIDATION_CONFIG` | realistic | no_assist | Physics validation |
| `EASY_CONFIG` | beginner | easy | Initial learning |
| `HARD_CONFIG` | advanced | realistic | Expert mode |

## Usage Examples

### Basic: Use a preset

```python
from foil_config import TEST_CONFIG
from foil_env.foil_physics import FoilPhysics

foil = FoilPhysics(TEST_CONFIG.to_physics_config())
```

### Customize foil only

```python
from foil_config import SimConfig, FOIL_ADVANCED, PUMP_TRAINING

config = SimConfig(
    foil=FOIL_ADVANCED,
    pump=PUMP_TRAINING,
)
```

### Fully custom

```python
from foil_config import SimConfig, FoilGeometry, PumpConfig, RiderConfig

custom_foil = FoilGeometry(
    name="my_foil",
    S=0.17,
    S_stab=0.034,
    AR=10.0,
)

config = SimConfig(
    foil=custom_foil,
    rider=RiderConfig(mass=80.0),
    pump=PumpConfig(pump_thrust_efficiency=0.12),
)
```

### Access individual values

```python
from foil_config import TEST_CONFIG

print(f"Wing area: {TEST_CONFIG.foil.S * 10000:.0f} cm²")
print(f"Total mass: {TEST_CONFIG.rider.total_mass} kg")
print(f"Weight: {TEST_CONFIG.rider.weight:.0f} N")
```

## Physics Constants

These are in `PhysicsConstants` and should NOT be changed:

```python
RHO_WATER = 1000.0          # Water density (kg/m³)
GRAVITY = 9.81              # Gravity (m/s²)
STALL_DRAG_BASE = 0.05      # Cd when stalled
STALL_DRAG_SLOPE = 2.0      # Cd increase per rad above stall
MOMENT_INERTIA_SCALE = 0.25 # I = m * scale
PUMP_VZ_THRESHOLD = 0.02    # Min vz for pump thrust
```

## Migration Guide

### OLD (don't do this):

```python
# BAD - hardcoded config
CONFIG = {
    'S': 0.20,
    'S_stab': 0.040,
    'Cd0': 0.008,
    'pump_thrust_efficiency': 0.18,
    'mass': 75,
    'board_mass': 6,
}
foil = FoilPhysics(CONFIG)
```

### NEW (do this):

```python
# GOOD - use central config
from foil_config import TEST_CONFIG
foil = FoilPhysics(TEST_CONFIG.to_physics_config())

# Or customize:
from foil_config import SimConfig, FOIL_TRAINING, PumpConfig
config = SimConfig(
    foil=FOIL_TRAINING,
    pump=PumpConfig(pump_thrust_efficiency=0.18),
)
foil = FoilPhysics(config.to_physics_config())
```

## Benefits

1. **Single source of truth** - Change once, affects all tests
2. **Documented presets** - Know what each config means
3. **Type safety** - Dataclasses catch errors
4. **Traceability** - Git tracks config changes
5. **Reproducibility** - Same config = same results

## Adding New Presets

1. Add to `foil_config.py`
2. Document in this file
3. Use in tests via import

```python
# In foil_config.py
FOIL_EXPERIMENTAL = FoilGeometry(
    name="experimental",
    S=0.19,
    S_stab=0.038,
    Cd0=0.009,
)

# In test file
from foil_config import FOIL_EXPERIMENTAL
```

# Pump Foil RL Project

## Overview

Training an RL agent to pump foil using MuJoCo physics simulation.

## Project Structure

```
foilpump/
├── mujoco/                    # MAIN - MuJoCo-based environment
│   ├── foil_env_mujoco.py     # Gymnasium environment (MuJoCo dynamics)
│   ├── foil_env_mjx.py        # JAX/MJX for GPU training
│   ├── foil_env_rigid.py      # Simplified rigid rider
│   ├── train_mjx.py           # Brax PPO training
│   ├── create_video_mujoco.py # MuJoCo 3D video renderer
│   ├── rider.xml              # MJCF articulated body model
│   ├── rider_rigid.xml        # MJCF rigid body model
│   ├── PROGRESS.md            # Migration progress tracking
│   └── requirements.txt       # MuJoCo dependencies
├── foil_env/                  # Shared physics (used by MuJoCo env)
│   ├── foil_physics.py        # Hydrodynamic force calculations
│   └── foil_config.py         # Foil geometry configs
├── venv_mujoco/               # Python 3.11/3.12 venv for MuJoCo
└── archive/                   # Legacy code (reference only)
```

## Quick Start

```bash
# Setup (requires Python 3.11 or 3.12 - MuJoCo won't build on 3.14)
cd mujoco
python3.12 -m venv ../venv_mujoco
source ../venv_mujoco/bin/activate
pip install -r requirements.txt

# Test environment
python foil_env_mujoco.py

# Train (CPU - slow)
python train_mjx.py -t 100000 -n 64

# Create video
python create_video_mujoco.py -d 10 -o demo.mp4
```

## Foil Configuration

Use `foil_env/foil_config.py` for realistic foil geometry:

```python
from foil_env.foil_config import PUMP_FOIL_BEGINNER

# PUMP_FOIL_BEGINNER:
#   Front wing: 2200cm² @ +1.0° (AR=8.0)
#   Stabilizer: 200cm² @ -2.0° (9% ratio)
#   Stable at 4.5+ m/s, requires pumping below
```

### Stabilizer Sizing Rules

| Wing Area | Stab Area | Ratio |
|-----------|-----------|-------|
| 800cm²    | 150cm²    | 18.8% |
| 1200cm²   | 165cm²    | 13.8% |
| 1800cm²   | 180cm²    | 10.0% |
| 2200cm²   | 200cm²    | 9.1%  |

## Physics

The MuJoCo environment uses:
- **MuJoCo**: Rigid body dynamics, contact, constraints
- **foil_physics.py**: Hydrodynamic forces (lift, drag, stabilizer moment)

Key physics parameters:
- `front_wing_angle`: +1.0° incidence (shim)
- `stab_angle`: -2.0° (toed down for pitch stability)
- `Cd0`: 0.008 (modern low-drag pump foil)
- `Cd_mast`: 0.06 (streamlined carbon mast)

Stability characteristics:
```
Velocity    Status
3.5 m/s     unstable (needs pumping)
4.0 m/s     marginally unstable
4.5 m/s     STABLE (can glide)
5.0+ m/s    STABLE
```

## Current Status

See `mujoco/PROGRESS.md` for detailed progress.

**Completed:**
- Phase 2.1: MJCF body model
- Phase 2.2: Hybrid physics loop
- Phase 2.3: MuJoCo video renderer
- Phase 2.4: MJX/Brax training

**In Progress:**
- Phase 3: Validation with realistic physics
  - [x] Fix stabilizer sizing (200cm² not 440cm²)
  - [x] Add front_wing_angle to physics
  - [ ] Train with realistic physics
  - [ ] Achieve sustained flight through learned pumping

## Do NOT

- Use files in `archive/` - they're legacy reference only
- Use Python 3.14 - MuJoCo won't build
- Use the old matplotlib visualizer - use MuJoCo renderer
- Bypass physics with artificial stabilization

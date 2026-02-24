# MuJoCo Pump Foil (Exploratory)

An attempt to simulate pump foiling using MuJoCo for the rider body physics and Python for the foil hydrodynamics.

**Status: Exploratory.** The hybrid environment runs and achieves sustained flight with scripted sinusoidal pumping, but RL training was not completed. The Python RL approach (see `../python-rl/`) proved more productive.

## What We Tried

### The Hybrid Approach

MuJoCo handles the articulated rider body (hips, knees, waist joints) while Python computes foil hydrodynamics. The board position is set directly from the foil physics simulation rather than applying forces, which avoids inverted pendulum instability.

```
MuJoCo (rider body) ──→ body motion ──→ Python (foil physics) ──→ board position
         ↑                                                              │
         └──────────────────────────────────────────────────────────────┘
```

### What Worked

- **Hybrid architecture**: MuJoCo body + Python foil physics is viable
- **Scripted pumping**: 2Hz sinusoidal joint commands achieve 60+ second flights
- **Physics validation**: Lift, drag, and stall behavior match reference implementations
- **Multiple rider models**: From simple pogo stick to full articulated body

### What Was Difficult

- **Force application**: Applying foil forces directly to a MuJoCo floating base caused instability
- **Joint coordination**: Getting MuJoCo joints to produce meaningful pump forces required high force amplification (~60x)
- **Training infrastructure**: MuJoCo environments are heavier to set up for RL than pure Python
- **Debugging**: Hard to tell if problems were physics, MuJoCo config, or RL hyperparameters

## Key Files

| File | Description |
|------|-------------|
| `hybrid_env.py` | Main environment: MuJoCo body + Python foil physics |
| `foil_config.py` | Central configuration with physics presets |
| `foil_physics.py` | Hydrodynamic force calculations |
| `rider_hybrid.xml` | MJCF model for hybrid environment (5-DOF) |
| `rider_independent_legs.xml` | Independent leg control model |

## Physics Documentation

Detailed writeups of the physics model and validation:

- [pump-physics-explained.md](docs/pump-physics-explained.md) - Why pump thrust efficiency exists
- [config-architecture.md](docs/config-architecture.md) - Configuration system design
- [model-validation.md](docs/model-validation.md) - Validation against reference foil physics
- [foilphysics-comparison.md](docs/foilphysics-comparison.md) - Comparison with lsegessemann/foilphysics

## Demo Videos

| Video | Description |
|-------|-------------|
| `videos/h1_2hz_pumping.mp4` | Humanoid pumping at 2Hz |
| `videos/simple_pump_v7.mp4` | Simple pump test |
| `videos/mujoco_pump_test.mp4` | Basic pump test |

## Setup

Requires Python 3.11 or 3.12 (MuJoCo doesn't support 3.14).

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Key parameters in `foil_config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Foil area | 2000 cm² | Training foil |
| Pump efficiency | 0.26 | Pump-to-thrust conversion |
| Force amplification | 60x | Body motion scaling |
| Velocity cap | 5.2 m/s | Maximum speed |

Foil presets: `FOIL_BEGINNER` (2200cm²), `FOIL_TRAINING` (2000cm²), `FOIL_REALISTIC` (1350cm²).

## Lessons Learned

1. **Direct position control > force application** for floating bases with attached mass
2. **Start simple**: the Python RL approach with matplotlib visualization was faster to iterate on
3. **MuJoCo adds value** when you need realistic contact physics or 3D rendering - for this problem, 2D was sufficient
4. **Validate physics independently** before coupling with RL training

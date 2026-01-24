# Pump Foil RL Project

## Overview

Training an RL agent to pump foil - maintaining flight through coordinated body movements (legs, arms, waist) that generate vertical forces and pitch torque on a hydrofoil.

## Project Structure

```
foilpump/
├── train.py           # Standard training script
├── create_video.py    # Standard video creation
├── foil_env/          # Core environment
│   ├── foil_physics.py           # Hydrodynamic physics
│   ├── body_model.py             # Rider body mechanics
│   ├── foil_visualizer.py        # Reusable visualization
│   ├── pump_foil_env_curriculum.py  # Main RL environment
│   ├── config.py                 # Central configuration (FoilConfig, LimbConfig, TrainingConfig)
│   ├── checkpoint_utils.py       # Save/load checkpoints with config
│   ├── metrics.py                # Oscillation detection & analysis
│   └── wrappers.py               # Gym wrappers (arm patterns, etc.)
├── checkpoints/       # Saved models + config files
├── requirements.txt
├── venv/
└── archive/           # Old experiments (reference only)
```

## Configuration System

All configurations are now centralized in `foil_env/config.py`:

```python
from foil_env.config import TrainingConfig, FoilConfig, LimbConfig, TRAINING_FOIL, DEFAULT_LIMBS

# Predefined configs
TRAINING_FOIL  # Larger stabilizer, more stable for learning
BIG_FOIL       # Smaller, faster, less stable

# Create custom config
config = TrainingConfig(
    foil=FoilConfig(S=0.18, S_stab=0.035, AR=8),
    limbs=LimbConfig(arm_start_position=np.pi/2, arm_centering_spring=2.0),
)
```

### Key Limb Parameters

- `arm_start_position`: Arms start at π/2 (DOWN), not 0 (forward)
- `arm_centering_spring`: Spring constant pulling arms back to neutral (default: 2.0)
- `max_arm_swing`: Maximum arm swing from center (default: 2.0 rad)

## Standard Training Procedure

### 1. Training Requirements

Every training run MUST:
- Save checkpoints at **1%, 25%, 50%, 75%, 100%** of training
- Save config alongside each checkpoint (`model_Npct_config.json`)
- Output to a named checkpoint directory

### 2. Running Training

```bash
# Local (slower, ~500k steps)
python train.py --timesteps 500000 --output checkpoints/experiment_name

# Modal GPU (faster, recommended for >500k steps)
python train.py --modal --timesteps 1000000 --output checkpoints/experiment_name
```

### 3. After Training: Create Video

Every training run should produce an evolution video:

```bash
python create_video.py --checkpoint-dir checkpoints/experiment_name -o experiment_evolution.mp4
```

The video will automatically load configs from checkpoints for reproducibility.

## Checkpoint Format

New checkpoints include config files:
```
checkpoints/experiment_name/
├── model_1pct.zip
├── model_1pct_config.json    # Training config
├── model_25pct.zip
├── model_25pct_config.json
├── model_50pct.zip
├── model_50pct_config.json
├── model_75pct.zip
├── model_75pct_config.json
├── model_100pct.zip
└── model_100pct_config.json
```

Loading checkpoints:
```python
from foil_env.checkpoint_utils import load_checkpoint, create_env_from_config

model, config = load_checkpoint("checkpoints/run/model_100pct.zip")
env = create_env_from_config(config)  # Uses same settings as training
```

## Metrics System

New metrics detect actual oscillation (not just min-max drift):

```python
from foil_env.metrics import analyze_episode, format_metrics

metrics = analyze_episode(frames, dt=0.01)
print(format_metrics(metrics))

# Key metrics:
# - leg_frequency: FFT-detected pumping frequency
# - leg_is_oscillating: True if actually pumping (not just drifting)
# - arm_is_oscillating: True if arms oscillate (not drifted to extreme)
# - velocity_correlation: Coordination between leg and arm movements
```

## Objective & Constraints

### What We're Trying to Achieve
- Agent learns to **pump** (rhythmic leg motion at ~2Hz, full range)
- Maintains flight indefinitely (target: **60s+** with corrected physics)
- Maintains velocity (target: ~4.5 m/s)
- Arms should oscillate around neutral, not drift

### Physical Constraints
- **Ceiling**: Foil breaches at z ≥ 0.2m (crash)
- **Floor**: Board touchdown at z ≤ -0.5m (crash)
- **Energy budget**: 50000J for long flights
- **Stall velocity**: < 1.5 m/s (crash)

### Success Metrics (per episode)
| Metric | Poor | Good | Excellent |
|--------|------|------|-----------|
| Flight duration | <10s | 30s | **60s (timeout)** |
| Leg oscillation | ✗ not pumping | ✓ pumping | ✓ 2Hz, 90%+ range |
| Arm behavior | drifted to extreme | centered | oscillating in sync |

## Video Format

Standard 4-panel comparison showing training evolution:
- **Top row**: 4 foil visualizations (1%, 25%, 75%, 100% checkpoints)
- **Bottom row**: 5 time series overlaid (altitude, velocity, leg, arm, waist)
- Shows how pumping behavior emerges through training

## Foil Configuration

### Physics Parameters (Updated for Sustained Flight)

The physics model was updated to allow sustained flight through pumping:

```python
FoilConfig(
    S=0.22,           # Wing area (m²) - 2200cm² big foil
    S_stab=0.044,     # Stabilizer area (m²)
    stab_angle=-4.0,  # Stabilizer angle (deg)
    AR=8,             # Aspect ratio
    Cd0=0.008,        # Parasitic drag (reduced from 0.013 for modern foils)
    Cd_mast=0.06,     # Mast drag (reduced from 0.12 for carbon masts)
    pump_thrust_efficiency=0.20,  # Pump thrust conversion (key for sustained flight)
)
```

**Key physics changes:**
- **Cd0**: 0.008 (was 0.013) - modern pump foils are very clean
- **Cd_mast**: 0.06 (was 0.12) - streamlined carbon masts
- **pump_thrust_efficiency**: 0.20 - models how pumping converts to forward thrust

With these parameters, **60+ second sustained flights are possible** with proper pumping technique.

### Recommended Training Script

```bash
# For sustained flight experiments
python train_sustained.py -t 1000000 -o checkpoints/sustained_experiment
```

## Key Insights

1. **Curriculum learning works**: Forcing pumping initially, then releasing control
2. **Frequency reward helps**: Rewarding phase alignment with 2Hz reference
3. **Deweighting matters**: Arm and torso acceleration creates vertical reaction forces
4. **Arm centering spring**: Prevents arms from drifting to extreme positions
5. **Config tracking**: Save config with checkpoints for reproducibility

See `docs/pump-physics.md` for detailed physics analysis and energy tradeoffs.

## CRITICAL: Always Fine-tune from Baseline

**NEVER train from scratch unless explicitly requested.** Always fine-tune from the baseline checkpoint:
- Baseline: `checkpoints/ent005_2M/model_100pct.zip` (5.0s, 2.3Hz pump, r=+0.44)
- Training from scratch loses learned pumping behavior
- Fine-tuning preserves what works and adds new capabilities

## Evaluation Requirements

When evaluating models, ALWAYS report:
1. **Termination reason**: `foil_breached` vs `board_touchdown` vs `stalled_velocity` vs `energy_exhausted`
2. **Final state**: altitude (z), velocity (vx), energy consumed
3. **Foil size**: Which foil config was used (1200cm², 1800cm², 2200cm², etc.)

Example evaluation output:
```
Ep 1: 6.5s | board_touchdown | z=-0.51m | vx=3.0m/s | energy=2329J
Ep 2: 1.3s | foil_breached   | z=0.20m  | vx=4.0m/s | energy=448J
```

**Why this matters**: A model that breaches (too much lift) has a different problem than one that touches down (not enough lift). Short flights from breaching may indicate the foil is too small or pumping is too aggressive - NOT that the technique doesn't work.

## Foil Sizes

| Name | Wing Area | Use Case |
|------|-----------|----------|
| MOCAP_FOIL | 1200 cm² | Matches real mocap data, smaller/harder |
| TRAINING_FOIL | 1800 cm² | Default for learning, more forgiving |
| BIG_FOIL | 2200 cm² | Easy mode, lots of lift margin |

Smaller foils are harder to control - they breach easily with aggressive pumping.

## Do NOT

- Create analysis images unless explicitly requested
- Use old environments from archive/
- Skip checkpoint saving during training
- Skip video creation after training
- Add new features without updating this doc
- **Train from scratch** - always fine-tune from baseline unless explicitly told otherwise
- Evaluate models with different config than they were trained with
- **Interpret short flights without checking termination reason** - breach ≠ can't pump

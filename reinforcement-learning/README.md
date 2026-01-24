# Pump Foil RL System

Reinforcement learning agent that learns to pump foil - maintaining flight through coordinated body movements.

## Status: Archived (Pre-MuJoCo Migration)

This is the custom physics + matplotlib visualization system. See `MIGRATION.md` for the plan to migrate to MuJoCo.

## Results

- **60+ second sustained flights** with proper pumping technique
- **2.3Hz pump frequency** matching real-world biomechanics
- **11% smoother motion** with style rewards (see `style_comparison.mp4`)

## Key Files

```
reinforcement-learning/
├── foil_env/                    # Core environment
│   ├── foil_physics.py          # Hydrodynamic physics (lift, drag, pump thrust)
│   ├── body_model.py            # Rider body mechanics
│   ├── pump_foil_env_curriculum.py  # Gym environment
│   └── config.py                # Foil & training configuration
├── train.py                     # Training script
├── create_video.py              # Visualization
├── checkpoints/                 # Trained models
│   ├── foil_1000cm2/            # Best performer (1000cm² foil)
│   ├── stylized_ft/             # Smooth motion variant
│   └── ent005_2M/               # Baseline reference
└── *.mp4                        # Demo videos
```

## Videos

| Video | Description |
|-------|-------------|
| `sustained_60s.mp4` | 60 second sustained flight |
| `style_comparison.mp4` | Baseline vs stylized motion |
| `foil_size_comparison.mp4` | Different foil sizes |

## Quick Start

```bash
cd reinforcement-learning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Evaluate best model
python train.py --eval checkpoints/foil_1000cm2/model_100pct.zip

# Create video
python create_video.py --checkpoint-dir checkpoints/foil_1000cm2 -o output.mp4
```

## Physics

Custom hydrodynamic simulation:
- Wing lift/drag from angle of attack
- Stabilizer for pitch stability
- Pump thrust: vertical motion → forward propulsion
- Rider body forces affect pitch and altitude

See `foil_env/foil_physics.py` for implementation.

## Next Steps

Migrating to MuJoCo for:
- Standard MJCF body model
- Built-in 3D rendering
- MJX/Brax for GPU-accelerated training

See `MIGRATION.md` for details.

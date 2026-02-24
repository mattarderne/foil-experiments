# Python RL Pump Foil

Reinforcement learning agent that learns to pump foil using custom Python physics and Stable Baselines3 PPO.

**This approach worked.** The agent learns sustained 60+ second flights with realistic 2.3Hz pumping.

## Results

- **60+ second sustained flights** with proper pumping technique
- **2.3Hz pump frequency** matching real-world biomechanics
- Coordinated leg, arm, and waist movements
- Trained model included: `checkpoints/baseline_2M/model_100pct.zip`

### Demo Videos

| Video | Description |
|-------|-------------|
| `videos/sustained_60s.mp4` | 60-second sustained flight |
| `videos/foil_size_comparison.mp4` | Agent on different foil sizes |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Train a new model (saves checkpoints at 1%, 25%, 50%, 75%, 100%)
python train.py --timesteps 500000 --output checkpoints/my_run

# Or fine-tune from baseline
python train.py --timesteps 500000 --output checkpoints/finetuned \
    --checkpoint checkpoints/baseline_2M/model_100pct.zip
```

## Create Evolution Video

After training, create a 4-panel video showing how the agent's behavior evolves:

```bash
python create_video.py --checkpoint-dir checkpoints/my_run -o evolution.mp4
```

Top row shows foil visualization at each checkpoint stage. Bottom row shows time series (altitude, velocity, leg angle, arm angle, waist).

## How It Works

### Physics Model (`foil_env/foil_physics.py`)

Custom hydrodynamic simulation:
- **Wing lift/drag** from angle of attack (thin airfoil theory)
- **Stabilizer** for pitch stability
- **Pump thrust**: vertical body motion converts to forward propulsion
- **Rider body forces** affect pitch and altitude

### RL Environment (`foil_env/pump_foil_env_curriculum.py`)

Gymnasium environment with curriculum learning:
- **Action space**: 5 continuous actions (left/right hip, left/right knee, waist)
- **Observation**: board state (z, vx, vz, pitch, omega) + joint positions/velocities
- **Reward**: altitude maintenance + velocity + pumping frequency + energy efficiency
- **Curriculum**: starts with forced pumping, gradually releases control

### Key Insights

1. **Curriculum learning works**: forcing pumping initially, then releasing control
2. **Frequency reward helps**: rewarding phase alignment with 2Hz reference
3. **Deweighting matters**: arm/torso acceleration creates vertical reaction forces
4. **Pump thrust efficiency** (~0.20) is the key parameter for sustained flight

## Configuration

Foil physics parameters in `foil_env/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Wing area | 2200 cm² | Large training foil |
| Stabilizer area | 440 cm² | For pitch stability |
| Cd0 | 0.008 | Parasitic drag (low for modern foils) |
| Pump efficiency | 0.20 | Vertical motion → forward thrust |

Three foil presets: `TRAINING_FOIL` (2200cm², easy), `BIG_FOIL` (1800cm²), `MOCAP_FOIL` (1200cm², hard).

## File Structure

```
python-rl/
├── train.py              # Training script with checkpoint saving
├── create_video.py       # 4-panel evolution video from checkpoints
├── requirements.txt
├── foil_env/
│   ├── __init__.py
│   ├── foil_physics.py           # Hydrodynamic physics
│   ├── body_model.py             # Rider body mechanics
│   ├── foil_visualizer.py        # Matplotlib visualization
│   ├── pump_foil_env_curriculum.py  # Main Gymnasium environment
│   ├── config.py                 # Central configuration
│   ├── checkpoint_utils.py       # Save/load with config
│   ├── metrics.py                # Oscillation detection
│   └── wrappers.py               # Gym wrappers
├── checkpoints/
│   └── baseline_2M/
│       └── model_100pct.zip      # Trained model (2M steps)
└── videos/
    ├── sustained_60s.mp4
    └── foil_size_comparison.mp4
```

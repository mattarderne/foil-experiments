# Pump Foil RL Environment

A Gymnasium-based reinforcement learning environment that simulates pump foiling physics. An RL agent learns to control a human rider to maintain flight and generate forward thrust through pumping motions.

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
foilpump/
├── foil_env/
│   ├── __init__.py           # Package exports
│   ├── foil_physics.py       # Physics engine (ported from foilphysics)
│   ├── pump_foil_env.py      # Gymnasium environment
│   └── renderer.py           # Visualization
├── train.py                  # Training script (PPO/SAC)
├── evaluate.py               # Evaluation and visualization
├── requirements.txt
└── README.md
```

## Quick Start

### Test the Environment

```python
from foil_env import PumpFoilEnv

env = PumpFoilEnv()
obs, info = env.reset()

# Run a few steps
for _ in range(100):
    action = env.action_space.sample()  # Random actions
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Train an Agent

```bash
# Train with PPO (default)
python train.py --timesteps 1000000

# Train with SAC
python train.py --algorithm SAC --timesteps 1000000

# Train with more parallel environments
python train.py --n-envs 16 --timesteps 2000000
```

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluate a Trained Model

```bash
# Evaluate with random policy (baseline)
python evaluate.py --episodes 5

# Evaluate a trained model
python evaluate.py --model models/pump_foil_ppo_*/best_model --episodes 10

# Evaluate with visualization
python evaluate.py --model models/pump_foil_ppo_*/best_model --render
```

## Environment Details

### Observation Space (11 dimensions)
| Index | Name | Description |
|-------|------|-------------|
| 0 | z | Vertical position (m) - positive = above water surface level |
| 1 | vx | Forward velocity (m/s) |
| 2 | vz | Vertical velocity (m/s) |
| 3 | theta | Pitch angle (rad) |
| 4 | omega | Pitch angular velocity (rad/s) |
| 5 | sin(phase) | Sine of internal phase (for cyclic reference) |
| 6 | cos(phase) | Cosine of internal phase |
| 7 | altitude_error | z - target_altitude |
| 8 | velocity_error | vx - target_velocity |
| 9 | energy_rate | Normalized power expenditure [0, 1] |
| 10 | is_stalled | 1.0 if wing is stalled, 0.0 otherwise |

### Action Space (2 dimensions)
| Index | Name | Range | Physical Meaning |
|-------|------|-------|-----------------|
| 0 | leg_force | [-1, 1] | Leg extension force (±800 N) |
| 1 | pitch_torque | [-1, 1] | Pitch torque (±50 Nm) |

### Reward Function
- **Altitude maintenance**: Penalize deviation from target altitude (15 cm below surface)
- **Forward velocity**: Reward for maintaining/increasing speed
- **Energy efficiency**: Penalize power expenditure
- **Stall penalty**: Large penalty when wing stalls
- **Crash penalty**: Terminal penalty for crashing

### Episode Termination
- Sunk too deep (z < -0.5 m)
- Breached surface too high (z > 0.3 m)
- Velocity too low (vx < 1.0 m/s)
- Excessive pitch angle (|theta| > 45°)
- Maximum steps reached (3000 steps = 30 seconds)

## Physics Model

The physics model is ported from the [foilphysics](./foilphysics/) JavaScript simulation and includes:

- **Front Wing**: Finite wing lift/drag with stall model
- **Stabilizer**: With downwash correction from front wing
- **Mast Drag**: Proportional to submerged depth
- **Surface Breach**: Zero hydrodynamic forces when wing exits water

Key parameters (from CONFIG):
- Wing area: 0.135 m² (1350 cm²)
- Aspect ratio: 13.9
- Stall angle: 13°
- Target speed: ~16 km/h (4.5 m/s)

## Training Tips

1. **Start with PPO**: PPO is more stable and easier to tune than SAC for this environment.

2. **Use RK4 integration** for more accurate physics (at slight performance cost):
   ```bash
   python train.py --use-rk4
   ```

3. **Monitor in TensorBoard**: Watch for:
   - `ep_rew_mean`: Should increase over training
   - `ep_len_mean`: Longer episodes = staying airborne longer

4. **Reward shaping**: Modify `_compute_reward()` in `pump_foil_env.py` to adjust:
   - Altitude vs. thrust trade-off
   - Energy efficiency importance
   - Exploration vs. exploitation

## Comparing to foilphysics

The web-based [foilphysics](./foilphysics/index.html) uses prescribed sinusoidal motion. This RL environment uses force-based dynamics where the agent must learn to generate appropriate pumping patterns.

To validate physics:
1. Run foilphysics with specific settings
2. Record lift/drag/power outputs
3. Compare to `foil_physics.py` calculations with same kinematics

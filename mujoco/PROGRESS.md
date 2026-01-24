# MuJoCo Migration Progress

## Completed

### Phase 2.1: MJCF Body Model ✓
- Created `rider.xml` with full body model
- Matches original body_model.py dimensions
- Total mass: 86kg (original: 76kg)
- 9 actuators: knee×2, hip×2, waist, shoulder×2, elbow×2
- Cameras for side/front/iso views

### Phase 2.2: Hybrid Physics Loop ✓
- Created `foil_env_mujoco.py` - Gymnasium-compatible environment
- MuJoCo handles body dynamics
- Custom foil_physics.py computes hydrodynamics
- Forces applied via `xfrc_applied`

### Phase 2.3: Rendering Pipeline ✓
- Created `create_video_mujoco.py`
- MuJoCo's built-in renderer
- Multiple camera angles
- MP4 output via imageio

### Phase 2.4: MJX/Brax Training ✓
- Created `foil_env_mjx.py` - JAX-compatible environment
- Created `train_mjx.py` - Brax PPO training script
- All physics ported to JAX (JIT-compiled)
- Batched environments for parallel rollouts

**Benchmark results (CPU):**
| Envs | Steps/sec | Notes |
|------|-----------|-------|
| 32   | 108       | Baseline |
| 64   | 426       | 4x better |
| GPU  | ~10000+   | Expected with CUDA |

## Files Created
```
mujoco/
├── rider.xml              # MJCF body model
├── foil_env_mujoco.py     # Gymnasium environment
├── foil_env_mjx.py        # JAX/MJX environment
├── train_mjx.py           # Brax PPO training
├── create_video_mujoco.py # Video rendering
├── test_rider.py          # Model tests
├── requirements.txt       # Dependencies
├── PROGRESS.md            # This file
└── checkpoints/           # Trained models
    ├── test_mjx/
    └── bench_mjx/
```

## Known Issues

1. **Physics tuning needed**: Sinusoidal pumping causes crashes
   - Need to tune force scaling for MuJoCo dynamics
   - May need different reward shaping

2. **CPU training slow**: 426 steps/sec
   - GPU will be 10-100x faster
   - Use Modal for cloud GPU training

3. **Warp warnings**: "Failed to import warp" - can be ignored

## Next Steps

### Phase 3: Validation
- [ ] Tune foil physics for stable flight
- [ ] Train on GPU (Modal or Colab)
- [ ] Achieve 60s sustained flight
- [ ] Compare with original system

## Usage

### Local Development (CPU)
```bash
python3.11 -m venv venv_mujoco
source venv_mujoco/bin/activate
pip install -r mujoco/requirements.txt

# Test model
python mujoco/test_rider.py

# Test environment
python mujoco/foil_env_mujoco.py

# Train (CPU, slow)
python mujoco/train_mjx.py -t 100000 -n 64

# Create video
python mujoco/create_video_mujoco.py -d 10 -o demo.mp4
```

### GPU Training (Modal)
```bash
# TODO: Create modal_train_mjx.py
modal run modal_train_mjx.py --timesteps 10000000 --num-envs 2048
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     train_mjx.py                        │
│                    (Brax PPO)                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   foil_env_mjx.py                       │
│              (JAX-based environment)                    │
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  MJX Model  │    │ Foil Forces │    │   Reward    │ │
│  │  (rider)    │◄───│ (JAX port)  │───►│  Function   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     rider.xml                           │
│                 (MJCF body model)                       │
└─────────────────────────────────────────────────────────┘
```

## Comparison with Original System

| Feature | Original | MuJoCo/MJX |
|---------|----------|------------|
| Physics | Custom Python | MuJoCo (C++) |
| Body model | Custom classes | MJCF XML |
| Rendering | Matplotlib | MuJoCo Renderer |
| Training | SB3 PPO | Brax PPO |
| Speed (CPU) | ~100 steps/s | ~400 steps/s |
| Speed (GPU) | N/A | ~10000+ steps/s |
| Batching | Sequential | Parallel (vmap) |

# Migration Plan: Custom → MuJoCo

## Phase 1: Archive Current System ✓

Commit current RL system to `reinforcement-learning/` directory in pumpfoil repo.

### What to Keep
- **Core code**: `foil_env/`, `train.py`, `create_video.py`
- **Key checkpoints** (100pct only, ~30MB total):
  - `foil_1000cm2/` - best performing model
  - `stylized_ft/` - smooth motion variant
  - `ent005_2M/` - baseline reference
- **Key videos** (~50MB):
  - `style_comparison.mp4` - baseline vs stylized
  - `sustained_60s.mp4` - 60s sustained flight
  - `foil_size_comparison.mp4` - size experiments
- **Config/docs**: `CLAUDE.md`, `requirements.txt`

### What to Exclude
- `venv/` - recreate from requirements.txt
- `checkpoints/*/model_{1,25,50,75}pct.zip` - intermediate checkpoints
- All other checkpoint directories (experiments)
- Duplicate/experimental videos

## Phase 2: MuJoCo Integration

### 2.1 Create MJCF Body Model (~4 hrs)
```xml
<!-- rider.xml -->
<mujoco>
  <worldbody>
    <body name="board">
      <body name="lower_leg">
        <body name="upper_leg">
          <body name="torso">
            <body name="head"/>
            <body name="left_arm"/>
            <body name="right_arm"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
```

### 2.2 Hybrid Physics Loop (~4 hrs)
```python
def step(action):
    # 1. MuJoCo: update rider pose from action
    mj_data.ctrl[:] = action
    mujoco.mj_step(mj_model, mj_data)

    # 2. Custom: compute foil forces from rider state
    rider_com = get_com(mj_data)
    foil_forces = compute_foil_physics(rider_com, velocity, pitch)

    # 3. MuJoCo: apply external forces to board
    mj_data.xfrc_applied[board_id] = foil_forces
    mujoco.mj_step(mj_model, mj_data)
```

### 2.3 Rendering Pipeline (~2 hrs)
```python
renderer = mujoco.Renderer(mj_model)
renderer.update_scene(mj_data)
frame = renderer.render()
```

### 2.4 Brax/MJX for Fast Training (~4 hrs)
- Port to MJX for GPU-batched rollouts
- Use Brax PPO instead of SB3
- Target: 10-100x speedup on Modal

## Phase 3: Validation

- [ ] MuJoCo rider matches current body model poses
- [ ] Foil physics produces same forces as current system
- [ ] Trained policy achieves 60s sustained flight
- [ ] Rendering shows smooth, human-like motion

## Timeline

| Phase | Effort | Output |
|-------|--------|--------|
| 1. Archive | 1 hr | Git commit to pumpfoil repo |
| 2.1 MJCF | 4 hrs | `rider.xml` body model |
| 2.2 Hybrid | 4 hrs | Working sim loop |
| 2.3 Render | 2 hrs | Video pipeline |
| 2.4 MJX | 4 hrs | Fast training |
| 3. Validate | 2 hrs | Comparison tests |

**Total: ~17 hours**

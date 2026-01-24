# Plan: 4-Limb Pump Foil Model + Modal GPU Training

## Current State (1-DOF leg + pitch)
- Agent learns 2.08 Hz pumping, -0.06 m/s velocity loss over 18s
- Energy: 248W mean, 849W peak bursts
- Limited by energy budget (4.5kJ)

## Phase 1: 4-Limb Physics Model

### New State Variables
```
Current (6 DOF foil + 2 control):
  - Foil: x, z, vx, vz, theta, omega
  - Control: leg_force (vertical), pitch_torque

New (6 DOF foil + 4 limb positions):
  - Foil: x, z, vx, vz, theta, omega
  - Left leg: position, velocity
  - Right leg: position, velocity
  - Left arm: position, velocity
  - Right arm: position, velocity
```

### Action Space (4 dimensions)
```python
action_space = Box(low=-1, high=1, shape=(4,))
# action[0]: Left leg velocity command
# action[1]: Right leg velocity command
# action[2]: Left arm velocity command
# action[3]: Right arm velocity command
```

### Physics Changes
```python
# Leg forces (asymmetric pumping possible)
left_leg_force = -RIDER_MASS/2 * left_leg_accel
right_leg_force = -RIDER_MASS/2 * right_leg_accel
total_vertical_force = left_leg_force + right_leg_force

# Pitch moment from leg asymmetry (feet ~30cm apart)
STANCE_WIDTH = 0.30  # m
leg_pitch_moment = (right_leg_force - left_leg_force) * STANCE_WIDTH / 2

# Arm forces (smaller mass, pitch/roll control)
ARM_MASS = 5.0  # kg per arm
left_arm_force = -ARM_MASS * left_arm_accel
right_arm_force = -ARM_MASS * right_arm_accel

# Arms primarily affect pitch (forward/back swing)
ARM_MOMENT_ARM = 0.5  # m from CoM
arm_pitch_moment = (left_arm_force + right_arm_force) * ARM_MOMENT_ARM
```

### Observation Space (17 dimensions)
```python
[z, vx, vz, theta, omega,                    # 5: foil state
 left_leg_pos, left_leg_vel,                 # 2: left leg
 right_leg_pos, right_leg_vel,               # 2: right leg
 left_arm_pos, left_arm_vel,                 # 2: left arm
 right_arm_pos, right_arm_vel,               # 2: right arm
 sin(phase), cos(phase),                     # 2: phase hint
 altitude_error, velocity_error]             # 2: errors
```

## Phase 2: Modal GPU Setup

### modal_train.py
```python
import modal

app = modal.App("pump-foil-rl")

image = modal.Image.debian_slim().pip_install(
    "gymnasium",
    "stable-baselines3",
    "torch",
    "numpy",
    "tensorboard"
)

@app.function(
    image=image,
    gpu="A10G",  # or "T4" for cheaper
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb")]  # optional
)
def train_4limb(total_timesteps: int = 2_000_000):
    from foil_env import PumpFoilEnv4Limb
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        return PumpFoilEnv4Limb()

    # 8 parallel envs on GPU
    env = SubprocVecEnv([make_env for _ in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        device="cuda",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("/tmp/pump_foil_4limb")

    # Return model bytes
    with open("/tmp/pump_foil_4limb.zip", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    model_bytes = train_4limb.remote(total_timesteps=2_000_000)
    with open("checkpoints/model_4limb_modal.zip", "wb") as f:
        f.write(model_bytes)
    print("Saved: checkpoints/model_4limb_modal.zip")
```

### Run on Modal
```bash
modal run modal_train.py
```

## Phase 3: Implementation Steps

1. **Create `foil_env/pump_foil_env_4limb.py`**
   - Copy from pump_foil_env.py
   - Add 4 limb state variables
   - Update observation/action spaces
   - Modify physics step for asymmetric forces

2. **Test locally first**
   - Quick training run (100k steps) to verify
   - Check that limbs move independently
   - Verify pitch control from leg asymmetry

3. **Deploy to Modal**
   - Create modal_train.py
   - Test with small run (500k steps)
   - Full training (2M+ steps)

4. **Analysis**
   - Compare 4-limb to 2-control performance
   - Analyze learned coordination patterns
   - Check if arms help with efficiency

## Expected Benefits of 4-Limb Model

1. **Asymmetric leg pumping** - different phase for each leg could generate more thrust
2. **Arm counterbalance** - arms can help with pitch stability
3. **More natural motion** - closer to real pump foiling technique
4. **Potentially higher frequency** - alternating legs = 2x effective frequency

## Files to Create
- `foil_env/pump_foil_env_4limb.py` - 4-limb environment
- `modal_train.py` - Modal GPU training script
- `analyze_4limb.py` - Analysis and comparison

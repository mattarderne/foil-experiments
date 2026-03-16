# Next Steps: Applying HUSKY Humanoid Skateboarding Techniques to Pump Foiling

## References

- **HUSKY paper**: [Humanoid Skateboarding System via Physics-Aware Whole-Body Control](https://arxiv.org/abs/2602.03205) (Feb 2026)
- **Code**: [TeleHuman/humanoid_skateboarding](https://github.com/TeleHuman/humanoid_skateboarding) (Unitree G1, MuJoCo + mjlab)

## Why HUSKY Is Relevant to Pump Foiling

Skateboarding and pump foiling share deep structural similarities:

| Property | Skateboarding | Pump Foiling |
|----------|--------------|--------------|
| Platform | Underactuated wheeled board | Underactuated hydrofoil board |
| Control mechanism | Body lean → truck steering | Body pump → pitch-heave thrust |
| Physics coupling | Lean angle → steering angle (tan σ = tan λ sin γ) | Pump motion → AoA changes → lift/thrust |
| Balance challenge | Dynamic balance on moving, coupled platform | Dynamic balance on flying, coupled platform |
| Contact dynamics | Multiple phases (push, ride, turn) | Multiple phases (takeoff, sustained pump, recovery) |
| Whole-body coordination | Arms + torso + legs coordinated for balance + propulsion | Arms + torso + legs coordinated for pump + stability |

Both are **human-on-platform** problems where the platform dynamics are driven indirectly by whole-body motion, and both require solving coupled contact/interaction dynamics.

## Key HUSKY Innovations and How They Apply

### 1. Adversarial Motion Priors (AMP) — Replace Hand-Crafted Rewards

**Current approach problem:** Our reward function in `pump_foil_env_curriculum.py` has ~10 hand-tuned components (velocity, pumping, amplitude, frequency, arm coordination, altitude, pitch, energy, efficiency, crash). Each weight is fragile — small changes cascade unpredictably, and the resulting behavior often satisfies the reward without looking natural.

**HUSKY's solution:** Use a learned discriminator trained on real motion capture data. The AMP discriminator replaces most style/naturalness reward terms with a single "does this look like real human motion?" signal.

**Application to pump foiling:**
- Collect motion capture data of real pump foiling (even from video-based pose estimation)
- Train an AMP discriminator on the reference motions
- Replace the pumping_reward, amplitude_reward, frequency_reward, and arm_coordination_reward with a single AMP reward
- Keep only the task-critical rewards: survival, velocity maintenance, altitude
- This eliminates reward weight tuning and produces more natural-looking pump technique

**Impact: HIGH** — This is the single biggest improvement. Our current reward engineering is the primary bottleneck for producing realistic behavior.

### 2. Physics-Aware Coupling Model — Formalize Pump-to-Thrust

**Current approach problem:** The `pump_thrust_efficiency` parameter (0.08–0.20) is a hand-tuned scalar that linearly converts vertical velocity to forward thrust. This is a crude approximation of the actual unsteady aerodynamic mechanism (Wagner effect, flapping foil propulsion, free-surface effects).

**HUSKY's approach:** They explicitly model the skateboard's kinematic coupling (lean → truck steering → turning radius) from first principles, then use this model to generate physics-aware control references.

**Application to pump foiling:**
- Model the pitch-heave-thrust coupling from oscillating airfoil theory (Theodorsen function, Garrick model)
- The relationship between pump frequency, amplitude, and generated thrust follows known unsteady aero equations
- Use this model to generate "physics-aware pump references" — the ideal pump trajectory given current speed and altitude
- Feed these references as observations or as trajectory targets for the policy
- This replaces the crude `pump_thrust_efficiency` scalar with a principled model

**Impact: MEDIUM-HIGH** — Better physics fidelity means policies transfer better to reality.

### 3. Phase-Wise Learning — Separate Distinct Control Regimes

**Current approach problem:** Our curriculum learning blends between "forced pump" and "free control" with a single `agent_blend` parameter. But pump foiling has distinct phases with different dynamics and objectives that a single blended policy struggles with.

**HUSKY's approach:** Separate policies (or policy phases) for distinct contact modes — pushing for propulsion, steering via lean, and transitions between them. Each phase has its own reward structure and training focus.

**Application to pump foiling — define 3 phases:**

| Phase | Goal | Key Dynamics |
|-------|------|-------------|
| **Takeoff** | Get airborne from surface riding | High-frequency aggressive pump, build speed |
| **Sustained pump** | Maintain altitude and speed indefinitely | Efficient rhythmic pump, energy conservation |
| **Recovery** | Correct from perturbations (gust, wave) | Adaptive, non-rhythmic corrections |

Train each phase with dedicated reward structures, then compose them with a learned transition mechanism (as HUSKY does for push→ride→turn transitions).

**Impact: MEDIUM** — Better than single-policy training, especially for the hard takeoff→flight transition.

### 4. GPU-Accelerated MuJoCo (mjlab / mujoco_warp) — Revive Full-Body Sim

**Current approach problem:** The MuJoCo hybrid approach (`mujoco/hybrid_env.py`) was abandoned because it was slow to iterate and debugging was hard. The Python-only approach was faster but sacrifices body dynamics fidelity.

**HUSKY's infrastructure:** mjlab combines MuJoCo's physics with Isaac Lab's scalable RL APIs and mujoco_warp for GPU acceleration. This enables thousands of parallel environments.

**Application to pump foiling:**
- Port the hybrid environment to mjlab/mujoco_warp for GPU-parallel training
- This makes the full-body MuJoCo approach practical — thousands of parallel envs instead of 4
- The 5-DOF rider model in `rider_hybrid.xml` becomes viable at scale
- Training that took hours on CPU takes minutes on GPU
- Enables proper hyperparameter search over physics/reward params

**Impact: HIGH** — GPU acceleration is what makes full-body MuJoCo + AMP feasible. Without it, the iteration cycle is too slow.

### 5. Full Articulated Body Model — Better Dynamics Fidelity

**Current approach problem:** The Python-only body model (`body_model.py`) is a 2D kinematic chain with simplified force calculations. It misses: inertial coupling between limbs, realistic contact forces, 3D balance effects, and proper momentum transfer.

**HUSKY's approach:** Full Unitree G1 humanoid in MuJoCo with all joints, contacts, and inertial properties. The physics naturally captures how arm swing affects balance, how knee bend creates CoM shift, etc.

**Application to pump foiling:**
- Build a full 3D humanoid-on-foilboard MuJoCo model
- Let MuJoCo handle all body dynamics, contacts (feet on board), and inertial effects
- The foil hydrodynamics can still be computed externally and applied as forces to the board body
- This captures effects our simplified model misses: rotational inertia of arm swing, 3D balance, realistic foot pressure distribution

**Impact: MEDIUM** — Matters most if the goal is sim-to-real transfer. For pure simulation research, the simplified model may suffice.

## Proposed Implementation Roadmap

### Phase 1: AMP Foundation (2-3 weeks)
1. **Collect reference motions** — Extract pump foiling motion data from video (using pose estimation like MediaPipe/MotionBERT) or from our own successful trained policies
2. **Implement AMP discriminator** — Adapt rsl_rl's AMP implementation for our observation space
3. **Simplify reward function** — Replace style rewards with AMP, keep only task rewards (survival + velocity + altitude)
4. **Train and compare** — Run A/B experiments against current hand-crafted reward

### Phase 2: GPU-Accelerated Training (1-2 weeks)
1. **Port to mjlab** — Adapt the hybrid MuJoCo environment to work with mjlab/mujoco_warp
2. **Parallelize** — Scale from 4 envs to 1000+ on GPU
3. **Hyperparameter sweep** — Use the speed to properly search reward weights, physics params, curriculum pacing

### Phase 3: Physics-Aware Pump Model (2 weeks)
1. **Implement Theodorsen/Garrick** — Replace `pump_thrust_efficiency` with unsteady aero model
2. **Generate pump references** — Compute optimal pump trajectory for given speed/altitude
3. **Add reference tracking** — Include physics-aware references in observation space

### Phase 4: Phase-Wise Training (1-2 weeks)
1. **Define phase boundaries** — Takeoff, sustained pump, recovery
2. **Separate training** — Train phase-specific policies with dedicated rewards
3. **Compose** — Learn transition mechanism between phases

### Phase 5: Full 3D Model (2-3 weeks)
1. **Build 3D humanoid-on-foilboard** — Full MJCF model with contact
2. **External foil forces** — Apply hydrodynamic forces as MuJoCo external forces
3. **Train with AMP** — Use reference motions for realistic 3D pump technique

## Quick Win: AMP on Current Python-RL Setup

Even without GPU acceleration or full MuJoCo, we can get immediate value from AMP:

1. Use our best trained policy (`checkpoints/baseline_2M/model_100pct.zip`) to generate "expert" reference trajectories
2. Train a simple discriminator on the joint trajectories (leg position, arm position, waist angle over time)
3. Replace the 6 style-related reward terms with `r_amp = discriminator(current_state, next_state)`
4. Keep `r_task = alive_bonus + velocity_reward + altitude_reward + crash_penalty`
5. Total reward: `r = r_task + λ * r_amp` (with λ ≈ 0.5)

This simplifies the reward from ~10 tunable weights to ~2 (task vs style balance), and the discriminator automatically captures the "right" pump frequency, amplitude, and coordination pattern.

## Summary

| Technique | Effort | Impact | Priority |
|-----------|--------|--------|----------|
| AMP (motion priors) | Medium | High | 1 — eliminates reward engineering |
| GPU-accelerated MuJoCo | Medium | High | 2 — enables everything else |
| Physics-aware pump model | Medium | Medium-High | 3 — better sim fidelity |
| Phase-wise learning | Low-Medium | Medium | 4 — better phase transitions |
| Full 3D body model | High | Medium | 5 — needed for sim-to-real |

The single highest-leverage change is adopting AMP to replace hand-crafted style rewards. The second is GPU-accelerated MuJoCo to make iteration fast enough for proper experimentation.

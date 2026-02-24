# Comparison: Original foilphysics vs Our Model

## Source
Original: https://github.com/lsegessemann/foilphysics

## Critical Difference: Kinematics vs Dynamics

| Aspect | Original foilphysics | Our Model |
|--------|---------------------|-----------|
| **Motion** | Prescribed: `z = A × cos(ωt)` | Computed from forces |
| **Velocity** | **Constant** (assumed) | **Variable** (computed) |
| **Energy** | Not conserved (infinite) | Must balance thrust vs drag |
| **Purpose** | Analyze forces at given motion | Simulate self-sustaining flight |

### Original Model Quote:
> "The motion is defined by math (z = A cos(ωt)), not by forces. The board is essentially moving on a predefined invisible rail."

**This is why the original doesn't need pump_thrust_efficiency** - it assumes you're already moving at constant speed and just calculates what forces occur.

## Physics Formulas: Identical ✓

Both models use the same core equations:

### Lift Coefficient
```
Original: CL = (2π × α) / (1 + 2/AR)
Ours:     Cl = lift_slope * alpha  where lift_slope = 2π / (1 + 2/AR)
```
✓ **Match**

### Drag Coefficient
```
Original: CD = CD0 + k × CL²  where k = 1/(π × AR)
Ours:     Cd = Cd0 + k_induced * Cl²  where k_induced = 1/(π × AR)
```
✓ **Match**

### Angle of Attack
```
Original: α = θ - γ  where γ = arctan(vz/U)
Ours:     alpha = theta - gamma  where gamma = arctan2(vz, vx)
```
✓ **Match**

### Lift Vector Components
```
Original: Lx = -L × sin(γ)
Ours:     Lx = -L_mag * sin(gamma)
```
✓ **Match**

### Stall Model
```
Original: Linear below 13°, CL saturates above
Ours:     Linear below stall_rad, Cl saturates, Cd increases
```
✓ **Match** (we add drag penalty)

### Stabilizer Downwash
```
Original: ε = 2×CL/(π×AR)
Ours:     epsilon = (2 * Cl) / (np.pi * AR)
```
✓ **Match**

### Stabilizer Moment
```
Original: M_stab = -L_stab × fuselage_length
Ours:     M_stab = -(L_stab_z * fuselage_len)
```
✓ **Match**

## What Original Has That We Don't

### 1. Added Mass
```javascript
m_added = ρπ(c/2)² × b × cos²(θ)
```
Hydrodynamic added mass from accelerating water around foil. We don't model this.

### 2. Elastic Efficiency
```javascript
P_linear = max(0, F_rider × |vz|) × elastic_efficiency
```
Energy recovery from tendons/elastic storage. We don't model this.

### 3. Two-Mass Dynamics
```javascript
m_moving = m_board + m_rider × swing_ratio
F_inertia = m_moving × (g + az)
```
Separates rider mass from board mass for inertial forces. We simplify to single mass.

## What We Have That Original Doesn't

### 1. Velocity Integration
```python
new_vx = vx + ax * dt  # Velocity changes!
```
Original assumes constant U. We compute velocity from net forces.

### 2. pump_thrust_efficiency
```python
pump_thrust = efficiency × |vz| × |Lift|
```
Compensates for missing unsteady aero effects. Original doesn't need this because velocity is fixed.

### 3. Full State Evolution
```python
state = foil.step(state, leg_force, pitch_torque, dt)
```
Complete dynamics simulation. Original only calculates instantaneous forces.

## Why pump_thrust_efficiency Exists

The original model sidesteps the energy problem:
- Assumes constant forward velocity
- Calculates forces but doesn't integrate them
- Never asks "is this motion self-sustaining?"

Our model must answer that question:
- Drag continuously decelerates the foil
- Must generate enough thrust to overcome drag
- Without pump_thrust_efficiency, the lift-vector thrust isn't enough

### The Gap

| Model | Thrust from Lx | Drag | Balance |
|-------|---------------|------|---------|
| Original | ~700N | N/A (v fixed) | N/A |
| Ours (eff=0) | ~700N | ~750N | **-50N** (sinks) |
| Ours (eff=0.10) | ~750N | ~750N | **0N** (sustains) |

The pump_thrust_efficiency of 0.08-0.10 represents:
1. Added mass effects (not modeled)
2. Unsteady aero (not modeled)
3. Elastic energy recovery (not modeled)

## Recommendation

Keep pump_thrust_efficiency = **0.08-0.10** as a lumped parameter for physics we're not explicitly modeling, rather than trying to add all the missing effects individually.

The core aerodynamics match the validated original model. The efficiency parameter bridges the gap between kinematics (original) and dynamics (ours).

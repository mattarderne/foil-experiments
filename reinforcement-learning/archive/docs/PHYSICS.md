# Pump Foil Physics Model

## Overview

This document describes the physics model used in the pump foil RL environment.
The model simulates a hydrofoil (underwater wing) carrying a rider, with the goal
of learning to "pump" - generate forward thrust through body motion.

## Coordinate System

- **X**: Forward (direction of travel)
- **Z**: Vertical (positive = up, water surface at z=0)
- **Theta**: Pitch angle (positive = nose up)

## State Variables

| Variable | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Position | x | m | Horizontal position |
| Altitude | z | m | Vertical position (0 = water surface) |
| Forward velocity | vx | m/s | Horizontal velocity |
| Vertical velocity | vz | m/s | Vertical velocity |
| Pitch angle | theta | rad | Board/foil pitch angle |
| Pitch rate | omega | rad/s | Angular velocity |

## Foil Geometry

```
                    BOARD (on surface)
                      ___________
                     |           |
                     |   MAST    |  (0.70m vertical)
                     |___________|
                          |
     STABILIZER ----[===]-+--[======]---- FRONT WING
     (rear, small)        |        (main lift, large)
                     FUSELAGE
                     (0.60m horizontal)
```

### Components

| Component | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| Front wing | S | 0.135 m² | Main lift surface |
| Front wing | AR | 13.9 | Aspect ratio |
| Stabilizer | S_stab | 0.012 m² | Pitch stability |
| Stabilizer | AR_stab | 8.0 | Aspect ratio |
| Stabilizer | stab_angle | -1° | Incidence angle |
| Fuselage | fuselage_len | 0.60 m | Wing to stab distance |
| Mast | mast_length | 0.70 m | Board to wing distance |
| Mast | mast_thickness | 0.015 m | Chord width |

## Force Model

### 1. Main Wing Lift & Drag

The front wing generates lift based on angle of attack (AoA):

```
gamma = atan2(vz, vx)           # Flight path angle
alpha = theta - gamma           # Angle of attack

# Pre-stall (linear region)
if |alpha| < stall_angle:
    Cl = lift_slope * alpha
    Cd = Cd0 + k * Cl²

# Post-stall (saturated lift, high drag)
else:
    Cl = Cl_max * sign(alpha)
    Cd = 0.05 + 2.0 * (|alpha| - stall_angle)

L = 0.5 * rho * S * Cl * v²     # Lift magnitude
D = 0.5 * rho * S * Cd * v²     # Drag magnitude
```

### 2. Stabilizer Forces

The stabilizer provides pitch stability via its lift force:

```
# Downwash from main wing reduces stabilizer AoA
epsilon = 2*Cl / (pi*AR)

# Stabilizer sees different vertical velocity (pitch rotation effect)
vz_stab = vz - omega * fuselage_len
alpha_stab = theta + stab_angle - gamma_stab - epsilon

L_stab = 0.5 * rho * S_stab * Cl_stab * v²
```

### 3. Mast Drag

Drag from the vertical mast, proportional to submerged depth:

```
depth = riding_depth - z
area_mast = depth * mast_thickness
D_mast = 0.5 * rho * area_mast * Cd_mast * v²
```

### 4. Moments

The pitch dynamics are controlled by three moments:

#### Stabilizer Moment (M_stab)
Automatic pitch stability - stabilizer lift creates restoring torque:
```
M_stab = -L_stab_z * fuselage_len
```

#### Rider Weight Moment (M_weight) - NEW
Rider's center of mass position affects trim:
```
M_weight = -rider_weight * rider_com_offset

# rider_com_offset > 0 (forward) → nose-down moment
# rider_com_offset < 0 (back)    → nose-up moment
```

#### Rider Control Torque (pitch_torque)
From leg asymmetry and arm swing:
```
leg_pitch_torque = (right_leg_force - left_leg_force) * stance_width / 2
arm_pitch_torque = -arm_mass * arm_accel * arm_moment
pitch_torque = leg_pitch_torque + arm_pitch_torque
```

### 5. Equations of Motion

```
# Total forces
Fx = Lx + Dx + L_stab_x + D_stab_x + D_mast_x
Fz = Lz + Dz + L_stab_z + D_stab_z + D_mast_z - W - leg_force

# Accelerations
ax = Fx / m_total
az = Fz / m_total
alpha_dot = (M_stab + M_weight + pitch_torque) / I

# Integration (Euler or RK4)
vx += ax * dt
vz += az * dt
theta += omega * dt
omega += alpha_dot * dt
```

## Rider Model (4-Limb)

### Controllable Degrees of Freedom

| DOF | Range | Description |
|-----|-------|-------------|
| Left leg | [-0.15, +0.15] m | Leg extension |
| Right leg | [-0.15, +0.15] m | Leg extension |
| Left arm | [-1.5, +1.5] rad | Arm swing |
| Right arm | [-1.5, +1.5] rad | Arm swing |
| Stance offset | configurable | CoM position (m) |

### NOT Modeled (Simplifications)

- **Waist lean**: Torso is fixed relative to hips
- **Ankle articulation**: Feet are fixed to board
- **3D effects**: All motion is in the sagittal plane

## Physical Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rider mass | 70 kg | Body weight |
| Board mass | 6 kg | Board + foil + gear |
| Water density | 1000 kg/m³ | Fresh water |
| Gravity | 9.81 m/s² | |
| Stall angle | 13° | AoA limit |

## Energy Model

The rider has a limited energy budget:
- **MAX_POWER**: 300 W sustainable output
- **MAX_ENERGY**: 4500 J total budget (~15s at max power)

Power consumption:
```
leg_power = |leg_force * leg_velocity|
arm_power = |arm_torque * pitch_rate|
total_power = leg_power + arm_power + holding_cost
```

## Termination Conditions

| Condition | Threshold | Description |
|-----------|-----------|-------------|
| Foil breach | z >= 0.20 m | Wing exits water |
| Board touchdown | z <= -0.50 m | Board hits water |
| Stall | vx < 1.5 m/s | Lost forward speed |
| Excessive pitch | |theta| > 30° | Loss of control |
| Energy exhausted | energy > 4500 J | Rider fatigued |
| Timeout | steps > 3000 | 30s max episode |

## Key Physics Insights

### Thrust Generation
Pumping generates thrust by tilting the lift vector forward during downward motion:
- When vz < 0 (descending), flight path angle gamma < 0
- Lift vector tilts forward, Lx component becomes positive (thrust)
- Larger vz oscillations = more thrust potential

### Stance Position Effect
Rider position relative to center of lift (CoL) is critical:
- **Forward stance**: Nose-down tendency, risk of pearling
- **Neutral stance**: Balanced, easier to control
- **Rear stance**: Nose-up tendency, risk of stall

### Optimal Pumping Strategy
From RL experiments:
- Frequency: ~2.0-2.5 Hz
- Pitch bias: ~0.4 (nose up)
- Alternating leg motion for continuous thrust
- Arms swing opposite to legs for balance

## File Reference

- `foil_env/foil_physics.py`: Core physics engine
- `foil_env/pump_foil_env_4limb.py`: RL environment with 4-limb rider
- `foil_env/body_model.py`: Rider body kinematics
- `demo_stance_physics.py`: Stance position demo

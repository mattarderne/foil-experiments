# Pump Foil Physics Explained

## The Basic Question

How does pumping (jumping up and down on a foil board) generate forward thrust and maintain flight?

## Key Concept: Angle of Attack (AoA)

The foil generates lift based on its angle relative to the incoming water flow:

```
AoA = board_pitch - flight_path_angle
    = board_pitch - atan2(vz, vx)
```

Where:
- `board_pitch` = how much the nose is tilted up
- `vz` = vertical velocity (positive = moving up)
- `vx` = forward velocity

## How Board Motion Affects AoA

### Stationary (vz = 0)
```
    Water flow →→→→→→→
                      ╲
    ════════════════════  Board at 10° pitch
                        ╲
                         ◇ Foil

    AoA = 10° - 0° = 10°
    Lift: 1400N UP ✓
```

### Moving DOWN (vz = -1.5)
```
    Water flow →→→→→↗    Flow comes from below
                    ↗    because board is descending
    ════════════════════
                        ╲
                         ◇

    AoA = 10° - (-21°) = 31°
    Lift: 2000N UP ✓ (capped by stall)
    Bonus: Lift vector tilts forward → THRUST
```

### Moving UP (vz = +1.5)
```
    Water flow →→→→→↘    Flow comes from above
                    ↘    because board is rising
    ════════════════════
                        ╲
                         ◇

    AoA = 10° - (+21°) = -11°
    Lift: 1700N DOWN ✗ (REVERSED!)
```

## The Pump Cycle Problem

A simple sinusoidal pump spends equal time going up and down:

| Phase | Board Motion | AoA | Lift |
|-------|--------------|-----|------|
| Power stroke | DOWN (vz < 0) | +31° | +2000N up |
| Recovery stroke | UP (vz > 0) | -11° | -1700N down |
| **Average** | | | **+150N** |

**We need 795N to support rider weight. We only get 150N. We sink.**

The negative lift during the "up" phase destroys our average lift.

## Why Gliding Beats Pumping (Without Correction)

| Mode | Average Lift | Result |
|------|--------------|--------|
| Glide (no pump) | 1400N | Float with 600N surplus |
| Pump (large amplitude) | 150N | Sink (645N deficit) |

Counterintuitive: Pumping makes us sink faster than just gliding!

## The Solution: pump_thrust_efficiency

This parameter compensates for the lift problem by adding forward thrust:

```python
pump_thrust = efficiency × |vz| × |Lift|
```

### How It Enables Sustained Flight

With extra thrust available, we can use **smaller pump amplitude**:

| Amplitude | AoA Range | Avg Lift | Thrust |
|-----------|-----------|----------|--------|
| Large (vz = ±1.5) | -11° to +31° | 150N ✗ | High |
| Small (vz = ±0.5) | +3° to +17° | 950N ✓ | Lower |

Small amplitude keeps AoA **positive throughout the cycle** → no lift reversal.

The pump_thrust_efficiency provides the extra thrust we lose from smaller amplitude.

## Physical Justification

The parameter represents real physics not captured in our simplified model:

### 1. Unsteady Aerodynamics
- Wagner effect: lift overshoots during rapid pitch changes
- Added mass: accelerating fluid adds effective inertia
- Contribution: +10-20%

### 2. Flapping Foil Propulsion
- Oscillating foils create "reverse Karman vortex street"
- Wake organizes into thrust-producing pattern
- Well-documented in marine propulsion research
- Contribution: +5-15%

### 3. Free Surface Effects
- Wave generation transfers momentum
- Surface proximity modifies pressure distribution
- Contribution: +5-10%

## Recommended Values

| pump_thrust_efficiency | Flight Duration | Use Case |
|------------------------|-----------------|----------|
| 0.00 | 5s | Incomplete physics |
| 0.08 | 8s | Conservative/realistic |
| 0.10 | 10s | Recommended for training |
| 0.15 | 14s | Upper bound |
| 0.18+ | 15s+ | Overtuned |

## Key Takeaways

1. **Pumping isn't just "flapping for thrust"** - it's a delicate balance of lift and thrust

2. **Large amplitude pumping backfires** - negative AoA during ascent reverses lift

3. **Real pump foiling uses small, efficient oscillations** - just enough to generate thrust while maintaining lift

4. **Weight shift (pitch control) is critical** - pitching nose-up during ascent prevents lift reversal

5. **pump_thrust_efficiency is not a cheat** - it's a lumped parameter for real unsteady aero effects

## The Dolphin Pump Technique

Efficient pump foiling:
```
1. Small vertical amplitude (board barely moves)
2. Pitch nose UP during ascent (prevents negative AoA)
3. Pitch nose DOWN during descent (maximizes thrust)
4. 2-2.5 Hz frequency (optimal energy/thrust ratio)
5. Coordinate arms with legs (same phase, +40% force)
```

This is why independent front/back leg control matters - it enables the weight shift needed for pitch modulation throughout the pump cycle.

# Pump Physics & Energy Tradeoffs

## Why 2.5 Hz is Optimal

For sinusoidal leg pumping:
- Position: `A × sin(ωt)`
- Velocity: `A × ω × cos(ωt)`
- Force (inertia): `m × A × ω² × sin(ωt)`
- **Average power ∝ A² × ω³** (frequency has CUBIC effect on energy)

| Frequency | Energy Cost | Thrust | Notes |
|-----------|-------------|--------|-------|
| 1.0 Hz | 1x (baseline) | Low | Most efficient but not enough lift |
| 2.0 Hz | ~2.5x | Medium | Good balance |
| 2.5 Hz | ~4x | Good | **Sweet spot** - enough thrust, manageable energy |
| 4.0 Hz | ~7x | High (diminishing) | Burns energy, thrust saturates |

## Energy Budget

- Survival environment: 18000J for 60s = ~300W average budget
- At 4 Hz pumping: ~200W just on legs (2/3 of budget for one axis!)
- At 2.5 Hz pumping: ~50W on legs, leaving headroom for pitch control

## Why Higher Frequency Isn't Better

1. **Thrust saturates**: Foil can only accelerate water so fast before flow separation/cavitation
2. **Energy scales cubically**: 2x frequency = 8x theoretical energy (measured ~2x due to physical limits)
3. **Control stability**: Less time per cycle to coordinate pitch with leg extension
4. **Cardio limits**: Real humans can't sustain >300W indefinitely

## Foil Size Effects

| Foil Area | Stall Velocity | Margin from 4.5 m/s | Strategy |
|-----------|---------------|---------------------|----------|
| 1800 cm² | 3.0 m/s | 1.5 m/s | Can glide without pumping |
| 1200 cm² | 3.7 m/s | 0.8 m/s | Needs occasional pumping |
| 800 cm² | 4.04 m/s | 0.46 m/s | **Must pump aggressively** |

Smaller foils force learning to pump because slow gliding crashes.

## Stall Velocity Calculation

```
v_stall = sqrt(2 * m * g / (rho * S * CL_max))
```

Where:
- m = 80 kg (rider mass)
- g = 9.81 m/s²
- rho = 1025 kg/m³ (seawater)
- S = foil area (m²)
- CL_max ≈ 1.2 (maximum lift coefficient)

## Energy Scaling Analysis (Measured)

Testing fixed-pattern pumping at different frequencies:

```
1.0Hz: 1.0x energy (baseline)
1.5Hz: 1.7x energy (theory: 3.4x)
2.0Hz: 2.6x energy (theory: 8.0x)
2.5Hz: 3.6x energy (theory: 15.6x)
3.0Hz: 4.6x energy (theory: 27.0x)
4.0Hz: 6.7x energy (theory: 64.0x)
```

Measured scaling is much gentler than ω³ theory because:
- Force-velocity product has phase offsets
- Physical limits cap maximum power output
- Damping absorbs some energy

## Arm Swing Discovery (Jan 2026)

**Key finding: Arms should swing SAME phase as legs, NOT opposite!**

Tested different arm patterns on trained model (800cm² foil):

| Pattern | Duration | Arm-Leg Correlation | Notes |
|---------|----------|---------------------|-------|
| Baseline (learned) | 5.35s | +0.65 (same) | Model discovered this |
| Opposite (deweighting) | 1.63s | -0.92 | 70% worse! |
| Fixed (no arms) | 3.56s | 0.00 | 33% worse |

### Force Analysis

```
Leg vertical force:   std=427N, range [-1097, 3000]N
Arm vertical force:   std=173N, range [-1131, 240]N  (40% of leg!)
Arm pitch torque:     std=62Nm
```

### Why Same-Phase Works

1. **Force amplification**: When legs push down AND arms accelerate down together, both reaction forces push the foil down harder (+40% total force)

2. **Deweighting theory is WRONG for pump foiling**: Opposite-phase arms CANCEL leg forces, reducing net pumping effect

3. **Whole-body pumping**: The optimal strategy uses arms, legs, and torso as one coordinated unit, all moving together

### Implications

- Don't train for "deweighting" arm swing
- Arms should be included in pumping curriculum (same phase as legs)
- The model naturally discovered this without explicit arm reward

## Implications for Training

1. **Reward velocity, not just survival**: Pure survival reward leads to "wiggle and balance" at low frequency
2. **Use small foils (800 cm²)**: Forces aggressive pumping since gliding is impossible
3. **Target 2-2.5 Hz**: Combined reward should penalize deviations from optimal frequency
4. **Energy awareness**: Agent observation includes power output to learn energy-efficient strategies
5. **Arm curriculum**: If forcing arms, use SAME phase as legs, not opposite

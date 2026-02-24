# Foil RL Pump

Teaching an RL agent to pump foil - maintaining flight through coordinated body movements that generate lift and forward thrust on a hydrofoil.

Two approaches were explored:

## [Python RL](python-rl/) - Custom Physics + PPO

**Status: Working.** Custom 2D hydrodynamic simulation with Stable Baselines3 PPO.

- 60+ second sustained flights
- 2.3Hz pump frequency matching real biomechanics
- Trained model included
- Fast iteration with matplotlib visualization

## [MuJoCo](mujoco/) - Articulated Body + Hybrid Physics

**Status: Exploratory.** MuJoCo handles the rider body while Python computes foil hydrodynamics.

- Hybrid architecture is viable
- Scripted pumping works (60s+ flights)
- RL training not completed
- Valuable physics documentation and validation

## Which to Look At

If you want to **see results**, start with [python-rl/](python-rl/). The trained model achieves sustained pumping flight.

If you're interested in **physics modeling** or **MuJoCo integration**, the [mujoco/](mujoco/) directory has detailed documentation on the hybrid approach and validated physics.

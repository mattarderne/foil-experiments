# Foil Design — Hydrofoil Wing Optimization

Autonomous optimization of hydrofoil wing and stabilizer shapes using the
[Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern.
Define what kind of foil you want in plain English, and Claude Code explores
the design space while you're away.

**Status:** planning

## The Problem

Foil wings (front wing + stabilizer) have dozens of interacting parameters —
profile shape, aspect ratio, chord distribution, sweep, dihedral, thickness.
Most designs are done by copying what works and tweaking by feel. Simulation
can systematically explore the space and find better tradeoffs.

## Simulation Stack

| Layer | Tool | Speed | What it does |
|-------|------|-------|-------------|
| 2D profile | [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) | ~1ms/eval | ML replacement for XFoil. 30-1000x faster, differentiable, no convergence failures. Cl/Cd/Cm at any AoA and Re. |
| 3D planform | [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) VLM | ~seconds | Vortex lattice method for full 3D wing analysis. Wraps NeuralFoil, supports gradient-based optimization with autodiff. |
| System validation | [Foilboard](https://github.com/dmitrynizh/foilboard) | ~seconds | Full hydrofoil system simulator (front wing + stab + mast + fuselage + rider). Finds equilibrium, predicts takeoff/cruise/max speed. |
| High-fidelity (optional) | [DAFoam](https://github.com/mdolab/dafoam) / OpenFOAM | minutes-hours | Full 3D CFD for validating best designs. Not needed for the outer loop. |

### Why NeuralFoil over XFoil

From a practitioner who built a hydrofoil optimizer with XFoil:

> XFoil is hard to get running in-process and multi-threaded. I had to mess
> with Fortran code to get the data I needed and recompile the XFoil python
> module. XFoil can sometimes just refuse to converge, my code has a lot of
> error handling and retrying to deal with that.

NeuralFoil is `pip install neuralfoil` — pure Python, no Fortran, no
convergence failures, 1000x faster in batch. Uses Kulfan/CST parameterization
natively. [Optimization tutorial](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools/02%20-%20NeuralFoil%20Optimization.ipynb).

### The Hard Parts

1. **Cost function design** — What does "good" mean? It's multi-objective:
   low-speed lift (takeoff), cruise efficiency, stall margin, speed range,
   stability. The `program.md` is where you define this in plain English.

2. **Parameterization** — Kulfan/CST is built into NeuralFoil. Must verify it
   can express real hydrofoil sections (thicker, more cambered than aircraft
   airfoils). We have Shape3D reference files to validate against.

3. **3D effects** — 2D profile optimization only gets you so far. Planform
   (span, taper, sweep, twist) and front-wing/stabilizer interaction matter.

## Reference Geometry

We have high-quality Shape3D (.s3dx) files for a complete foil system:

| File | Type | Span | Chord | Thickness | Area |
|------|------|------|-------|-----------|------|
| `K2_800_Rocket.s3dx` | Front wing | 78.4 cm | 20.3 cm | 3.9 cm (19% t/c) | 805 cm² |
| `Stab.s3dx` | Stabilizer | 32.0 cm | 5.5 cm | 0.6 cm (11% t/c) | 140 cm² |

The `.s3dx` format is XML containing degree-3 Bezier curves defining:
- **Planform outline** (`Otl`) — wing shape from above
- **Stringer curves** (`StrBot`, `StrDeck`) — camber/rocker lines along span
- **Surface patches** (`curveDefTop`, `curveDefSide`) — full 3D surface geometry

These will be parsed, sliced into 2D cross-sections at spanwise stations, and
fitted to Kulfan/CST parameters to validate the parameterization and serve as
optimization baselines.

## Planned Architecture

```
foil-design/
├── CLAUDE.md                  # Agent instructions for outer loop
├── program.md                 # Research strategy (human steers in plain English)
├── optimize_profile.py        # Phase 1: 2D section optimization (agent modifies)
├── optimize_wing.py           # Phase 2: 3D planform optimization (agent modifies)
├── results.tsv                # Append-only experiment log
├── requirements.txt           # aerosandbox, neuralfoil, numpy, matplotlib
├── foilsim/
│   ├── s3dx_parser.py         # Parse Shape3D .s3dx → Bezier → cross-sections
│   ├── profile_fitting.py     # Fit extracted profiles to CST/Kulfan
│   ├── objectives.py          # Cost functions (the hard part)
│   ├── system_model.py        # Full system: front wing + stab + mast + rider
│   └── export.py              # .dat profiles, comparison plots, STL
├── profiles/
│   ├── reference/             # Shape3D source files (.s3dx)
│   │   ├── K2_800_Rocket.s3dx
│   │   └── Stab.s3dx
│   ├── extracted/             # 2D .dat profiles sliced from Shape3D wings
│   └── fitted/                # CST/Kulfan parameter fits
└── results/
```

## Optimization Phases

### Phase 1: 2D Profile Optimization
- Parameterize with CST/Kulfan (built into NeuralFoil)
- Validate Kulfan can represent the K2 800 Rocket and Stab profiles
- Sweep AoA range and Reynolds numbers for foil speeds (Re ~500k–2M)
- Optimize for: max L/D, Cl at target speed, stall angle, drag bucket width
- Knobs: camber, thickness, max-thickness position, LE radius, TE angle

### Phase 2: 3D Planform Optimization (AeroSandbox VLM)
- Front wing: span, root/tip chord, taper, sweep, dihedral, twist distribution
- Stabilizer: same + distance from front wing, relative size, moment arm
- Multi-objective: lift at cruise, stability margin, turning, pump efficiency
- Use K2 800 Rocket planform as baseline to beat

### Phase 3: Integrated Design
- Best profile on best planform
- Multi-speed optimization (takeoff vs cruise vs top speed)
- Rider weight sensitivity (60–100 kg)

### Phase 4 (optional): CFD Validation
- DAFoam/OpenFOAM on Modal GPUs for top designs
- Cavitation analysis at high speed

## Other Useful References

| Tool | What | URL |
|------|------|-----|
| xfoil-optimization-toolbox | Complete 2D airfoil optimization pipeline | [GitHub](https://github.com/The-Fonz/xfoil-optimization-toolbox) |
| Xoptfoil2 | Mature Fortran airfoil optimizer | [GitHub](https://github.com/jxjo/Xoptfoil2) |
| pymead | Parametric multi-element airfoil design | [GitHub](https://github.com/mlau154/pymead) |
| Airfoil-Shape-Optimization-RL | RL-based foil optimization | [GitHub](https://github.com/atharvaaalok/Airfoil-Shape-Optimization-RL) |
| XFLR5 | GUI integrating XFoil + VLM, popular with foil designers | [SourceForge](https://sourceforge.net/projects/xflr5/) |
| AeroPy | XFoil wrapper with CST parameterization | [GitHub](https://github.com/leal26/AeroPy) |
| OF2 | Open-source kitesurf hydrofoil hardware | [GitHub](https://github.com/ludopenfoil/OF2) |
| hydrofoil-profile-selection | Hydrofoil mast profile optimizer — evaluates structural rigidity (Iy/Ix) vs drag across symmetric airfoils using viiflow CFD. Useful reference for jointly optimizing aero + structural metrics | [GitHub](https://github.com/cdorfer/hydrofoil-profile-selection) |

## Getting Started (when ready to build)

```bash
cd foil-design
python3 -m venv venv && source venv/bin/activate
pip install aerosandbox neuralfoil numpy matplotlib

# Launch Claude Code
claude
# Then: "Read README.md and program.md. Start with Phase 1."
```

# Recursive Board Design

**Status:** planning

Autonomous foil board design through recursive self-improvement. Load existing
board geometries as baselines, simulate how they perform across critical riding
scenarios (touchdown, takeoff, pumping, stability, turning), infer what makes a
good design, then recursively generate and evaluate new designs that push the
performance frontier.

Follows the [Karpathy autoresearch](https://github.com/karpathy/autoresearch)
pattern: a human writes the research strategy in `program.md`, and Claude Code
runs the experiment loop autonomously — proposing designs, simulating them,
keeping improvements, reverting failures, and iterating.

## How It Works

### Phase 1 — Baseline Analysis

Load board profiles from `profiles/baseline/` (Shape3D `.s3dx` files) and
extract geometry + simulated performance:

| Scenario | What it measures | Key metric |
|----------|-----------------|------------|
| Touchdown at 5–20° AoA | Water re-entry forgiveness after jumps | Touchdown forgiveness ratio |
| Takeoff speed | Minimum foiling speed | V_takeoff (knots) |
| Tail sink / pumping | Pump stroke effectiveness | Pump authority (°/kg) |
| Pitch stability | Resistance to oscillation at speed | Pitch damping ratio |
| Low-speed turning | Manoeuvrability at 8–12 knots | Turn radius (m) |

### Phase 2 — Design Model

Build a surrogate model that maps board geometry → predicted performance. Start
with linear regression, graduate to Gaussian processes or neural networks as data
accumulates.

### Phase 3 — Recursive Generation

Use the model to propose new board geometries. Simulate them. If they improve the
Pareto frontier, add them to the baseline pool and retrain the model. Every
successful design makes the next proposal better.

After N iterations: a diverse set of Pareto-optimal boards across all 5
performance axes, plus a trained model that understands which geometry changes
improve which aspects of ride quality.

## Quick Start

```bash
cd recursive-board-design
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch Claude Code and say:
# "Read program.md and start designing. Run experiments autonomously."
```

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Human-written research strategy (the only file you edit) |
| `CLAUDE.md` | Agent instructions for the autonomous loop |
| `design.py` | Main experiment script (agent-modified) |
| `evaluate.py` | Performance evaluation across 5 riding scenarios |
| `model.py` | Surrogate design model |
| `results.tsv` | Append-only experiment log |
| `profiles/baseline/` | Reference board geometries (.s3dx) |

## Baseline Boards

| Board | Dims | Volume | Notes |
|-------|------|--------|-------|
| TT60 | 164 × 49.5 × 11.7 cm | ~60L | All-round foil board, squash tail. Ken Adgate inspired, by Tom Partington. |

More baselines welcome — drop any `.s3dx` file into `profiles/baseline/`.

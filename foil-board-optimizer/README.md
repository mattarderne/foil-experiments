# Foil Board Internal Structure Optimizer

Autonomous topology optimization of hydrofoil board internal structures using
FEA simulation and an AI-driven experimental outer loop, following the
[Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern.

## Problem

Current foil boards are made hollow, which causes:
- Energy absorption through deck flex ("soft" feeling)
- Poor energy transfer from rider through the deck into the foil mast
- Suboptimal stiffness-to-weight ratio

## Solution

Inspired by [Saab's AI-designed 3D-printed aircraft fuselage](https://www.saab.com/newsroom/stories/2025/december/saab-announces-worlds-first-software-defined-aircraft-fuselage-with-divergent-technologies)
(99% fewer parts, ~45% less weight), this project designs an optimal internal
structure that:
- Maximizes stiffness along the load path (rider → deck → mast mount)
- Minimizes material usage (weight)
- Is 3D-printable (exports STL)
- Optimizes in full 3D (not just 2D cross-sections)

## Architecture: autoresearch Pattern

Following Karpathy's autoresearch, the system has a clean separation:

```
┌──────────────────────────────────────────────────────────┐
│  program.md     — research strategy (human modifies)     │
│  optimize.py    — experiment code (agent modifies)       │
│  results.tsv    — experiment log (append-only)           │
└──────────────────────────────────────────────────────────┘

OUTER LOOP (Claude Code, autonomous):
  1. Read program.md for strategy
  2. Read results.tsv for past experiments
  3. Form hypothesis
  4. Modify optimize.py (parameters, approach)
  5. Run: python optimize.py  (~5 min per experiment)
  6. Evaluate: if compliance improved → KEEP, else → REVERT
  7. Commit kept improvements
  8. Repeat forever (the human might be asleep)

INNER LOOP (optimize.py, each run):
  1. Generate 3D hex mesh of board
  2. Apply rider loads + mast mount BCs
  3. SIMP topology optimization (iterative FEA)
  4. Log compliance/stiffness to results.tsv
  5. Export STL for 3D printing
```

## Quick Start

```bash
cd projects/foil-board-optimizer
pip install -r requirements.txt

# Run a single experiment (baseline)
python optimize.py

# View results
cat results.tsv

# Launch Claude Code as autonomous researcher
# (point it at program.md and let it run overnight)
```

## Autonomous Research Mode

Launch Claude Code in this directory and tell it:
> "Read program.md and start optimizing. Run experiments autonomously."

Claude Code will:
1. Read `program.md` for the research strategy
2. Check `results.tsv` for the current best compliance
3. Modify parameters in `optimize.py`
4. Run the experiment, evaluate, keep/revert
5. Loop indefinitely, exploring the parameter space

The agent has freedom to try different mesh resolutions, SIMP parameters,
load case combinations, Heaviside projection, continuation methods, etc.

## Programmatic Research Mode

For scripted multi-experiment campaigns without Claude Code:

```bash
# Run the built-in auto-researcher (exploration → refinement → validation)
python -m foilopt research --max-experiments 30 --per-generation 4
```

## Key Files

| File | Who modifies | Purpose |
|------|-------------|---------|
| `program.md` | Human | Research strategy and agent instructions |
| `optimize.py` | Agent | Experiment configuration (the knobs) |
| `results.tsv` | Append-only | All experiment results (never delete) |
| `foilopt/` | Nobody | Core optimization library |

## Key Modules

- `foilopt/geometry/` — Board shape, mast mount zones, hex mesh generation
- `foilopt/fea/` — 8-node hex element stiffness, sparse assembly, linear solver
- `foilopt/topology/` — SIMP optimizer, density filters, Heaviside projection
- `foilopt/ml/` — Neural surrogate model, U-Net topology predictor (PyTorch)
- `foilopt/harness/` — Programmatic auto-researcher with persistent state
- `foilopt/utils/` — STL export, 3D voxel visualization, convergence plots

## Physics

The board is modeled as a 3D domain with:
- **Deck surface** (top): rider applies downward force in the foot zone
- **Mast mount** (bottom center): fixed boundary where foil mast bolts through
- **Load cases**: normal riding, pumping, jump landing, carving turns

The SIMP optimizer places material where it's structurally needed along the
load path from deck to mast, and removes it elsewhere — creating internal
ribs, lattices, and channels rather than a simple hollow shell.

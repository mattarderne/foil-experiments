# Foil Board Internal Structure Optimizer

Topology optimization of a hydrofoil board's internal structure using FEA simulation.
Places material where it structurally matters along the rider → deck → mast load path,
producing a 3D-printable internal structure rather than a hollow shell.

## Results

### Continuous internal structure (Phase 3 — marching cubes STL from 23 cross-sections)
https://github.com/mattarderne/foil-experiments/raw/main/foil-board-optimizer/media/continuous-internal-structure.mp4

### Optimized cross-section PNGs (Phase 2 — 2D SIMP at each bulkhead)
https://github.com/mattarderne/foil-experiments/raw/main/foil-board-optimizer/media/cross-section-pngs.mp4

### Interactive HTML viewer — load transfer + safety factor overlay (Phase 1)
https://github.com/mattarderne/foil-experiments/raw/main/foil-board-optimizer/media/viewer-board-blocks.mp4

---

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

## Pipeline

Three phases, each building on the last:

```
Phase 1 — 3D bulkhead SIMP (modal_run.py)
  ↓ 56×20×12 hex mesh, X-column design vars, finds transverse rib positions
  ↓ Output: density.bin, strain_energy per load case, meta.json

Phase 2 — 2D cross-section SIMP (run_cross_sections.py)
  ↓ 100×30 Q4 mesh at each X-slice, load-weighted from Phase 1 strain energy
  ↓ Board-shape-aware inside/shell masks, mast zone loads
  ↓ Output: density_x{pos}.npy, section_x{pos}.png per slice

Phase 3 — 3D assembly (build_3d_structure.py)
  ↓ Stack slices → linear interpolation → marching cubes
  ↓ Output: structure.stl (watertight, 3D-printable), structure_overview.png
```

## Quick Start

```bash
pip install -r requirements.txt

# Phase 1: 3D coarse SIMP on Modal (finds bulkhead positions)
python modal_run.py --nelx 56 --nely 20 --nelz 12 --max-iter 100 \
  --target-mass 8.0 --output results/run1

# Phase 1 bulkhead mode (X-column vars, cleaner plate results)
python modal_run.py --bulkhead-mode --bulkhead-xmin 0.65 --bulkhead-xmax 1.31 \
  --output results/bulkhead

# Phase 2: 2D cross-sections at all active slices
python run_cross_sections.py --phase1 results/bulkhead \
  --xmin 0.65 --xmax 1.31 --nely 100 --nelz 30 --volfrac 0.30 \
  --output results/cross_sections

# Phase 3: assemble into continuous 3D STL
python build_3d_structure.py --cross-sections results/cross_sections

# Build interactive HTML viewer
python build_viewer.py results/bulkhead
open viewer.html
```

## Key Files

| File | Purpose |
|------|---------|
| `modal_run.py` | Phase 1: run 3D SIMP on Modal cloud |
| `run_cross_sections.py` | Phase 2: run 2D SIMP at each bulkhead X-slice |
| `build_3d_structure.py` | Phase 3: interpolate slices → watertight STL |
| `build_viewer.py` | Build interactive HTML viewer from Phase 1 results |
| `foilopt/topology/simp.py` | 3D SIMP optimizer (hex elements, bulkhead mode) |
| `foilopt/topology/cross_section.py` | 2D SIMP optimizer (Q4 plane-stress) |
| `foilopt/fea/solver.py` | FEA solver, board-shape-aware BCs |
| `foilopt/geometry/board.py` | FoilBoard geometry, BoardShape (.s3dx parser) |

## Key Modules

- `foilopt/geometry/` — Board shape, mast mount zones, hex mesh generation
- `foilopt/fea/` — 8-node hex element stiffness, sparse assembly, linear solver
- `foilopt/topology/` — 3D SIMP + 2D cross-section SIMP, density filters
- `foilopt/utils/` — STL export, 3D voxel visualization, convergence plots

## Physics

The board is modeled as a 3D domain with:
- **Deck surface** (top): rider applies downward force in the foot zone
- **Mast mount** (bottom center): fixed boundary where foil mast bolts through
- **Load cases**: normal riding (1g), pumping (1.5g), jump landing (2.5g), carving turn

The SIMP optimizer places material where it's structurally needed along the
load path from deck to mast, and removes it elsewhere — creating internal
ribs, webs, and lightening holes rather than a simple hollow shell.

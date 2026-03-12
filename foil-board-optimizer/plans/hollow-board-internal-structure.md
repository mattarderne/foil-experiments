# Hollow Board Internal Structure — Design Plan

## Status

- [x] **Phase 1** — 3D SIMP in bulkhead mode: finds transverse rib positions (`modal_run.py --bulkhead-mode`)
- [x] **Phase 2** — 2D cross-section SIMP at each X-slice at 5mm×4mm resolution (`run_cross_sections.py`)
- [x] **Phase 3** — Interpolate slices → continuous 3D STL (`build_3d_structure.py`)
- [x] **Phase 4** — Outer shell + internal structure merged into complete board STL (`build_complete_board.py`)

---

## Next Steps

The architecture is sound. Next steps are about better physics feeding the pipeline and closing the structural validation loop — not changing the phase structure.

### 1. Re-run Phase 1 with richer load model *(immediate)*

PR #2 (merged) added explicit per-foot 3D force vectors, two new load cases (`front_foot_drive`, `back_foot_drive`), objective weights, and mast torque application. Re-run Phase 1 on Modal to get updated strain energy maps under asymmetric rider loading.

- [ ] `python modal_run.py --bulkhead-mode --bulkhead-xmin 0.65 --bulkhead-xmax 1.31 --rider-kg 85`
- [ ] Compare new strain energy map — expect stronger front/back asymmetry

### 2. Let richer Phase 1 drive Phase 2 more deliberately

Currently Phase 1 → Phase 2 passes only scalar strain energy per slice. With explicit foot-patch geometry and direction now in metadata, Phase 2 can inherit per-foot vector loads at each X position.

- [ ] In `run_cross_sections.py`, read `front_foot_force` / `back_foot_force` from meta per load case
- [ ] Pass directional loads to `CrossSectionOptimizer` alongside deck force magnitude
- [ ] Cross-sections near front foot vs back foot should show different topologies

### 3. 3D validation of the assembled continuous structure

Interpolation between cross-sections is not the last structural word. After assembling the continuous volume, run a 3D FEA pass to check adequacy, find weak spots in interpolated regions, confirm load paths are intact end-to-end.

- [ ] Load `complete_board_volume.npy`, assemble stiffness from solid voxels
- [ ] Apply Phase 1 load cases to the assembled geometry
- [ ] Map compliance back onto the volume for visualisation
- [ ] Flag high-compliance interpolated regions as candidates for additional slices

### 4. Full-resolution Phase 2 on Modal

Cross-sections at 100×30 lose thin ribs. At 200×60 (2.5mm×2mm): genuine thin-wall ribs, lightening holes, diagonal tension members.

- [ ] Add Modal function to `run_cross_sections.py` — fan out all slices as parallel remote calls
- [ ] Run at `--nely 200 --nelz 60 --max-iter 150`
- [ ] Rebuild complete board

### 5. ~~Merge internal structure with outer shell~~ ✓ Done

`build_complete_board.py` — outer shell + internal structure, discrete-plate and continuous modes.

### 6. Extend beyond the foot zone

- [ ] `run_cross_sections.py --xmin 0 --xmax 1.64 --volfrac 0.12` (full board at lower fill)
- [ ] Rebuild complete board

### 7. Manufacturability constraints

- [ ] Minimum member thickness, maximum overhang angle, infill-to-shell bond points

### 8. Fiber-reinforced shell + printed core hybrid

- [ ] Dual-material `CrossSectionConfig`: `E0_shell` (carbon 70GPa) vs `E0_core` (print 5GPa)

---

## Background: Why the 2.5D Approach Was Right

The original plan evaluated four options:

| Option | Verdict |
|--------|---------|
| A: Tune SIMP + MaxSolid | Implemented — MaxSolid disabled in bulkhead mode, sensitivity averaging added |
| B: Increase 3D mesh resolution | Impractical — 2.6M elements, 50+ hrs/run |
| C: 2.5D cross-section optimizer | **Chosen** — implemented as Phase 2, works well |
| D: Moving Morphable Components | Not needed — cross-section approach gives clean plate-like output |

The cross-section approach produces directly CNC-cuttable or 3D-printable bulkhead profiles at each X position. The continuous interpolation between slices (Phase 3) gives the smooth 3D structure needed for a printed core.

---

## Physical Reality: What Is a Hollow Foil Board?

```
Cross-section (rear foot zone):
┌─────────────────────────────┐  ← deck skin (0.5–1mm carbon)
│   rib    rib    rib    rib  │  ← transverse bulkheads (~4mm carbon)
│                             │
│         ←air→               │
└─────────────────────────────┘  ← hull skin (0.5–1mm carbon)
         ↑ stringer
```

Key structural members:
- **Deck/hull skins** — modelled as the forced-solid shell
- **Transverse bulkheads** — resist rider weight, local crushing
- **Longitudinal stringers** — resist nose-to-tail bending + torsion
- **Diagonal ribs near mast** — resist torsional loads from foil

---

## References

- Fernández et al. (2020): *Imposing min/max member size, cavity size, separation distance* — CMAME, arXiv:2003.00263
- Guest (2009): *Imposing maximum length scale* — Struct. Multidisc. Optim.
- Lazarov, Wang & Sigmund (2016): *Length scale and manufacturability in density-based TO* — Arch. Applied Mechanics

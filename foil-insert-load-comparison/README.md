# Foil Insert Load Comparison

Compare how a Tuttle box and a regular two-track mast plate load the same foil board shell and core.

The aim is not to model mast flex or insert-to-mast slop. The mast is treated as rigidly coupled to the insert in both cases. The comparison is only about how load moves from the insert into the board structure.

## Existing Setup Review

There are two relevant starting points in this repo:

- `foil-board-optimizer/` already has a 3D hex-mesh solver, TT60 board-shape handling, and working viewer code for board load visualization.
- That existing project is materially better than starting from scratch, but it still does not answer this insert-specific comparison directly.

What is still missing for this specific question:

- Two explicit insert topologies in the same board.
- Separate material regions for carbon shell, EPS core, and HD foam insert.
- Distinct bond-line definitions.
- A result viewer focused on shell stress, foam stress, and bond-line shear around the insert.

## Experiment Goal

Use the same board outer geometry for both variants and isolate only the insert-to-board transfer path:

- Variant A: Tuttle box
- Variant B: Regular dual-track mast cassette

Primary question:

- Which insert puts lower peak stress and lower bond-line demand into the shell and surrounding core for the same mast-side force and moment set?

Secondary question:

- How does the load split through top shell, bottom shell, HD foam, and nearby EPS differ between the two insert architectures?

## Geometry Definition

Use the same board shell for both cases:

- Board reference: the same TT60 board used by the active foil-board optimizer Shape3d model / board STL.
- Outer shell: identical in both variants.
- Carbon shell thickness: 4 mm.
- Internal fill: EPS foam everywhere except the insert cassette.

Insert geometry:

- Tuttle box overall: 2.0 x 7.5 x 4.75 in
- Tuttle box metric: 50.8 x 190.5 x 120.65 mm
- Dual-track cassette overall: 20.0 x 7.5 x 1.6 in
- Dual-track cassette metric: 508.0 x 190.5 x 40.64 mm
- Mast chord at insert interface: 4.5 in = 114.3 mm

Bonding assumptions:

- Tuttle box is bonded to both top shell and bottom shell.
- Dual-track cassette is bonded only to the bottom shell.
- No slip and no flex between mast and insert.

## Material Model

First-pass comparison:

- Carbon shell: linear elastic shell/solid region, 4 mm thick.
- EPS core: low-modulus isotropic solid.
- HD foam insert: higher-modulus isotropic solid than EPS.
- Mast and hardware: rigid body or very high modulus solid.

Recommended modeling approach:

- First pass: linear elastic comparison with tied interfaces so the force path is stable and easy to compare.
- Second pass: add cohesive contact or interface elements on bond lines to estimate shear and peel demand directly.

This study is about relative architecture performance, so exact laminate values are less important than keeping the same material set and mesh strategy between both variants. Manufacturer coupon data can be dropped in later.

## Load Cases

Run the same load cases for both insert variants. Use the mast base as a rigid coupling surface and distribute rider reactions over the stance patches already used by the TT60 board model.

Recommended normalized load set for the first pass:

1. Pure vertical transfer
   - Mast resultant: +1.0 kN vertical
   - Rider reaction: -1.0 kN split across front and back foot patches
   - Purpose: compare bottom-skin bearing and general shell spread

2. Vertical plus pitch moment
   - Mast resultant: +1.0 kN vertical
   - Mast moment: 150 N m pitch about the board Y-axis
   - Rider reaction: balanced over front/back foot patches
   - Purpose: compare fore-aft shell bridging and top-vs-bottom load sharing

3. Vertical plus lateral carve load
   - Mast resultant: +0.8 kN vertical, +0.4 kN lateral
   - Mast moment: 100 N m roll / torsion
   - Rider reaction: matching deck-side reaction
   - Purpose: compare sidewall loading, insert torsion path, and shell asymmetry

4. Pump impulse envelope
   - Same directions as case 2 but at 2.0x magnitude
   - Purpose: rank the same architecture under a higher transient envelope

Because the first pass is linear elastic, results can be scaled later. The key is to compare both inserts under identical resultant sets before arguing about exact absolute load magnitudes.

## Meshing Strategy

Use the actual board outer geometry, not a rectangular block.

Recommended mesh:

- Global board mesh: tetra or hex-dominant equivalent around 6-8 mm.
- Local refinement around insert and bond lines: 1.5-2.5 mm.
- Keep the shell through-thickness resolved with at least 2 elements if using 3D solids for the shell.
- If shell elements are used instead, tie them to the foam solids with compatible interfaces.

Trimmed-domain option:

- For fast iteration, cut a local board section around the mast area with enough surrounding shell to avoid edge effects.
- For final comparison, re-run on the full board so the real stance and shell curvature are preserved.

## Outputs To Compare

For each load case and each insert variant, capture:

- Peak von Mises or Tsai-Hill equivalent in the carbon shell.
- Peak principal tension in the shell near the insert perimeter.
- Peak compressive stress in HD foam.
- Peak compressive and shear stress in the nearby EPS.
- Bond-line shear and peel on each bonded face.
- Mast-base displacement and rotation.
- Strain energy split by region:
  - top shell
  - bottom shell
  - HD foam insert
  - nearby EPS

Two comparison metrics matter most:

- Peak shell stress normalized by applied mast load.
- Peak bond-line shear normalized by applied mast moment.

## Expected Structural Difference

Hypothesis before solving:

- The Tuttle box should spread pitch moment through a through-thickness load path because it engages both top and bottom shell directly.
- The dual-track cassette should put more of the bending and peel demand into the bottom shell because its primary bond path is bottom-only.
- The tracks may still do well in pure vertical loading because the cassette is much longer in X and can spread load over more bottom skin area.

That means the winner may change by load case:

- Tracks may look better in pure vertical load spread.
- Tuttle may look better in pitch-dominated transfer because it closes the couple through both skins.

## Viewer Design

The viewer should be a side-by-side comparison, not a single-scene plot.

Required viewer layers:

- Transparent outer board shell.
- EPS core.
- HD foam insert.
- Bond-line surfaces.
- Rigid mast interface patch.
- Force and moment arrows.
- Stress or strain-energy field.

Required interactions:

- Toggle between `tuttle` and `dual-track`, or show both side by side.
- Switch load case.
- Toggle material layer visibility.
- Cut plane in X, Y, and Z.
- Probe readout at cursor.
- Show reaction split by material region.

Best field to visualize first:

- Strain energy density or von Mises stress in the shell.

Best bond visualization:

- Bond-line shear heatmap on the interface faces.

## Recommended Implementation Order

1. Reuse the TT60 board geometry and stance geometry from `foil-board-optimizer/`.
2. Replace the single generic mast patch with two explicit insert solids.
3. Add multi-material region tagging:
   - shell
   - EPS
   - HD foam
   - rigid mast interface
4. Solve the normalized load cases.
5. Export per-element results as VTK or compact JSON.
6. Build the side-by-side viewer from those result files.

## Artifacts In This Project

- `experiment-config.json`: machine-readable study setup
- `viewer-prototype.html`: geometry and load-path visualization mockup for the comparison UI

## Practical Note

This project uses the same TT60 Shape3d source already present in this repo under `foil-board-optimizer/board_shape.s3dx`, with a fallback to the board-design reference copy.

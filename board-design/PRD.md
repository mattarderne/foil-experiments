# Board Design — Product Requirements Document

## Vision

A Claude Code session as a CAD collaborator for foil board design. You describe
what you want in plain English, and Claude modifies the board's Shape3D file,
shows you a 3D preview, and writes a commit explaining what it changed and why.
You iterate. When you're happy, the file goes back into Shape3D for final
polish, or directly into the FEA optimizer for structural analysis.

This is not trying to replace Shape3D. Shape3D is the right tool for detailed
surface modeling. This is the layer above it: **design intent in conversation**.

---

## The Problem

A foil board designer typically:

1. Opens Shape3D, makes small changes, re-renders, thinks, makes more changes
2. Has no record of *why* a decision was made — just the final file
3. Can't easily say "take my current design and make it 15% stiffer across the
   tail block" — that's 20 minutes of manual control point nudging
4. Has no bridge between the shape design and the structural analysis (FEA)
5. Can't explore "what if" variants quickly side-by-side

Claude Code can act on design intent directly: it knows the board geometry,
understands surfboard/foil board language, can modify Bezier control points
mathematically, render a preview, and justify every choice in a commit message.

---

## Reference Design

**TT60 — Ken Adgate Inspired** (by Tom Partington / tt design ltd)

| Parameter | Value |
|-----------|-------|
| Length | 164.1 cm |
| Width | 49.5 cm |
| Thickness | 11.7 cm |
| Nose rocker | 12.8 mm |
| Tail rocker | 5.8 mm |
| Volume | ~60L (estimated from FEA mesh) |
| Surface area | 6,478 cm² |
| Rider weight | 85 kg |

**Hardware boxes:** Leashplug, Twin Track, HD Foam insert, back footstrap,
front footstrap centre, front footstrap outer.

**Surface complexity:** 21 top-surface curve definitions including stringer,
Box Hump Curve, Bottom outside edge, Chine Upper Edge, outer rail, Deck Upper
Curve. Standard Shape3D foil board schema.

This is the baseline. Every design iteration starts from here (or a named
variant derived from it) and the original is never overwritten.

---

## User Persona

**Primary:** An intermediate-to-advanced foil boarder who has some familiarity
with board design concepts (rocker, outline, concave, volume) but is not a
professional shaper. They want to iterate on a real design they've sourced from
a shaper, personalise it for their weight and style, and understand the
structural implications of changes.

**Secondary:** A shaper who wants to use AI to explore variants quickly — "give
me 5 outline variations around this tail width" — before committing time to
detailed work in Shape3D.

---

## Core Interaction Model

### Single session, conversational

The user opens a Claude Code session in the `board-design/` directory. Claude
reads the current design state from `design.s3dx` (the working file) plus
`design_log.md` (history of all changes and rationale).

The user types requests. Claude:
1. Parses the intent
2. Identifies which Bezier curves / parameters to modify
3. Applies the change mathematically
4. Writes the modified `design.s3dx`
5. Generates a 3D preview (HTML viewer, opens in browser)
6. Explains what it changed, why, and what tradeoffs exist
7. Appends a log entry to `design_log.md`
8. Commits with a descriptive message

The user reviews the preview. If good, says "keep it". If not, says "undo" or
"try a narrower version". No manual file editing required.

### Example conversation

```
User: The tail feels too wide for pumping. Narrow it by about 3cm,
      keep the outline smooth.

Claude: Narrowing tail from 38.2cm to 35.2cm. This tightens the squash
        tail outline, which should improve pump sensitivity. Tradeoff:
        slightly less stability at low speed. Rail volume stays the same.
        [generates preview] → design_preview.html

User: Nice. Also increase the nose rocker slightly, maybe 15→17mm.

Claude: Nose rocker increased from 12.8 to 14.8mm (closer to your
        target, staying within the stringer curve). More lift entry,
        marginally softer rail-to-rail at speed. [generates preview]

User: What's the volume now?

Claude: Estimated volume: 58.2L (down from 60.1L — tail narrowing removed
        ~1.5L, rocker change had minimal effect). For 85kg rider that's
        still well within comfortable range.

User: Run the FEA optimizer on this.

Claude: Writing design to foil-board-optimizer/board_shape.s3dx and
        launching optimization...
```

---

## Design Parameters — What's Modifiable

### Planform / Outline
- **Tail width** — at 1/4 board, 1/12 board, tail tip
- **Nose width** — at 1/4, 1/12 from nose
- **Maximum width** — position and value
- **Outline curve** — "wider in the hips", "more parallel rails", "pulled-in nose"
- **Tail shape** — squash, swallow, round, pin (requires understanding tail block geometry)

### Rocker
- **Nose rocker** — total and curve shape (early/late kick)
- **Tail rocker** — total and entry point
- **Overall flip** — "add 2mm rocker throughout"

### Thickness / Volume
- **Thickness distribution** — where the board is thickest, how it tapers
- **Rail thickness** — thicker/thinner rails
- **Overall volume** — scale thickness uniformly to target a specific volume

### Bottom Contours (complex — phase 2)
- **Concave depth** — VConcave settings
- **Chine shape** — Chine Upper Edge curve
- **Single vs double concave** — transition point

### Hardware Placement
- **Footstrap positions** — move front/back/centre straps (modifies Box x/y position)
- **Leash plug position**
- **Twin Track position** — longitudinal (trim adjustment)

### Out of scope (for Shape3D, not here)
- Fin box size/type — structural geometry, best done in Shape3D
- Glass schedule, material properties — not in s3dx
- Detailed rail cross-section shape — requires surface patch editing, too risky to automate

---

## Technical Architecture

### File structure

```
board-design/
├── PRD.md                    # This document
├── CLAUDE.md                 # Agent instructions for design sessions
├── design.s3dx               # Working design file (modified by Claude)
├── design_log.md             # Append-only log of all changes + rationale
├── preview.html              # 3D viewer, regenerated after each change
├── boardcad/
│   ├── parser.py             # s3dx XML → Python data model
│   ├── bezier.py             # Bezier evaluation, modification, re-fitting
│   ├── writer.py             # Python data model → s3dx XML (roundtrip)
│   ├── preview.py            # Generate three.js HTML viewer from design
│   ├── metrics.py            # Compute volume, area, rocker from geometry
│   └── transforms.py        # High-level design operations (narrow_tail, etc.)
├── profiles/
│   └── reference/
│       └── TT60.s3dx         # Original — never modified
└── variants/                 # Named saves: "TT60_narrow_tail.s3dx", etc.
```

### Core modules

#### `parser.py` — s3dx → data model
Parse the Shape3D XML into a structured Python object:
- `Board` with named attributes (length, width, thickness, rocker, volume)
- `Outline` — the Otl Bezier control points (planform shape)
- `StringerBot` / `StringerDeck` — rocker/camber lines
- `SurfaceCurves` — list of named Bezier3d curve definitions
- `Boxes` — hardware placement (position, orientation, dimensions)

The format is already well-understood from the foil-design project's s3dx work.

#### `bezier.py` — Bezier math
- Evaluate a degree-3 Bezier at parameter t → 3D point
- Dense-sample a Bezier → point cloud
- Fit a Bezier to a target point cloud (least-squares CST fit or direct CP optimization)
- Move a subset of control points while preserving C1 continuity at boundaries
- Compute arc length, area under curve

#### `transforms.py` — design intent → Bezier operations
The intelligence layer. Translates high-level intent into Bezier modifications:

```python
def narrow_tail(board, delta_cm, taper="smooth"):
    """Narrow tail width by delta_cm, preserving outline smoothness."""
    # Identify tail control points in Otl (x > board.length * 0.85)
    # Scale y-coordinates inward by delta_cm/2 (symmetric)
    # Re-fit while fixing nose-side CPs, propagating C1 constraints

def set_nose_rocker(board, target_mm):
    """Set nose rocker to target_mm, adjusting stringer curve."""

def move_footstrap(board, box_name, dx_cm, dy_cm):
    """Translate a hardware box position."""

def scale_volume(board, target_litres):
    """Scale thickness distribution to achieve target volume."""

def estimate_volume(board):
    """Numerical integration over cross-sections → volume in litres."""
```

#### `writer.py` — data model → s3dx XML
Roundtrip write-back. Critical constraint: the output must be valid Shape3D XML
that opens correctly in Shape3D X. Strategy: use the parsed XML tree as a
template, replacing only the modified control point coordinates while leaving
all metadata, license fields, and structural XML intact.

#### `preview.py` — 3D viewer generation
Generate a self-contained HTML file using three.js:
- Evaluate all surface Bezier patches to a dense mesh
- Render top, bottom, rails as a shaded 3D model
- Overlay hardware box positions
- Show key dimensions (length, width, thickness, rocker) as annotations
- Orbit/zoom controls
- Single file, opens instantly in browser — no server needed

This extends the existing `viewer.html` from the foil-board-optimizer project.

#### `metrics.py` — computed properties
After every modification, recompute:
- Volume (numerical integration across cross-sections)
- Surface area (projected and developed)
- Nose rocker / tail rocker (stringer endpoints vs midpoint)
- Width at key stations (1/4, 1/2, 3/4 board length)
- Estimated weight (volume × assumed density, configurable)

These are printed after every change and appended to `design_log.md`.

---

## Preview System

After every design change, Claude generates `preview.html` and opens it in the
browser. The viewer shows:

- **3D shaded board** — orbitable with mouse
- **Outline view** — top-down flat view with dimension callouts
- **Rocker profile** — side view stringer curve
- **Cross-sections** — selectable spanwise slices showing thickness profile
- **Hardware overlay** — footstrap, leash, track positions as coloured markers
- **Diff highlight** — changed regions shown in a different colour vs previous
  version (compare `design.s3dx` to last git commit)

The preview is purely cosmetic — it doesn't need to be perfect, just clear
enough for the user to assess "yes that looks right" or "the nose is weird".

---

## Design History

Every change is captured in two places:

**`design_log.md`** — human-readable append-only log:
```
## 2026-03-10 14:32 — Narrow tail
- Change: Tail width 38.2 → 35.2 cm (−3cm)
- Rationale: User wants better pump response
- Tradeoffs: ~1.5L volume reduction, slightly lower low-speed stability
- Volume: 58.2L (was 60.1L)
- Rocker: unchanged
```

**Git commits** — every kept change is committed:
```
Narrow tail −3cm for pump response

Tail width: 38.2 → 35.2 cm
Outline control points adjusted in Otl, preserving C1 continuity.
Volume: 60.1 → 58.2L
```

The user can `git diff`, `git log`, or `git checkout` any previous state.
Named variants are saved to `variants/` with descriptive names.

---

## Integration with FEA Optimizer

When the user says "run the optimizer" or "check the structure":

1. Claude copies `design.s3dx` to `../foil-board-optimizer/board_shape.s3dx`
2. Updates the board geometry in `foilopt/geometry/board.py` to match the new
   dimensions (length, width, thickness, mast mount position)
3. Runs `python optimize.py` (or `python modal_run.py` for cloud)
4. Returns the compliance score and an STL of the optimal internal structure
5. The user can view both the shape preview and the FEA result side-by-side

This closes the loop: design the shape, verify the structure, iterate.

---

## CLAUDE.md (agent instructions)

When a Claude Code session opens in `board-design/`:

```
You are a foil board design assistant with access to Shape3D geometry files.

The working design is design.s3dx. The original reference is profiles/reference/TT60.s3dx.
Never modify the reference file.

For each user request:
1. Read design.s3dx to understand current state
2. Identify which geometric parameters need to change
3. Use boardcad/transforms.py to apply the change
4. Recompute metrics (volume, rocker, width stations)
5. Write the updated design.s3dx
6. Generate preview.html and open it in the browser
7. Explain what changed, what the tradeoffs are, what stayed the same
8. Append to design_log.md
9. Commit with a descriptive message

Speak like a shaper, not an engineer. Say "tail is sitting about 2cm narrower"
not "Y-coordinates of control points 27-31 reduced by 10mm each".

When uncertain whether a change is feasible, explain why and ask.
When a change would break symmetry or produce invalid geometry, warn the user.
```

---

## Development Phases

### Phase 1 — Foundation (build first)
- `parser.py` — read TT60.s3dx into a Python data model, print all key dimensions
- `writer.py` — roundtrip: parse → write → verify file opens in Shape3D
- `metrics.py` — compute volume, rocker, width stations from parsed geometry
- `preview.py` — basic 3D HTML viewer of the outline and stringer

**Test:** Parse TT60, print "Length: 164cm, Width: 49.5cm, Volume: ~60L", write back,
confirm the file is byte-for-byte equivalent except whitespace.

### Phase 2 — Planform manipulation
- `transforms.py` — `narrow_tail`, `set_nose_rocker`, `scale_volume`
- Git workflow in CLAUDE.md
- Diff highlighting in preview

**Test:** Narrow tail by 3cm, verify outline is smooth, verify volume decreased
proportionally, write back, open in Shape3D, visually confirm.

### Phase 3 — Hardware and full session
- `transforms.py` — box/footstrap placement
- `design_log.md` automation
- Full conversational session end-to-end
- FEA integration

### Phase 4 — Bottom contours (complex)
- Concave, chine, vee — these require surface patch editing, not just outline
- Low priority: high complexity, high risk of producing invalid geometry

---

## Success Criteria

1. **Roundtrip fidelity** — Parse TT60.s3dx → write back → file opens correctly in Shape3D with no visible differences
2. **Natural language control** — User can say "narrow the tail 3cm" and get a geometrically correct, smooth result without touching the XML
3. **Preview legible** — The HTML viewer clearly shows the board shape and key dimensions after each change
4. **Traceable history** — Every design decision is recorded in `design_log.md` and git
5. **FEA pipeline** — The modified .s3dx can flow directly into the foil-board-optimizer for structural analysis
6. **Shape3D compatible** — Output files can be opened, viewed, and further edited in Shape3D X

---

## Open Questions

1. **Roundtrip fidelity risk** — Shape3D's XML parser may be sensitive to precision, whitespace, or encoding. Need to test write-back before building anything else.

2. **Bottom contour complexity** — The Chine, concave, and rail cross-section curves interact in ways that may be too complex to modify safely without Shape3D's constraint system. Scope to outline-only for Phase 1-2.

3. **Volume calculation** — Shape3D computes volume internally. We'll need to verify our numerical integration matches Shape3D's reported value before using it as a metric.

4. **Which tail type?** — The TT60 has a squash tail. Supporting swallow tail, pin, round tail requires different outline topology (different number of control points). Phase 1 should only support same-topology modifications.

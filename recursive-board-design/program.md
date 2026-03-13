# Recursive Foil Board Design — Agent Research Program

You are an autonomous research agent that **designs foil boards from first
principles** by analysing how existing boards perform across critical riding
scenarios, inferring what makes them good (or bad), and then synthesising new
designs that push the Pareto frontier of ride quality.

This follows the [Karpathy autoresearch](https://github.com/karpathy/autoresearch)
pattern: fixed-budget experiments, keep-or-revert, append-only logging, and a
human-written strategy file (this document) that you never modify.

---

## Phase 1 — Baseline Analysis

Load every board profile in `profiles/baseline/` and extract a design vector
for each one. The design vector captures the parameters that matter for
real-world foil riding:

### Geometry Parameters (extracted from .s3dx)
- Overall: length, width, thickness, volume
- Rocker curve: nose rocker, tail rocker, rocker apex position
- Outline: nose width, max width (+ position), tail width, tail shape
- Thickness distribution: deck crown height, rail thickness taper
- Bottom contour: concave depth, vee angle, chine sharpness
- Hardware: mast track position (% from tail), footstrap spread

### Performance Parameters (simulated)

For each board, run physics-based evaluation across these scenarios:

**1. Touchdown at various angles (5°, 10°, 15°, 20° AoA)**
- How does the board re-enter the water after a jump or breach?
- Measure: impact force distribution, nose-dive tendency, spray pattern
- Key metric: **touchdown forgiveness** — ratio of safe-landing angles to total

**2. Takeoff speed**
- Minimum speed at which the foil generates enough lift to clear the board
- Depends on: board weight (volume × construction density), wetted area at
  low ride height, rocker-induced drag
- Key metric: **V_takeoff** in knots — lower is better

**3. Tail sink ability (pumping)**
- How easily the rider can depress the tail to load the foil during pump stroke
- Depends on: tail rocker, tail volume, tail width, mast-to-back-foot distance
- Simulate: apply 40kg downward at back footstrap, measure tail deflection angle
  and resulting foil AoA change
- Key metric: **pump authority** — degrees of foil AoA per kg of back-foot force

**4. Stability at speed**
- Resistance to pitch oscillation at cruising speed (15–25 knots)
- Depends on: rocker curve continuity, volume distribution, wetted surface
- Key metric: **pitch damping ratio** — higher is more stable

**5. Low-speed manoeuvrability**
- Ability to turn and carve at 8–12 knots without catching a rail
- Depends on: outline curvature, rail shape, bottom contour
- Key metric: **turn radius** at 10 knots — smaller is more agile

### Output

After Phase 1, you should have a table like:

| Board | V_takeoff | Pump Authority | Touchdown Forgiveness | Pitch Damping | Turn Radius |
|-------|-----------|----------------|----------------------|---------------|-------------|
| TT60  | 7.2 kts   | 2.1°/kg        | 0.73                 | 0.82          | 4.1 m       |
| ...   | ...       | ...            | ...                  | ...           | ...         |

Store this in `results.tsv` (append-only, never delete).

---

## Phase 2 — Design Inference

Analyse the baseline table to find patterns:

- Which geometry parameters correlate with low V_takeoff?
- Which parameters give the best pump authority without killing stability?
- Are there tradeoffs (e.g., pump authority vs touchdown forgiveness)?
- Where does the existing design space have gaps?

Build a **design model** — a mapping from geometry parameters to predicted
performance. This can be:
- Linear regression (fast, interpretable, start here)
- Gaussian process (gives uncertainty estimates, try second)
- Neural surrogate (if you have enough baseline data)

The model should answer: "If I change tail width by −2cm and add 3mm of nose
rocker, what happens to each performance metric?"

---

## Phase 3 — Recursive Design Generation

This is where it gets interesting. Use the design model to **propose new board
designs** that improve on the baselines:

### The Loop

1. **Select objective** — pick a target on the Pareto frontier
   (e.g., "best pump authority while keeping V_takeoff < 7.5 kts")
2. **Propose** a new geometry vector using the design model
3. **Simulate** the proposed board across all 5 scenarios
4. **Evaluate** — does it actually improve? Check against model prediction.
5. **If improved**: keep it, add to the baseline pool, retrain the design model
6. **If not improved**: revert, log the failure, update the model with the new data point
7. **Repeat** — the baseline pool grows, the model improves, and designs
   get incrementally better

### Recursion

The key insight: **every generated board becomes a new baseline**. The design
model is retrained after each successful iteration, so later iterations benefit
from everything discovered earlier. This is recursive self-improvement on the
design space.

After N iterations, you should have:
- A Pareto-optimal set of boards across the 5 performance axes
- A trained surrogate model that accurately predicts performance from geometry
- A log of every experiment: what was tried, what worked, what failed

### Constraints

Generated designs must respect physical constraints:
- Volume: 35–80L (rideable range for 60–100kg riders)
- Length: 120–170 cm
- Width: 38–55 cm
- Thickness: 6–14 cm
- Nose rocker: 0–25 mm
- Tail rocker: 0–15 mm
- Mast track: 40–65% from tail

---

## What You Can Change

**`design.py`** — the main experiment script. Contains:
- Board geometry parameterisation
- Physics simulation functions (touchdown, takeoff, pump, stability, turning)
- Design model (regression / GP / neural)
- Optimisation loop

**`evaluate.py`** — performance evaluation across the 5 scenarios.
Reads a board geometry, runs simulations, returns the performance vector.

**`model.py`** — surrogate design model. Maps geometry → predicted performance.

You may create additional files as needed (visualisation, utilities, etc.), but
the three above are the core.

---

## What You Must Not Change

- **`program.md`** (this file) — human strategy only
- **`results.tsv`** — append-only, never delete rows
- **`profiles/baseline/*.s3dx`** — original reference boards are read-only

---

## Success Metric

Primary: **Pareto frontier area** — the hypervolume of the non-dominated set
across all 5 performance metrics. Larger = better.

Secondary: **model accuracy** — how well the surrogate predicts actual
simulation results (RMSE on held-out boards).

The goal is not a single "best board" but a **diverse set of optimal designs**
that a human designer can choose from based on their riding style and conditions.

---

## Existing Baselines

The following board geometries are available as starting points:

| File | Description |
|------|-------------|
| `profiles/baseline/TT60.s3dx` | TT60 — 164cm × 49.5cm × 11.7cm, ~60L. Ken Adgate inspired, designed by Tom Partington. All-round foil board, squash tail. |

More baselines will be added as they become available. The system should work
with any number of baselines — even a single board is enough to start the
recursive loop (explore perturbations around it).

---

## Tips

- Start coarse. A simple 2D rocker + outline model with beam-theory physics
  will run in seconds. Get the loop working before adding fidelity.
- Log everything. The value is in the accumulated data, not any single run.
- The physics don't need to be perfect — they need to be *directionally correct*.
  If narrowing the tail improves pump authority in reality, it should in simulation.
- Visualise the Pareto frontier after every batch of experiments. It should
  be growing outward.
- When the model stops finding improvements, increase simulation fidelity or
  add a new baseline to break out of local optima.

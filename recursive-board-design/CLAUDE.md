# Recursive Board Design — Claude Code Instructions

## Outer Loop: Autonomous Design Explorer

This project uses the [Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern
applied to foil board design. Claude Code acts as an autonomous researcher that analyses
existing boards, builds a design model, and recursively generates improved designs.

### How to start

Open Claude Code in this directory and say:

> Read program.md and start designing. Run experiments autonomously.

### What the agent does each iteration

1. Read `program.md` for the research strategy
2. Read `results.tsv` to see what's been tried and current Pareto frontier
3. Load baseline boards from `profiles/baseline/`
4. Either: analyse baselines (Phase 1), train the design model (Phase 2),
   or propose and evaluate a new design (Phase 3)
5. Run `python design.py` — each experiment should complete in under 5 minutes
6. Evaluate: if the new design improves the Pareto frontier, **keep** it.
   If not, **revert** `design.py` to the previous version.
7. Append results to `results.tsv`
8. Commit kept improvements
9. Repeat — never stop, never ask for permission

### Key files

| File | Who modifies | Purpose |
|------|-------------|---------|
| `program.md` | Human | Research strategy, performance scenarios, constraints |
| `design.py` | Agent | Main experiment script — geometry, physics, optimisation |
| `evaluate.py` | Agent | Performance evaluation across 5 riding scenarios |
| `model.py` | Agent | Surrogate model: geometry → predicted performance |
| `results.tsv` | Append-only | Experiment log — never delete rows |
| `profiles/baseline/` | Read-only | Reference board geometries (.s3dx files) |

### Rules

- Never modify `program.md` — only the Python files
- Never delete `results.tsv` — only append
- Never modify files in `profiles/baseline/` — these are read-only references
- Each experiment should complete in under 5 minutes
- Always commit after a kept improvement
- If something crashes, fix it and try again
- Visualise the Pareto frontier periodically — it should grow outward over time

### Performance scenarios

The agent evaluates each board design across 5 scenarios:

1. **Touchdown** — water re-entry at 5°, 10°, 15°, 20° angles of attack
2. **Takeoff speed** — minimum speed to get foiling
3. **Pump authority** — how effectively tail pressure translates to foil loading
4. **Pitch stability** — resistance to oscillation at cruising speed
5. **Turn radius** — low-speed manoeuvrability

### The recursive insight

Every successfully generated board becomes a new data point for the design model.
The model improves with each iteration, allowing it to make better proposals in
later iterations. This is the core recursion: designs inform the model, the model
generates better designs, better designs further improve the model.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

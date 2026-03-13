# Foil Experiments

AI-powered experiments in surf hydrofoil design. Each project runs autonomously using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — you describe what to explore, and the AI runs experiments while you're away from the keyboard.

No programming experience required. Browse the experiments and demos at **[dwfoil-com.github.io/foil-experiments](https://dwfoil-com.github.io/foil-experiments/)**.

---

## Getting Started with Claude Cowork (easiest)

The fastest way to get going — no terminal experience needed. [Claude Cowork](https://claude.com/product/cowork) is a desktop companion in Claude Desktop that can clone repos, install dependencies, and run experiments for you.

1. **Get a Claude plan** — [Claude Pro](https://claude.com/pricing) is the best-value starting point. Move to Max 5× only if you need longer or more frequent sessions.
2. **Open Claude Cowork** — Launch [Claude Desktop](https://claude.ai/download) and start a Cowork session.
3. **Ask it to set up the repo** — Tell Cowork: *"Clone dwfoil-com/foil-experiments, install what the repo needs, and run the viewers I want to explore."*

Cowork handles the git clone, Python environment, and dependency installs. Once you're oriented, switch into a tighter optimization loop: keep the agent focused on one narrow objective, review the results, refine the target, and continue iterating.

> See the [Cowork getting-started guide](https://support.claude.com/en/articles/13345190-get-started-with-cowork) for more detail.

---

## Getting Started with Claude Code (terminal)

If you're comfortable in a terminal, you can drive experiments directly with Claude Code.

### 1. Install Claude Code

```bash
# Requires Node.js — install from https://nodejs.org first
npm install -g @anthropic-ai/claude-code
```

Sign up at [console.anthropic.com](https://console.anthropic.com) and add an API key when prompted on first run.

> If you have a Claude Max subscription, Claude Code is included — just run `claude` and log in.

### 2. Clone and set up

```bash
git clone https://github.com/dwfoil-com/foil-experiments.git
cd foil-experiments

# Set up a project (each has its own requirements.txt)
cd foil-board-optimizer  # or foil-rl-pump/python-rl
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Claude Code

```bash
claude
```

That's it. Claude reads the project's `CLAUDE.md` and `program.md` files to understand what to do.

---

## Running the Autonomous Outer Loop

This is the core idea: the [Karpathy autoresearch pattern](https://github.com/karpathy/autoresearch). You define a research strategy in plain English (`program.md`), and Claude Code runs experiments in a loop — modifying parameters, running simulations, evaluating results, keeping improvements, reverting failures, and repeating. Overnight if you want.

### Start an experiment loop

Navigate into a project directory and launch Claude Code:

```bash
cd foil-board-optimizer
claude
```

Then type:

> Read program.md and start optimizing. Run experiments autonomously.

Claude will:
1. Read the research strategy from `program.md`
2. Check past results in `results.tsv`
3. Form a hypothesis, modify experiment parameters
4. Run the simulation (`python optimize.py`)
5. Evaluate: if stiffness improved, keep the change. If not, revert.
6. Commit the improvement and loop back to step 2

You can walk away. Come back and check `results.tsv` to see what it found.

### Steer the research

Edit `program.md` in plain English to change direction:

- "Focus on jump landing loads — I want a board that survives impacts"
- "Try lower volume fractions, I want the lightest possible board"
- "Only optimize for riding and pumping, ignore carving"

Claude reads this file at the start of each loop and adjusts its approach.

---

## GPU Acceleration with Modal (optional)

For bigger, higher-resolution experiments, you can run simulations on cloud GPUs via [Modal](https://modal.com). The free tier gives you enough credits to get started.

### Setup

```bash
pip install modal
modal setup  # opens browser to create account + authenticate
```

### Run on cloud

```bash
# High-resolution optimization on Modal cloud
python modal_run.py --nelx 70 --nely 25 --nelz 10 --max-iter 200
```

Modal spins up a machine, runs the simulation, and sends results back to your laptop. No server management needed.

---

## Projects

### [board-design](board-design/) — Conversational CAD for foil boards
**Status:** planning

A Claude Code session as a CAD collaborator. Describe what you want ("narrow the tail 3cm", "add more nose rocker") and Claude modifies the Shape3D file, shows a 3D preview, commits the change with rationale, and feeds the result into the FEA optimizer. Baseline design is a TT60 foil board by Tom Partington.

### [recursive-board-design](recursive-board-design/) — Recursive foil board design
**Status:** planning

Autonomous foil board design through recursive self-improvement. Load existing board geometries as baselines, simulate performance across critical riding scenarios (touchdown at various angles, takeoff speed, pump authority, pitch stability, turning), infer optimal design parameters, then recursively generate new boards that push the Pareto frontier. Each successful design feeds back into the model, making the next iteration smarter. Inspired by the Karpathy autoresearch pattern.

### [foil-design](foil-design/) — Hydrofoil wing optimization
**Status:** planning

Autonomous optimization of hydrofoil wing and stabilizer shapes using NeuralFoil (ML replacement for XFoil) and AeroSandbox (3D vortex lattice). Includes real Shape3D reference geometry from a K2 800 front wing and stabilizer. Optimizes profile shape, planform, and front-wing/stab interaction.

### [foil-board-optimizer](foil-board-optimizer/) — Board structure topology optimization
**Status:** in-progress

Autonomous topology optimization of hydrofoil board internal structures. Uses FEA simulation and SIMP optimization to design internal ribs and lattices that maximize stiffness while minimizing weight. The AI outer loop explores the parameter space — mesh resolution, material properties, load cases, volume fractions — and discovers optimal structures that a human designer wouldn't conceive. Exports 3D-printable STL files.

**Start here** if you want to see the autoresearch pattern in action.

### [foil-insert-load-comparison](foil-insert-load-comparison/) — Tuttle vs dual-track insert comparison
**Status:** poc

Side-by-side structural comparison of a Tuttle box and a regular dual-track mast cassette in the same TT60 foil board. The project holds the outer board shape fixed and changes only the insert architecture, bond path, and local HD foam geometry.

Includes a browser-based 3D viewer on the actual TT60 board shape with adjustable insert dimensions and track footprint shapes, plus a reduced-order structural solve that updates the shell field after each parameter change.

### [foil-rl-pump](foil-rl-pump/) — RL pump foil simulation
**Status:** mvp

Reinforcement learning agent that learns to pump foil — maintaining flight through coordinated body movements. The trained agent achieves 60+ second sustained flights with realistic 2.3Hz pumping frequency. Includes training scripts, pre-trained models, and video generation.

### [foil-mocap](foil-mocap/) — Motion capture from foil videos
**Status:** mvp

Extract body pose from pump foil videos using MediaPipe. Produces skeleton overlays, dot traces, and biomechanical features from any foil video.

### [hydrofoil](hydrofoil/) — Interactive surf hydrofoil explainer
**Status:** in-progress

Long-form interactive article and simulation exploring how surf hydrofoils work — lift, drag, trim, stance, pumping, and more. Inspired by [ciechanow.ski/airfoil](https://ciechanow.ski/airfoil/).

---

## Contributing

Contributions are welcome — open an issue or pull request. The easiest way to get started is to use [Claude Cowork](https://claude.com/product/cowork) to clone the repo and explore. Each project has its own `CLAUDE.md` with context for the AI to work from.

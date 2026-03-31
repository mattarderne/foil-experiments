# Ocean Wave Simulation — Spec

## Objective

Build a physically accurate ocean wave simulation focused on **wave train sets and group velocity** — the key phenomena that govern downwind foiling but are almost universally missing from existing real-time ocean simulations.

The goal is not just visual realism but **physical representativeness**: wave energy must propagate at the correct group velocity, wave trains must form coherent sets, and a rider/object should be able to interact with these propagating energy packets in a way that matches real open-ocean experience.

### Why this matters for downwind

In downwind foiling, riders connect between "bumps" — coherent wave groups that propagate through open water. The skill is reading which bumps are building (energy arriving) vs dying (energy departing). This requires:

1. **Wave trains** — coherent groups of waves moving together
2. **Group velocity** — energy propagates at half the phase velocity in deep water (`v_g = v_p / 2`), so wave crests appear at the back of a group, move through it, and disappear at the front
3. **Set waves** — periodic arrival of larger wave groups from interference of swell trains with slightly different frequencies
4. **Wave-wave interaction** — multiple swell directions creating constructive/destructive interference patterns

Most ocean simulations (Tessendorf FFT, Gerstner waves) produce statistically plausible snapshots but **do not propagate energy correctly** — they look like ocean but don't behave like ocean.

---

## Background & Inspiration

### Twitter / X Threads

- [Matt Arderne on Three.js Water Pro replication](https://x.com/mattarderne/status/2025839142761041968) — Tried replicating real water in Three.js Water Pro, got close visually but identified the key gap: *"The issue you have imo is you don't have the wave train sets. There is a distinct 'set' moving through the top of the screen, and I never see them reliably in sims."*

- [Matt Arderne / Dan Greenheck conversation on wave trains](https://x.com/mattarderne/status/2014984126080667748) — Discussion about simulating wave train interactions in open water for foil boarding. Dan Greenheck (Three.js Water Pro author) suggests starting with two sine waves with sliders for frequency and phase velocity to get wave groups. Key insight: the empirically experienced wave interactions while riding are hard to find in any simulation.

### Reference Videos

- [Downwind Bump Reading Tutorial](https://www.youtube.com/watch?v=yPfagLeUa7k) — Demonstrates the wave reading skill required for downwind foiling. The water looks chaotic but is actually quite orderly — wave trains and sets create predictable patterns that experienced riders exploit.

### Key Physics Reference

- [Group velocity — Wikipedia](https://en.wikipedia.org/wiki/Group_velocity) — The group velocity is `v_g = dω/dk`. For deep water gravity waves, the dispersion relation is `ω² = gk`, giving `v_g = v_p / 2`. This means wave energy travels at half the speed of individual wave crests.

---

## The Gap: Why Existing Simulations Fail

### What they do well
- Generate statistically representative ocean surfaces (correct power spectra)
- Look visually convincing (wave shapes, foam, lighting)
- Run in real-time with GPU acceleration

### What they miss
- **No energy propagation** — Tessendorf FFT generates each frame independently from spectral components. There is no concept of a wave packet carrying energy from point A to point B.
- **No wave groups** — Because each frame is a fresh spectral realization, you never see coherent wave groups forming, propagating, and dissipating.
- **No set waves** — The periodic arrival of larger wave groups (from interference of close-frequency swell trains) does not emerge.
- **No spatial coherence of energy** — In reality, if you're at point A and see a big set arrive, an observer 100m downwind will see that same set arrive later. This causality is absent in standard approaches.

### Why this happens
The Tessendorf method evaluates:

```
h(x, t) = Σ_k ĥ(k, t) · exp(ik·x)
```

where `ĥ(k, t) = ĥ₀(k) · exp(iω(k)t) + ĥ₀*(-k) · exp(-iω(k)t)`

The dispersion relation `ω(k) = √(gk)` is correct, and in principle wave groups **should** emerge from interference of nearby frequency components. The problem is practical:

1. **Tiling** — FFT surfaces tile periodically, destroying long-range coherence
2. **Spectral resolution** — Limited grid size means coarse frequency resolution, preventing the closely-spaced frequencies needed for visible wave groups
3. **No spatial windowing** — Energy is uniform across the domain rather than localized in propagating packets

---

## Technical Approach

### Architecture Options

There are two viable approaches. Both correctly model group velocity; they differ in representation.

#### Option A: Lagrangian Wave Packets (Water Wave Packets method)

Based on Jeschke & Wojtan (SIGGRAPH 2017). Wave energy is carried by Lagrangian particles, each representing a wave packet:
- Each packet has position, wavelength, amplitude, and direction
- Packets propagate at **group velocity** `c_g = dω/dk`
- Surface is reconstructed by summing packet contributions at each render point
- Naturally handles dispersion, diffraction, refraction, reflection
- Energy conservation guaranteed by construction
- Source code available: [github.com/jeschke/water-wave-packets](https://github.com/jeschke/water-wave-packets)

**Pros:** Explicit group velocity, intuitive control, parallelizable, proven in research.
**Cons:** C++/DirectX (would need porting), particle count scales with domain, surface reconstruction cost.

#### Option B: Hybrid NLS Envelope + FFT Surface

The simulation combines two layers:

##### Layer 1: Wave Envelope Model (Physics Core)
Solves the **nonlinear Schrödinger equation (NLS)** or a simplified envelope equation to track wave group propagation:

```
i ∂A/∂t + c_g ∂A/∂x + (c_g'/2) ∂²A/∂x² + α|A|²A = 0
```

Where `A(x, t)` is the complex wave envelope, `c_g` is the group velocity, and the nonlinear term captures wave steepening and modulational instability (Benjamin-Feir instability).

This gives us:
- Correct group velocity propagation
- Wave train formation and interaction
- Set wave emergence from multi-swell interference
- Spatial coherence (energy packets move through the domain)

**Pros:** Continuous field (no particle artifacts), natural multi-swell superposition, connects to rich NLS physics (modulational instability, rogue waves).
**Cons:** More complex math, 2D extension is nontrivial, less intuitive than packets.

##### Layer 2: Surface Realization (Rendering)
Uses the envelope from Layer 1 to modulate a local FFT or Gerstner wave synthesis:

```
h(x, t) = Re[A(x, t) · exp(i(k₀x - ω₀t))] + high-frequency detail
```

The envelope controls the local wave amplitude, while FFT/Gerstner provides the fine-scale wave texture within each group.

##### Layer 3: Interaction Model
Physics coupling for objects (foil board, rider) interacting with the wave field:
- Local wave slope and orbital velocity from the surface realization
- Energy exchange (wave forcing on object, wake generation)
- Buoyancy and hydrodynamic forces

### Multiple Swell Trains

Real ocean conditions have multiple independent swell trains from different storms. The simulation should support:

- **N independent swell trains**, each with direction, period, and amplitude
- Each train has its own envelope propagation
- The surface is the superposition of all trains
- Interference between trains creates the "set" patterns experienced in real conditions

### Domain & Scale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Domain size | 1-5 km | Downwind runs are typically 5-15 km |
| Resolution | ~1m | Need to resolve individual waves (8-15s period = 100-350m wavelength) |
| Time step | ~0.1s | Resolve wave crests passing |
| Swell periods | 6-18s | Typical ocean swell range |
| Wind speed | 15-30 kts | Typical downwind conditions |

---

## Existing Libraries & Tools Evaluated

### Visual / Real-Time Ocean Rendering

| Library | Platform | Approach | Group Velocity? | Notes |
|---------|----------|----------|-----------------|-------|
| [Three.js Water Pro](https://docs.threejswaterpro.com/) | Three.js/WebGPU | FFT + Gerstner | No | Best-in-class visual quality. $99 commercial. Could serve as rendering frontend. |
| [GarrettGunnell/Water](https://github.com/GarrettGunnell/Water) | Unity/C# | FFT (Tessendorf) + Gerstner + JONSWAP | No | MIT license. Good educational reference. Dual JONSWAP with 4 frequency bands. |
| [jbouny/ocean](https://github.com/jbouny/ocean) | Three.js/WebGL | FFT (Tessendorf) | No | Open source Three.js ocean. Older but functional. |
| [Postrediori/OceanSimulation](https://github.com/Postrediori/OceanSimulation) | C++/OpenGL | FFT (Phillips spectrum) | No | Clean C++ implementation. MIT license. Good reference for FFT pipeline. |
| [Ocean Simulation with FFT and WebGPU](https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/) | WebGPU | FFT | No | Blog post with implementation details. |

### Physics / Scientific Wave Models

| Library | Platform | Approach | Group Velocity? | Notes |
|---------|----------|----------|-----------------|-------|
| [SWAN](https://swanmodel.sourceforge.io/) | Fortran | Spectral action balance | **Yes** | Industry standard nearshore wave model. Solves action balance equation with `c_g`. Overkill but physically correct. |
| [WAVEWATCH III](https://github.com/NOAA-EMC/WW3) | Fortran | Spectral action balance | **Yes** | NOAA global wave model. Correct group velocity propagation. Very heavy. |
| [Veros](https://github.com/team-ocean/veros) | Python/JAX | Primitive equations | No (wrong scale) | Ocean circulation model — km-scale, not wave-scale. Could provide background currents. |
| [pakodekker/oceansar](https://github.com/pakodekker/oceansar) | Python/Numba | Linear wave theory | **Implicitly yes** | SAR ocean simulation. The surface generation module uses correct dispersion, so groups emerge from superposition. GPL-3.0. |

### Wave Packet / Envelope Methods (Key Category)

| Library | Platform | Approach | Group Velocity? | Notes |
|---------|----------|----------|-----------------|-------|
| [Water Wave Packets](https://github.com/jeschke/water-wave-packets) | C++/DirectX | Lagrangian wave energy packets | **Yes — explicit** | **SIGGRAPH 2017** (Jeschke & Wojtan). Each particle carries a wave energy packet that propagates at group velocity. Handles dispersion, diffraction, refraction, reflection. Energy conserving. Most directly relevant method for this project. |
| [Water Surface Wavelets](https://dl.acm.org/doi/10.1145/3197517.3201336) | Paper | Wavelet-based extension | **Yes** | **SIGGRAPH 2018** follow-up. Scales to larger domains using wavelets instead of individual packets. |

### NLS / Envelope Solvers

| Library | Platform | Notes |
|---------|----------|-------|
| [simple-idealized-1d-nlse](https://arxiv.org/html/2509.05901v1) | Python | Pseudo-spectral solver for 1D NLS. Split-step Fourier method. Open source. Directly applicable to wave envelope evolution. |
| [Athanassoulis (2024) NCNLS](https://onlinelibrary.wiley.com/doi/10.1111/sapm.12774) | Paper | Nonconservative NLS for wind-forced ocean waves. Relevant physics for wind-swell growth. |
| [gnlse-python](https://gnlse.readthedocs.io/en/latest/gnlse_intro.html) | Python | Generalized NLS solver. Built for optics but same equation. Modular, well-documented. |

### Additional Three.js / Web Ocean Libraries

| Library | URL | Notes |
|---------|-----|-------|
| [jbouny/fft-ocean](https://github.com/jbouny/fft-ocean) | Three.js/WebGL | FFT Tessendorf, 64x64 / 256x256 grid. Older (r72 era). |
| [Mohido/Ocean](https://github.com/Mohido/Ocean) | Three.js | GPU Gems-based JavaScript ocean. |
| [iFFT Ocean Module](https://discourse.threejs.org/t/ifft-ocean-wave-generator-module/51800) | Three.js | Community iFFT module, WebGPU + WebGL2. Actively developed. |
| [Three.js GPGPU Water](https://threejs.org/examples/webgl_gpgpu_water.html) | Three.js | GPUComputationRenderer ripples. Interactive, small-scale. |
| [david.li/waves](http://david.li/waves/) | WebGL | Standalone WebGL ocean demo. |
| [MohamedQatish/BoatPhysics3D](https://github.com/MohamedQatish/BoatPhysics3D) | Three.js | Boat sim with wave interaction and buoyancy. |

### Academic References

- [Tessendorf (2001/2004) — Simulating Ocean Water](https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf) — Foundation paper for FFT ocean rendering
- [Water Wave Packets (SIGGRAPH 2017)](https://dl.acm.org/doi/10.1145/3072959.3073678) — Jeschke & Wojtan. Lagrangian wave packets with explicit group velocity. The most relevant prior art.
- [Water Surface Wavelets (SIGGRAPH 2018)](https://dl.acm.org/doi/10.1145/3197517.3201336) — Extension of wave packets to larger scales
- [Physically accurate real-time ocean waves for maritime simulators](https://www.vliz.be/imisdocs/publications/80/394980.pdf) — Extends Tessendorf toward physical accuracy
- [Foil-Net: Deep Wave Classification for Hydrofoil Surfing](https://link.springer.com/chapter/10.1007/978-3-031-47966-3_9) — ML approach to classifying waves for foil surfing
- [Wave groups — Coastal Dynamics (Bosboom & Stive)](https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/03:_Ocean_waves/3.05:_Wind_wave_generation_and_dispersion/3.5.3:_Wave_groups) — Textbook treatment of wave group physics
- [SWAN Technical Documentation](https://falk.ucsd.edu/modeling/swantech.pdf) — Full mathematical treatment of spectral wave propagation
- [MIT Phase-Resolved Wave Modeling](https://dspace.mit.edu/handle/1721.1/33450) — Direct simulation and deterministic prediction of large-scale nonlinear ocean wave fields

---

## Recommended Implementation Path

### Phase 1: 1D Wave Train Demonstration
**Goal:** Prove that wave groups and sets emerge correctly in 1D.

- Implement 1D linear superposition of N wave components with correct dispersion (`ω = √(gk)`)
- Visualize wave groups propagating at `v_g = v_p / 2`
- Add two swell trains with slightly different periods → observe set wave beating pattern
- Add NLS envelope tracking to verify group velocity matches theory
- **Validation:** Measure group velocity from simulation, compare to `dω/dk`

**Tech:** Python + NumPy. Simple matplotlib or Plotly visualization. This is Dan Greenheck's suggestion ("start with two sine waves with sliders") but done rigorously.

### Phase 2: 2D Surface with Envelope Modulation
**Goal:** Generate a 2D ocean surface where wave groups are visible and propagate correctly.

- Extend to 2D: multiple swell trains with different directions
- Implement envelope extraction (Hilbert transform) to visualize wave groups
- Use envelope-modulated FFT for surface detail
- Add wind-sea on top of swell (local wind waves with broader spectrum)
- **Validation:** Track wave group propagation in 2D, verify against theoretical group velocity for each swell component

**Tech:** Python + NumPy/SciPy for physics. Three.js or similar for 3D visualization. Could use oceansar's surface generation as a starting point.

### Phase 3: Interactive Downwind Simulation
**Goal:** A rider/board interacting with the wave field.

- Couple wave field to a simple rigid-body foil board model (could connect to existing `foil-rl-pump` physics)
- Compute local wave slope, orbital velocity, and pressure at board position
- Render in real-time (WebGPU or similar)
- Allow configuration of swell conditions (period, direction, height for multiple trains)
- **Validation:** Qualitative comparison with downwind foiling video footage

**Tech:** Three.js + WebGPU for rendering. Python backend for physics (or port to WebAssembly/GPU compute). Integration with existing foil physics from `foil-rl-pump/python-rl/foil_env/foil_physics.py`.

### Phase 4: Full Environment
**Goal:** Realistic training/testing environment.

- Wind field modeling (spatial/temporal variation)
- Wave breaking and whitecapping
- Current effects on wave propagation
- Multiple simultaneous swell events from real buoy data (NOAA NDBC)
- Metrics dashboard: wave height, period, group velocity, set frequency
- Connection to RL training environment for autonomous downwind agents

---

## Key Design Decisions

### 1. Why not just use SWAN/WW3?
SWAN and WAVEWATCH III solve the right equations (action balance with group velocity) but are designed for O(km) resolution forecasting, not real-time interactive simulation. They output spectral parameters, not instantaneous surface elevation. We need the physics without the operational forecasting overhead.

### 2. Why NLS over action balance?
The NLS equation directly gives us the wave envelope in space and time, which is exactly what we need — where wave energy is concentrated and how it moves. The action balance equation (SWAN) operates in spectral space and requires reconstruction to get spatial wave groups. NLS is more natural for our "where are the bumps?" question.

### 3. Why not just increase FFT resolution?
In principle, a sufficiently large FFT domain with fine spectral resolution would produce wave groups. But:
- The domain would need to be many km to avoid tiling artifacts
- The spectral resolution would need Δf < 0.001 Hz to resolve wave groups
- This requires FFT grids of 8192+ per side, which is computationally expensive
- And you still don't get proper energy propagation — just a statistically correct snapshot

### 4. Web vs Native?
Start with Python for physics validation (Phase 1-2), then port to web (Three.js + WebGPU) for interactive use (Phase 3-4). The NLS solver is lightweight enough for WebAssembly. Surface rendering is a solved problem (Three.js Water Pro demonstrates this).

---

## Success Criteria

1. **Wave groups visible** — Distinct groups of 5-10 waves with higher amplitude, separated by calmer regions
2. **Correct group velocity** — Measured group velocity within 5% of theoretical `v_g = v_p / 2`
3. **Set waves emerge** — When two swell trains with close frequencies are superimposed, periodic set arrivals are visible
4. **Spatial coherence** — A wave group observed at position A is later observed at position B, with the correct time delay
5. **Downwind-relevant** — A simulated rider can "read" the wave field and find energy to connect between bumps, matching qualitative experience from real downwind sessions
6. **Physically representative** — Wave statistics (significant wave height, peak period, spectral shape) match input swell parameters

---

## Open Questions

- Should we target 2D (plan view) or 3D (perspective) as the primary visualization?
- What level of nonlinearity is needed? Linear superposition may be sufficient for group velocity; NLS adds modulational instability.
- How to handle wave breaking? Important for realism but adds significant complexity.
- Integration path with `foil-rl-pump` — should the wave sim replace or augment the existing simple wave model?
- Real-time performance target? 30fps? 60fps? Or is offline simulation acceptable for Phase 1-2?

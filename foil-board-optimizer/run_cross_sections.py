"""Phase 2: 2D cross-section optimizer for each bulkhead position.

Reads Phase 1 bulkhead positions, runs 2D SIMP at each to find optimal
internal rib/web layout at mm resolution, saves density fields + PNGs.

Usage:
    python run_cross_sections.py
    python run_cross_sections.py --phase1 results/modal_bulkhead5 --strain-energy results/modal_rib
    python run_cross_sections.py --nely 200 --nelz 60 --volfrac 0.30
"""

import argparse
import json
import os
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))


def find_bulkhead_positions(density: np.ndarray, meta: dict, min_solid: int = 90,
                            xmin: float = 0.0, xmax: float = 0.0):
    """Identify X-slice indices to optimize.

    When xmin/xmax are set, return ALL slices in that X range (continuous mode).
    Otherwise, return only slices with solid element count > min_solid.
    """
    nelx = meta["nelx"]
    lx = meta["lx"]
    dx = lx / nelx

    ix_map = np.arange(len(density)) % nelx
    positions = []
    for i in range(nelx):
        x_pos = (i + 0.5) * dx
        if xmin > 0 or xmax > 0:
            # Continuous mode: include all slices in the active X range
            if (xmin <= 0 or x_pos >= xmin) and (xmax <= 0 or x_pos <= xmax):
                sl = density[ix_map == i]
                n_solid = int((sl > 0.99).sum())
                positions.append((i, x_pos, n_solid))
        else:
            sl = density[ix_map == i]
            n_solid = int((sl > 0.99).sum())
            if n_solid > min_solid:
                positions.append((i, x_pos, n_solid))

    return positions  # list of (ix, x_pos, n_solid)


def compute_slice_loads(strain_energy_dir: str, meta: dict):
    """Sum strain energy per X-slice across all load cases."""
    nelx = meta["nelx"]
    ix_map = np.arange(meta["nelx"] * meta["nely"] * meta["nelz"]) % nelx
    se_total = np.zeros(meta["nelx"] * meta["nely"] * meta["nelz"])

    load_cases = [lc["name"] for lc in meta.get("load_cases", [])]
    if not load_cases:
        load_cases = [
            "riding_normal",
            "pumping",
            "jump_landing",
            "carving",
            "front_foot_drive",
            "back_foot_drive",
        ]

    for name in load_cases:
        path = os.path.join(strain_energy_dir, f"{name}.bin")
        if os.path.exists(path):
            se = np.frombuffer(open(path, "rb").read(), dtype=np.float32)
            if len(se) == len(se_total):
                se_total += se

    # Sum per X-slice
    slice_se = np.array([se_total[ix_map == i].sum() for i in range(nelx)])
    return slice_se


def load_mean_mast_force(meta: dict):
    """Compute mean mast force [Fx, Fy, Fz] across load cases."""
    forces = [lc.get("mast_force", [0, 0, 0]) for lc in meta.get("load_cases", [])]
    if not forces:
        return np.array([0.0, 0.0, -785.0])
    return np.mean(forces, axis=0)


def load_peak_vertical_deck_force(meta: dict) -> float:
    """Return the largest downward deck load magnitude across all load cases."""
    peak = 0.0
    for lc in meta.get("load_cases", []):
        if lc.get("front_foot_force") is not None or lc.get("back_foot_force") is not None:
            front = np.asarray(lc.get("front_foot_force") or [0.0, 0.0, 0.0], dtype=float)
            back = np.asarray(lc.get("back_foot_force") or [0.0, 0.0, 0.0], dtype=float)
            peak = max(peak, abs(front[2]) + abs(back[2]))
        elif lc.get("total_force") is not None:
            peak = max(peak, float(lc["total_force"]))
    if peak > 0:
        return float(peak)
    return 2200.0


def plot_result(result, out_path: str, title: str = ""):
    """Save density field as PNG with board outline overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        d = result.density  # (nelz, nely)
        nelz, nely = d.shape

        fig, ax = plt.subplots(figsize=(12, 4))
        cmap = LinearSegmentedColormap.from_list("bw", ["white", "#1a1a2e"])
        im = ax.imshow(
            d,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0, vmax=1,
            extent=[0, result.ly * 1000, 0, result.lz * 1000],
        )
        plt.colorbar(im, ax=ax, label="Density")
        ax.set_xlabel("Y — board width (mm)")
        ax.set_ylabel("Z — board thickness (mm)")
        ax.set_title(title or f"Cross-section at X={result.x_pos:.3f}m")
        ax.set_xlim(0, result.ly * 1000)
        ax.set_ylim(0, result.lz * 1000)

        # Compliance history inset
        if result.compliance_history:
            ax2 = ax.inset_axes([0.75, 0.6, 0.22, 0.35])
            ax2.plot(result.compliance_history, "b-", linewidth=0.8)
            ax2.set_xlabel("iter", fontsize=6)
            ax2.set_ylabel("C", fontsize=6)
            ax2.tick_params(labelsize=5)
            ax2.set_title("compliance", fontsize=6)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available — skipping PNG)")


def main():
    parser = argparse.ArgumentParser(description="Run 2D cross-section optimizers")
    parser.add_argument("--phase1", default="results/modal_bulkhead5",
                        help="Phase 1 results directory")
    parser.add_argument("--strain-energy", default=None,
                        help="Strain energy dir (default: phase1/strain_energy)")
    parser.add_argument("--output", default="results/cross_sections",
                        help="Output directory")
    parser.add_argument("--nely", type=int, default=200,
                        help="2D mesh Y-elements (board width direction)")
    parser.add_argument("--nelz", type=int, default=60,
                        help="2D mesh Z-elements (board thickness direction)")
    parser.add_argument("--volfrac", type=float, default=0.30,
                        help="Target material fraction within board outline")
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=1.5)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--min-solid", type=int, default=90,
                        help="Min solid element count to identify a bulkhead")
    parser.add_argument("--xmin", type=float, default=0.0,
                        help="Run all slices with X >= xmin (continuous mode, overrides --min-solid)")
    parser.add_argument("--xmax", type=float, default=0.0,
                        help="Run all slices with X <= xmax (continuous mode)")
    args = parser.parse_args()

    from foilopt.geometry.board import FoilBoard, BoardShape, load_board_shape
    from foilopt.topology.cross_section import CrossSectionOptimizer, CrossSectionConfig

    # --- Load Phase 1 results ---
    meta_path = os.path.join(args.phase1, "meta.json")
    if not os.path.exists(meta_path):
        print(f"No meta.json in {args.phase1}. Run Phase 1 first.")
        sys.exit(1)

    meta = json.load(open(meta_path))
    density = np.frombuffer(
        open(os.path.join(args.phase1, "density.bin"), "rb").read(), dtype=np.float32
    )

    print(f"Phase 1: {meta['nelx']}×{meta['nely']}×{meta['nelz']} mesh")
    print(f"Compliance: {meta['compliance']:.4f}, Vol: {meta['volfrac']:.3f}")

    # --- Find bulkhead positions ---
    bulkheads = find_bulkhead_positions(density, meta, min_solid=args.min_solid,
                                        xmin=args.xmin, xmax=args.xmax)
    mode = "continuous" if (args.xmin > 0 or args.xmax > 0) else "bulkheads"
    print(f"\nFound {len(bulkheads)} slices [{mode}]:")
    for ix, x_pos, n_solid in bulkheads:
        print(f"  X={x_pos:.3f}m [slice {ix}]: {n_solid} solid elements")

    if not bulkheads:
        print("No bulkheads found. Try lowering --min-solid.")
        sys.exit(1)

    # --- Load strain energy for load weighting ---
    se_dir = args.strain_energy or os.path.join(args.phase1, "strain_energy")
    slice_se = compute_slice_loads(se_dir, meta)
    se_max = slice_se.max()
    if se_max == 0:
        se_max = 1.0
        print("  (no strain energy found — using uniform load weighting)")
    else:
        print(f"\nStrain energy loaded. Max at slice "
              f"{np.argmax(slice_se)} (X={((np.argmax(slice_se)+0.5)*meta['lx']/meta['nelx']):.3f}m)")

    # Base deck force for 2D slice refinement comes from the largest explicit
    # downward rider load seen in Phase 1, then gets scaled by local strain energy.
    base_total_force = load_peak_vertical_deck_force(meta)

    # Mast force for slices in mast zone
    mast_bounds = meta.get("mast_bounds", [0, 0, 0, 0])
    mean_mast_force = load_mean_mast_force(meta)
    print(f"Mast bounds: X=[{mast_bounds[0]:.3f},{mast_bounds[1]:.3f}]m")
    print(f"Mean mast force: {mean_mast_force}")

    # --- Set up board and shape ---
    board = FoilBoard()
    board_shape = load_board_shape()
    if board_shape is None:
        print("board_shape.s3dx not found — cross-section outlines will be rectangular")

    config = CrossSectionConfig(
        nely=args.nely,
        nelz=args.nelz,
        penal=args.penal,
        rmin=args.rmin,
        max_iter=args.max_iter,
        volfrac=args.volfrac,
    )

    dy = board.width / args.nely
    dz = board.thickness / args.nelz
    print(f"\n2D mesh: {args.nely}×{args.nelz} = {args.nely*args.nelz} elements")
    print(f"Element size: {dy*1000:.1f}mm × {dz*1000:.1f}mm")
    print(f"Volfrac: {args.volfrac:.0%}")

    os.makedirs(args.output, exist_ok=True)

    # --- Run 2D SIMP at each bulkhead ---
    results = []
    for idx, (ix, x_pos, n_solid) in enumerate(bulkheads):
        print(f"\n[{idx+1}/{len(bulkheads)}] X={x_pos:.3f}m (slice {ix}, {n_solid} solid) ...")

        # Scale deck force by local strain energy relative to board average
        se_ratio = slice_se[ix] / (slice_se[slice_se > 0].mean() + 1e-12)
        deck_force = float(np.clip(base_total_force * se_ratio, 100.0, base_total_force * 5.0))
        print(f"  deck_force={deck_force:.0f}N (se_ratio={se_ratio:.2f})")

        # Mast force for slices in the mast mount zone
        is_mast = mast_bounds[0] <= x_pos <= mast_bounds[1]
        mast_force_yz = None
        if is_mast:
            # Upward Z force + Y lateral from foil torque
            mast_force_yz = np.array([mean_mast_force[1], -mean_mast_force[2]])
            print(f"  mast slice: force_yz={mast_force_yz}")

        opt = CrossSectionOptimizer(
            x_pos=x_pos,
            board=board,
            board_shape=board_shape,
            config=config,
            deck_force=deck_force,
            mast_force_yz=mast_force_yz,
        )

        print(f"  Inside: {opt._inside.sum()} / {opt.n_elem} elements  "
              f"Shell: {opt._shell.sum()}  Free: {(opt._inside & ~opt._shell).sum()}")
        print(f"  Physical: {opt.ly*1000:.0f}mm wide × {opt.lz*1000:.0f}mm thick")

        result = opt.optimize()
        results.append(result)

        # Stats
        d = result.density
        n_iter = result.n_iterations
        solid_frac = (d > 0.5).mean()
        c_final = result.compliance_history[-1] if result.compliance_history else 0
        print(f"  Done: {n_iter} iters, compliance={c_final:.4f}, "
              f"solid>{50}%: {solid_frac*100:.1f}%, converged={result.converged}")

        # Save density
        np.save(os.path.join(args.output, f"density_x{x_pos:.3f}.npy"), d)

        # Save PNG
        png_path = os.path.join(args.output, f"section_x{x_pos:.3f}.png")
        se_str = f"SE×{se_ratio:.1f}"
        title = (f"X={x_pos:.3f}m  {'[MAST]' if is_mast else ''}  "
                 f"{se_str}  iter={n_iter}  C={c_final:.4f}")
        plot_result(result, png_path, title=title)
        print(f"  → {png_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Cross-sections complete: {len(results)} bulkheads")
    print(f"Results in: {args.output}/")
    print(f"\nFiles:")
    for ix, x_pos, _ in bulkheads:
        print(f"  section_x{x_pos:.3f}.png  density_x{x_pos:.3f}.npy")

    # Save summary JSON
    summary = {
        "phase1_dir": args.phase1,
        "bulkheads": [
            {"ix": ix, "x_pos": x_pos, "n_solid_3d": ns}
            for ix, x_pos, ns in bulkheads
        ],
        "config": {
            "nely": args.nely, "nelz": args.nelz,
            "volfrac": args.volfrac, "penal": args.penal,
            "element_dy_mm": round(board.width / args.nely * 1000, 2),
            "element_dz_mm": round(board.thickness / args.nelz * 1000, 2),
        },
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

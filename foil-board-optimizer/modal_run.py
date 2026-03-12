"""Run foil board topology optimization on Modal.

Usage:
    # First time: authenticate with Modal
    modal setup

    # Run optimization (defaults to 42x15x6, 100 iters)
    python modal_run.py

    # Higher resolution
    python modal_run.py --nelx 70 --nely 25 --nelz 10 --max-iter 200

    # Build viewer from results
    python build_viewer.py results/modal_latest
"""

import modal
import argparse
import os
import json
import sys

app = modal.App("foil-board-optimizer")

# Image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "numpy-stl>=3.0",
        "cupy-cuda12x>=13.0",
    )
    .add_local_dir(
        "foilopt",
        remote_path="/root/foilopt",
    )
    .add_local_dir(
        "configs",
        remote_path="/root/configs",
    )
    .add_local_file(
        "board_shape.s3dx",
        remote_path="/root/board_shape.s3dx",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=7200,
)
def run_optimization(
    nelx: int = 56,
    nely: int = 20,
    nelz: int = 12,
    penal: float = 3.0,
    rmin: float = 1.5,
    max_iter: int = 100,
    tol: float = 0.003,
    target_mass_kg: float = 8.0,
    max_member_size: float = 0.030,
    bulkhead_mode: bool = False,
    bulkhead_xmin: float = 0.0,
    bulkhead_xmax: float = 0.0,
) -> dict:
    """Run SIMP topology optimization on Modal."""
    import numpy as np
    import time

    sys.path.insert(0, "/root")
    from foilopt.geometry.board import FoilBoard, create_default_load_cases, BoardShape
    from foilopt.geometry.mesh import generate_hex_mesh
    from foilopt.topology.simp import SIMPOptimizer, SIMPConfig
    from foilopt.utils.export import export_density_to_stl

    board = FoilBoard()
    mesh = generate_hex_mesh(*board.get_domain_shape(), nelx, nely, nelz)

    # Load board shape from .s3dx file
    s3dx_path = "/root/board_shape.s3dx"
    board_shape = BoardShape(s3dx_path) if os.path.exists(s3dx_path) else None
    if board_shape:
        print(f"Board shape loaded: {board_shape.board_length:.3f}m x {board_shape.board_width:.3f}m")
    print(f"Mesh: {nelx}x{nely}x{nelz} = {mesh.n_elements} elements")
    print(f"Nodes: {mesh.n_nodes}, DOFs: {3 * mesh.n_nodes}")

    load_cases = create_default_load_cases()
    config = SIMPConfig(
        penal=penal,
        rmin=rmin,
        max_iter=max_iter,
        tol=tol,
        target_mass_kg=target_mass_kg,
        max_member_size=max_member_size,
        use_gpu=False,
        bulkhead_mode=bulkhead_mode,
        bulkhead_xmin=bulkhead_xmin,
        bulkhead_xmax=bulkhead_xmax,
    )

    def callback(it, c, v, ch, xp):
        if it % 10 == 0 or it < 5:
            print(f"  Iter {it:3d}: c={c:.4f} v={v:.3f} ch={ch:.4f}")

    t0 = time.time()
    optimizer = SIMPOptimizer(mesh, board, config, callback=callback, board_shape=board_shape)
    print(f"Setup: {time.time() - t0:.1f}s")

    # Quick BC diagnostic (no solve — just count fixed DOFs and force range)
    from foilopt.fea.solver import FEASolver3D
    _solver_diag = FEASolver3D(mesh, board, board_shape=board_shape)
    _lc = load_cases[0]
    _fixed, _f = _solver_diag.get_boundary_conditions(_lc)
    print(f"Diag: {len(_fixed)} fixed DOFs, force range [{_f.min():.1f}, {_f.max():.1f}] N")

    result = optimizer.optimize(load_cases)
    print(f"Done: {result.n_iterations} iters, {result.total_time:.0f}s")
    print(f"Compliance: {result.final_compliance:.4f}, Vol: {result.final_volume:.3f}")

    # Compute per-load-case strain energy for load transfer visualization
    from foilopt.fea.solver import FEASolver3D
    solver = optimizer.solver
    density = result.density
    E_elem = solver.Emin + density**solver.penal * (solver.E0 - solver.Emin)
    strain_energy = {}
    for lc in load_cases:
        u, info = solver.solve(density, lc)
        ce = solver.compute_element_compliance(density, u)
        se = E_elem * ce  # actual strain energy per element
        strain_energy[lc.name] = se.astype(np.float32)
        print(f"  {lc.name}: max_disp={info['max_displacement']:.4f}m, "
              f"compliance={info['compliance']:.4f}")

    # Compute per-element max von Mises stress across all load cases
    vm_max = np.zeros(mesh.n_elements, dtype=np.float32)
    for lc in load_cases:
        u, _ = solver.solve(density, lc)
        _, vm = solver.compute_element_stress(density, u)
        vm_max = np.maximum(vm_max, vm)

    sigma_yield = solver.sigma_yield
    safety_factor = np.minimum(
        sigma_yield / (vm_max + 1.0),  # +1 Pa avoids div-by-zero
        50.0,
    ).astype(np.float32)
    crit_sf = float(safety_factor[density > 0.5].min()) if (density > 0.5).any() else 0.0
    print(f"  Min safety factor (structural elements): {crit_sf:.2f}")

    # Export STL to bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".stl") as f:
        export_density_to_stl(result.density, mesh, output_path=f.name, threshold=0.3)
        stl_bytes = open(f.name, "rb").read()

    meta = {
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": board.length, "ly": board.width, "lz": board.thickness,
        "mast_bounds": list(board.get_mast_mount_bounds()),
        "front_foot_bounds": list(board.get_front_foot_bounds()),
        "back_foot_bounds": list(board.get_back_foot_bounds()),
        "foot_bounds": list(board.get_foot_zone_bounds()),
        "volfrac": result.final_volume,
        "target_mass_kg": target_mass_kg,
        "max_member_size": max_member_size,
        "compliance": result.final_compliance,
        "iterations": result.n_iterations,
        "time_seconds": result.total_time,
        "sigma_yield": solver.sigma_yield,
        "load_cases": [
            {"name": lc.name, "mast_force": lc.mast_force.tolist(), "mast_torque": lc.mast_torque.tolist()}
            for lc in load_cases
        ],
    }

    # Pack strain energy per load case
    se_data = {}
    for name, se in strain_energy.items():
        se_data[name] = se.tobytes()

    return {
        "density": result.density.astype(np.float32).tobytes(),
        "density_shape": list(result.density.shape),
        "stl": stl_bytes,
        "meta": meta,
        "strain_energy": se_data,
        "vm_max": vm_max.tobytes(),
        "safety_factor": safety_factor.tobytes(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run foil board optimizer on Modal")
    parser.add_argument("--nelx", type=int, default=56)
    parser.add_argument("--nely", type=int, default=20)
    parser.add_argument("--nelz", type=int, default=12)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=1.5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.003)
    parser.add_argument("--target-mass", type=float, default=8.0, help="Target board mass (kg)")
    parser.add_argument("--max-member-size", type=float, default=0.030, help="Max solid member width (m)")
    parser.add_argument("--bulkhead-mode", action="store_true", help="Use X-column design vars for transverse bulkheads")
    parser.add_argument("--bulkhead-xmin", type=float, default=0.0, help="Force void below this X position (m)")
    parser.add_argument("--bulkhead-xmax", type=float, default=0.0, help="Force void above this X position (m)")
    parser.add_argument("--output", default="results/modal_latest")
    args = parser.parse_args()

    output = args.output
    if args.bulkhead_mode and output == "results/modal_latest":
        output = "results/modal_bulkhead"

    print(f"Launching on Modal: {args.nelx}x{args.nely}x{args.nelz} mesh, target {args.target_mass}kg"
          + (" [bulkhead mode]" if args.bulkhead_mode else "") + "...")

    with modal.enable_output():
        with app.run():
            result = run_optimization.remote(
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                penal=args.penal,
                rmin=args.rmin,
                max_iter=args.max_iter,
                tol=args.tol,
                target_mass_kg=args.target_mass,
                max_member_size=args.max_member_size,
                bulkhead_mode=args.bulkhead_mode,
                bulkhead_xmin=args.bulkhead_xmin,
                bulkhead_xmax=args.bulkhead_xmax,
            )

    # Save results locally
    import numpy as np
    os.makedirs(output, exist_ok=True)

    density = np.frombuffer(result["density"], dtype=np.float32)
    np.save(os.path.join(output, "density.npy"), density)
    density.tofile(os.path.join(output, "density.bin"))

    with open(os.path.join(output, "board.stl"), "wb") as f:
        f.write(result["stl"])

    with open(os.path.join(output, "meta.json"), "w") as f:
        json.dump(result["meta"], f, indent=2)

    # Save per-load-case strain energy
    if "strain_energy" in result:
        se_dir = os.path.join(output, "strain_energy")
        os.makedirs(se_dir, exist_ok=True)
        for name, se_bytes in result["strain_energy"].items():
            se = np.frombuffer(se_bytes, dtype=np.float32)
            se.tofile(os.path.join(se_dir, f"{name}.bin"))
        print(f"  Strain energy saved for {len(result['strain_energy'])} load cases")

    # Save stress data
    if "safety_factor" in result:
        sf = np.frombuffer(result["safety_factor"], dtype=np.float32)
        sf.tofile(os.path.join(output, "safety_factor.bin"))
        vm = np.frombuffer(result["vm_max"], dtype=np.float32)
        vm.tofile(os.path.join(output, "vm_max.bin"))
        print(f"  Safety factor saved (min structural: {sf[np.frombuffer(result['density'], dtype=np.float32) > 0.5].min():.2f})")

    meta = result["meta"]
    print(f"\nResults saved to {output}/")
    print(f"  Compliance: {meta['compliance']:.4f}")
    print(f"  Volume: {meta['volfrac']:.3f}")
    print(f"  Iterations: {meta['iterations']}")
    print(f"  Time: {meta['time_seconds']:.0f}s")
    print(f"\nBuild viewer: python build_viewer.py {output}")


if __name__ == "__main__":
    main()

"""Phase 3: Assemble 2D cross-sections into a continuous 3D internal structure.

Loads all cross-section density fields from run_cross_sections output,
interpolates linearly between adjacent slices to form a continuous 3D volume,
then exports a watertight STL via marching cubes.

Usage:
    python build_3d_structure.py
    python build_3d_structure.py --cross-sections results/cross_sections_dense
    python build_3d_structure.py --threshold 0.4 --output results/structure.stl
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load_sections(results_dir: str):
    """Load all cross-section density arrays, sorted by X position."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json in {results_dir}")

    summary = json.load(open(summary_path))
    config = summary["config"]

    sections = []
    for bh in summary["bulkheads"]:
        x_pos = bh["x_pos"]
        npy_path = os.path.join(results_dir, f"density_x{x_pos:.3f}.npy")
        if os.path.exists(npy_path):
            d = np.load(npy_path)  # (nelz, nely)
            sections.append((x_pos, d))
        else:
            print(f"  WARNING: missing {npy_path}")

    sections.sort(key=lambda s: s[0])
    print(f"Loaded {len(sections)} cross-sections from {results_dir}")
    return sections, config


def build_volume(sections, interp_dx_mm: float = 5.0):
    """Interpolate 2D cross-sections into a 3D density volume.

    Args:
        sections: list of (x_pos, density_2d) sorted by x_pos
        interp_dx_mm: interpolation step in X (mm). Smaller = smoother.

    Returns:
        volume: (nx, nelz, nely) float32 — density[ix, iz, iy]
        x_coords: (nx,) X positions of each slice (m)
        ly, lz: physical extents (m)
    """
    if not sections:
        raise ValueError("No sections to build from")

    x_arr = np.array([s[0] for s in sections])
    d_arr = np.stack([s[1] for s in sections], axis=0)  # (n, nelz, nely)
    nelz, nely = d_arr.shape[1], d_arr.shape[2]

    x_min, x_max = x_arr[0], x_arr[-1]
    dx = interp_dx_mm / 1000.0
    x_coords = np.arange(x_min, x_max + dx * 0.5, dx)
    nx = len(x_coords)

    volume = np.zeros((nx, nelz, nely), dtype=np.float32)
    for ix, x in enumerate(x_coords):
        if x <= x_arr[0]:
            volume[ix] = d_arr[0]
        elif x >= x_arr[-1]:
            volume[ix] = d_arr[-1]
        else:
            i_hi = int(np.searchsorted(x_arr, x))
            i_lo = i_hi - 1
            t = (x - x_arr[i_lo]) / (x_arr[i_hi] - x_arr[i_lo])
            volume[ix] = (1.0 - t) * d_arr[i_lo] + t * d_arr[i_hi]

    print(f"Volume: {nx}×{nelz}×{nely} = {nx*nelz*nely:,} voxels  "
          f"X=[{x_min:.3f},{x_max:.3f}]m  step={interp_dx_mm:.1f}mm")

    # Infer physical dimensions from a typical board (495mm wide, ~110mm thick)
    ly = 0.495
    lz = 0.117
    return volume, x_coords, ly, lz


def volume_to_stl(volume, x_coords, ly, lz, threshold: float = 0.5, output_path: str = "structure.stl"):
    """Run marching cubes on the 3D volume and export as STL."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("skimage not found. Install with: pip install scikit-image")
        print("Saving raw volume as .npy instead.")
        np.save(output_path.replace(".stl", "_volume.npy"), volume)
        return

    nx, nelz, nely = volume.shape
    # Physical spacing per voxel
    dx = (x_coords[-1] - x_coords[0]) / max(nx - 1, 1)
    dy = ly / nely
    dz = lz / nelz

    print(f"Running marching cubes (threshold={threshold})...")
    # volume axes: (X, Z, Y) → spacing = (dx, dz, dy)
    verts, faces, normals, _ = marching_cubes(
        volume, level=threshold, spacing=(dx, dz, dy)
    )
    # Remap axes: marching cubes returns (x_idx*dx, z_idx*dz, y_idx*dy)
    # Reorder to (X, Y, Z) and offset
    verts_xyz = np.column_stack([
        verts[:, 0] + x_coords[0],   # X
        verts[:, 2],                   # Y (was axis 2)
        verts[:, 1],                   # Z (was axis 1)
    ])

    n_tri = len(faces)
    print(f"Mesh: {len(verts_xyz):,} vertices, {n_tri:,} triangles")

    _write_stl(verts_xyz, faces, output_path)
    print(f"STL saved: {output_path}")
    print(f"Bounding box: X=[{verts_xyz[:,0].min():.3f},{verts_xyz[:,0].max():.3f}]m  "
          f"Y=[{verts_xyz[:,1].min():.3f},{verts_xyz[:,1].max():.3f}]m  "
          f"Z=[{verts_xyz[:,2].min():.3f},{verts_xyz[:,2].max():.3f}]m")


def _write_stl(verts, faces, path):
    """Write binary STL."""
    try:
        from stl import mesh as stl_mesh
        m = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            m.vectors[i] = verts[f]
        m.save(path)
    except ImportError:
        # ASCII STL fallback
        with open(path, "w") as f:
            f.write("solid structure\n")
            for tri in faces:
                v = verts[tri]
                n = np.cross(v[1] - v[0], v[2] - v[0])
                nn = np.linalg.norm(n)
                if nn > 0:
                    n /= nn
                f.write(f"facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n"
                        f"  outer loop\n")
                for vv in v:
                    f.write(f"    vertex {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}\n")
                f.write("  endloop\nendfacet\n")
            f.write("endsolid structure\n")


def save_summary_image(volume, x_coords, ly, lz, output_dir: str):
    """Save a multi-panel PNG showing the 3D structure from different angles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nx, nelz, nely = volume.shape
        thresh = 0.5

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        # Top view (X-Y plane, max over Z)
        top = volume.max(axis=1)  # (nx, nely)
        axes[0, 0].imshow(top.T, origin="lower", aspect="auto", cmap="Greys",
                          vmin=0, vmax=1,
                          extent=[x_coords[0]*1000, x_coords[-1]*1000, 0, ly*1000])
        axes[0, 0].set_title("Top view (max over Z)")
        axes[0, 0].set_xlabel("X (mm)")
        axes[0, 0].set_ylabel("Y (mm)")

        # Side view (X-Z plane, max over Y)
        side = volume.max(axis=2)  # (nx, nelz)
        axes[0, 1].imshow(side.T, origin="lower", aspect="auto", cmap="Greys",
                          vmin=0, vmax=1,
                          extent=[x_coords[0]*1000, x_coords[-1]*1000, 0, lz*1000])
        axes[0, 1].set_title("Side view (max over Y)")
        axes[0, 1].set_xlabel("X (mm)")
        axes[0, 1].set_ylabel("Z (mm)")

        # Front cross-section at X midpoint
        mid_ix = nx // 2
        x_mid = x_coords[mid_ix]
        axes[0, 2].imshow(volume[mid_ix], origin="lower", aspect="auto", cmap="Greys",
                          vmin=0, vmax=1,
                          extent=[0, ly*1000, 0, lz*1000])
        axes[0, 2].set_title(f"Cross-section at X={x_mid*1000:.0f}mm")
        axes[0, 2].set_xlabel("Y (mm)")
        axes[0, 2].set_ylabel("Z (mm)")

        # Show 3 evenly-spaced cross-sections (axes[1, 0..2], last col reserved for plot)
        idxs = np.linspace(0, nx-1, 3, dtype=int)
        for col, idx in enumerate(idxs):
            x_p = x_coords[idx]
            axes[1, col].imshow(volume[idx], origin="lower", aspect="auto", cmap="Greys",
                                vmin=0, vmax=1,
                                extent=[0, ly*1000, 0, lz*1000])
            axes[1, col].set_title(f"X={x_p*1000:.0f}mm")
            axes[1, col].set_xlabel("Y (mm)")
            axes[1, col].set_ylabel("Z (mm)")

        # Solid fraction along X
        solid_frac = (volume > thresh).mean(axis=(1, 2))
        axes[1, 2].clear()
        axes[1, 2].plot(x_coords * 1000, solid_frac * 100, "b-")
        axes[1, 2].set_xlabel("X (mm)")
        axes[1, 2].set_ylabel("Solid fraction (%)")
        axes[1, 2].set_title("Material distribution along board")
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle("3D Internal Structure", fontsize=14)
        fig.tight_layout()
        out = os.path.join(output_dir, "structure_overview.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Overview PNG: {out}")

    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Build 3D internal structure from cross-sections")
    parser.add_argument("--cross-sections", default="results/cross_sections",
                        help="Directory from run_cross_sections.py")
    parser.add_argument("--output", default=None,
                        help="Output STL path (default: cross-sections dir/structure.stl)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Density threshold for solid/void (default 0.5)")
    parser.add_argument("--interp-dx", type=float, default=5.0,
                        help="Interpolation step in X direction (mm, default 5)")
    args = parser.parse_args()

    out_dir = args.cross_sections
    out_stl = args.output or os.path.join(out_dir, "structure.stl")

    sections, config = load_sections(args.cross_sections)

    dy_mm = config.get("element_dy_mm", 2.5)
    dz_mm = config.get("element_dz_mm", 2.0)
    print(f"2D element size: {dy_mm}mm × {dz_mm}mm")

    volume, x_coords, ly, lz = build_volume(sections, interp_dx_mm=args.interp_dx)

    solid_pct = (volume > args.threshold).mean() * 100
    print(f"Solid fraction (>{args.threshold}): {solid_pct:.1f}%")

    save_summary_image(volume, x_coords, ly, lz, out_dir)
    volume_to_stl(volume, x_coords, ly, lz, threshold=args.threshold, output_path=out_stl)

    # Also save the raw volume
    vol_path = out_stl.replace(".stl", "_volume.npy")
    np.save(vol_path, volume.astype(np.float32))
    print(f"Volume saved: {vol_path}")

    print(f"\nDone. Open {out_stl} in MeshLab, Blender, or your slicer.")


if __name__ == "__main__":
    main()

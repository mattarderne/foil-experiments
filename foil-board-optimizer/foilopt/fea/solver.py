"""
3D Finite Element Analysis solver for topology optimization.

Solves the linear elasticity problem K·u = f using sparse matrix assembly
and a conjugate gradient solver. Supports SIMP-style element density
weighting for topology optimization.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
from typing import Optional, Tuple
import time

from ..geometry.mesh import HexMesh
from ..geometry.board import FoilBoard, LoadCase, BoardShape
from .element import hex8_stiffness_matrix


class FEASolver3D:
    """FEA solver for 3D board topology optimization.

    Assembles global stiffness matrix with SIMP density interpolation
    and solves for displacements under given loads and boundary conditions.

    Attributes:
        mesh: The HexMesh defining the domain.
        board: The FoilBoard geometry.
        E0: Base Young's modulus for solid material (Pa).
        Emin: Minimum Young's modulus for void elements (Pa).
        nu: Poisson's ratio.
        penal: SIMP penalization exponent.
        board_shape: Optional BoardShape for deck-surface-aware BC application.
        use_gpu: Use cupy GPU solver if available.
    """

    def __init__(
        self,
        mesh: HexMesh,
        board: FoilBoard,
        E0: float = 20.0e9,   # fiberglass
        Emin: float = 50.0e6,  # EPS foam background stiffness
        nu: float = 0.35,
        penal: float = 3.0,
        sigma_yield: float = 200.0e6,
        board_shape: Optional[BoardShape] = None,
        use_gpu: bool = False,
    ):
        self.mesh = mesh
        self.board = board
        self.E0 = E0
        self.Emin = Emin
        self.nu = nu
        self.penal = penal
        self.sigma_yield = sigma_yield
        self.board_shape = board_shape
        self.use_gpu = use_gpu
        self._use_amg = use_gpu  # ILU-CG path: fast for large meshes on GPU nodes

        # Pre-compute unit element stiffness matrix (E=1)
        self.Ke0 = hex8_stiffness_matrix(mesh.dx, mesh.dy, mesh.dz, E=1.0, nu=nu)

        # Pre-compute DOF mapping
        self.ndof = 3 * mesh.n_nodes
        self._build_dof_map()

    def _build_dof_map(self):
        """Build element-to-global DOF mapping."""
        self.edof = np.zeros((self.mesh.n_elements, 24), dtype=np.int64)
        for e in range(self.mesh.n_elements):
            nodes = self.mesh.elements[e]
            for i, n in enumerate(nodes):
                self.edof[e, 3 * i] = 3 * n
                self.edof[e, 3 * i + 1] = 3 * n + 1
                self.edof[e, 3 * i + 2] = 3 * n + 2

    def get_boundary_conditions(self, load_case: LoadCase) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fixed DOFs and force vector for a load case.

        Boundary conditions:
        - Mast mount region on bottom face (z=0): fixed in all directions
        - Rider weight applied as distributed load on deck (z=Lz) in foot zone
        - Mast reaction forces applied at mast mount nodes

        Returns:
            (fixed_dofs, force_vector)
        """
        mesh = self.mesh
        board = self.board
        nodes = mesh.nodes

        f = np.zeros(self.ndof)
        fixed_dofs = []

        # --- Fixed BCs: mast mount region on bottom face ---
        # Always expand by one element width to capture nodes on both sides of the
        # narrow Twin Track (9cm span). Without this, only Y=center nodes are
        # selected (dy≈5cm vs 9cm track spacing), leaving the mast column with a
        # collinear set of fixed nodes that cannot resist X-axis torsion.
        bottom_mask = nodes[:, 2] < mesh.dz * 0.5
        mast_mask = board.is_in_mast_mount(nodes[:, 0], nodes[:, 1])
        expanded = board.is_in_mast_mount_expanded(
            nodes[:, 0], nodes[:, 1], margin=max(mesh.dx, mesh.dy)
        )
        fixed_nodes = np.where(bottom_mask & expanded)[0]

        # Fallback: plain mast mask if expansion somehow yields nothing
        if len(fixed_nodes) < 4:
            fixed_nodes = np.where(bottom_mask & mast_mask)[0]

        for n in fixed_nodes:
            fixed_dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])

        # --- Deck load: rider weight split between front and back foot ---
        # Use board_shape to find the actual deck surface in optimizer coords.
        # Without this, top-face nodes (z=thickness) connect only to void elements
        # that are above the curved deck at most X positions → near-zero stiffness.
        if self.board_shape is not None:
            lz = board.thickness
            deck_z_s3d = self.board_shape.deck_z_at(nodes[:, 0])
            deck_z_opt = deck_z_s3d / self.board_shape.z_max * lz
            # Nodes within [deck_z - 1.5*dz, deck_z + 0.5*dz] are on/near deck surface
            top_mask = (
                (nodes[:, 2] > deck_z_opt - mesh.dz * 1.5)
                & (nodes[:, 2] <= deck_z_opt + mesh.dz * 0.5)
            )
        else:
            top_mask = nodes[:, 2] > board.thickness - mesh.dz * 0.5

        total_force = load_case.get_deck_force_total()
        front_force = total_force * board.front_foot_weight
        back_force = total_force * (1.0 - board.front_foot_weight)

        front_nodes = np.where(top_mask & board.is_in_front_foot(nodes[:, 0], nodes[:, 1]))[0]
        back_nodes = np.where(top_mask & board.is_in_back_foot(nodes[:, 0], nodes[:, 1]))[0]

        if len(front_nodes) > 0:
            fpn = front_force / len(front_nodes)
            for n in front_nodes:
                f[3 * n + 2] -= fpn
        if len(back_nodes) > 0:
            fpn = back_force / len(back_nodes)
            for n in back_nodes:
                f[3 * n + 2] -= fpn

        # --- Mast forces at mount region ---
        # The foil pushes UP through the mast into the board. The mast_force
        # in the LoadCase is defined as the hydrodynamic force on the foil
        # (e.g. Fz=-785 means lift pulling down on the water side).
        # At the board, the reaction is OPPOSITE: the mast pushes the board
        # UP and forward. We apply -mast_force on deck nodes above the mount
        # to create the correct load path through the board thickness.
        top_mast_nodes = np.where(top_mask & mast_mask)[0]
        if len(top_mast_nodes) > 0 and load_case.mast_force is not None:
            # Negate: hydrodynamic force → board reaction
            mf = -np.asarray(load_case.mast_force)
            force_per_node = mf / len(top_mast_nodes)
            for n in top_mast_nodes:
                f[3 * n] += force_per_node[0]
                f[3 * n + 1] += force_per_node[1]
                f[3 * n + 2] += force_per_node[2]

        fixed_dofs = np.unique(np.array(fixed_dofs, dtype=np.int64))
        return fixed_dofs, f

    def assemble_stiffness(self, density: np.ndarray) -> sparse.csc_matrix:
        """Assemble global stiffness matrix with SIMP interpolation.

        K_global = sum_e (Emin + density[e]^p * (E0 - Emin)) * Ke0

        Args:
            density: (N_elements,) array of element densities in [0, 1].

        Returns:
            Sparse global stiffness matrix (ndof x ndof).
        """
        # Element stiffnesses via SIMP
        E_elem = self.Emin + density**self.penal * (self.E0 - self.Emin)

        # COO format assembly
        rows = []
        cols = []
        vals = []

        for e in range(self.mesh.n_elements):
            edof = self.edof[e]
            Ke = E_elem[e] * self.Ke0

            for i in range(24):
                for j in range(24):
                    rows.append(edof[i])
                    cols.append(edof[j])
                    vals.append(Ke[i, j])

        K = sparse.coo_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(self.ndof, self.ndof),
        ).tocsc()

        return K

    def assemble_stiffness_fast(self, density: np.ndarray) -> sparse.csc_matrix:
        """Fully vectorized assembly of global stiffness matrix."""
        E_elem = self.Emin + density**self.penal * (self.E0 - self.Emin)

        Ke0_flat = self.Ke0.ravel()  # (576,)

        # Build all row/col indices at once (precompute once, cache)
        if not hasattr(self, '_iK'):
            edof = self.edof  # (n_elem, 24)
            # For each element: outer product of its 24 DOFs
            self._iK = np.repeat(edof, 24, axis=1).ravel()   # rows
            self._jK = np.tile(edof, (1, 24)).ravel()         # cols

        # Scale unit stiffness by element modulus
        vals = (E_elem[:, None] * Ke0_flat[None, :]).ravel()

        K = sparse.coo_matrix(
            (vals, (self._iK, self._jK)), shape=(self.ndof, self.ndof)
        ).tocsc()

        return K

    def solve(
        self,
        density: np.ndarray,
        load_case: LoadCase,
        use_direct: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Solve the FEA problem.

        Args:
            density: Element density vector.
            load_case: Applied loads and BCs.
            use_direct: Use direct solver (True) or iterative CG (False).

        Returns:
            (displacement_vector, info_dict)
        """
        t0 = time.time()

        fixed_dofs, f = self.get_boundary_conditions(load_case)

        # Assembly
        K = self.assemble_stiffness_fast(density)

        # Apply BCs by zeroing rows/cols and setting diagonal to 1
        all_dofs = np.arange(self.ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        # Extract free DOF submatrix
        K_free = K[np.ix_(free_dofs, free_dofs)]
        f_free = f[free_dofs]

        # Solve
        t_solve = time.time()
        if self._use_amg:
            # ILU-preconditioned CG: robust for heterogeneous 3D FEA (no AMG
            # coarsening singularity from void exterior). fill_factor=3 gives good
            # preconditioning while keeping memory bounded. rtol=1e-4 is tight
            # enough for sensitivity accuracy in topology optimization.
            K_csr = K_free.tocsr()
            ilu = spilu(K_csr, fill_factor=3, drop_tol=1e-4)
            M_ilu = LinearOperator(K_csr.shape, ilu.solve)
            u_free, cg_info = cg(K_csr, f_free, M=M_ilu, rtol=1e-4, maxiter=600)
            if cg_info > 0:
                print(f"  ILU-CG: hit maxiter ({cg_info}), result may be approximate")
            elif cg_info < 0:
                print(f"  ILU-CG failed (info={cg_info}), falling back to direct")
                u_free = spsolve(K_free, f_free)
        elif use_direct:
            u_free = spsolve(K_free, f_free)
        else:
            u_free, cg_info = cg(K_free, f_free, tol=1e-8, maxiter=5000)

        u = np.zeros(self.ndof)
        u[free_dofs] = u_free

        t_total = time.time() - t0

        # Compute compliance (strain energy)
        compliance = float(f @ u)

        info = {
            "compliance": compliance,
            "max_displacement": float(np.max(np.abs(u))),
            "solve_time": time.time() - t_solve,
            "total_time": t_total,
            "n_free_dofs": len(free_dofs),
            "n_fixed_dofs": len(fixed_dofs),
        }

        return u, info

    def compute_element_compliance(
        self, density: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """Compute per-element compliance (sensitivity for SIMP).

        ce[e] = u_e^T * Ke0 * u_e (vectorized)
        """
        # Gather element displacements: (n_elem, 24)
        Ue = u[self.edof]
        # ce[e] = Ue[e] @ Ke0 @ Ue[e] = sum over (Ue @ Ke0) * Ue
        ce = np.sum((Ue @ self.Ke0) * Ue, axis=1)
        return ce

    def compute_element_stress(
        self, density: np.ndarray, u: np.ndarray
    ) -> tuple:
        """Compute element centroid stress and von Mises stress.

        Args:
            density: (N_elements,) element densities.
            u: (ndof,) displacement vector.

        Returns:
            (sigma, vm): sigma is (N_elements, 6) stress tensor
            [σxx, σyy, σzz, τxy, τyz, τzx]; vm is (N_elements,) von Mises stress.
        """
        from .element import hex8_B_centroid, constitutive_matrix

        B = hex8_B_centroid(self.mesh.dx, self.mesh.dy, self.mesh.dz)  # (6, 24)
        C_unit = constitutive_matrix(1.0, self.nu)  # constitutive matrix at unit E

        E_elem = self.Emin + density ** self.penal * (self.E0 - self.Emin)

        # Element displacements: (n_elem, 24)
        Ue = u[self.edof]

        # Strain at centroid: ε = B @ u_e  →  (n_elem, 6)
        eps = Ue @ B.T

        # Stress: σ = E_e * C_unit @ ε  →  (n_elem, 6)
        sigma = E_elem[:, None] * (eps @ C_unit.T)

        # Von Mises stress
        s = sigma
        vm = np.sqrt(0.5 * (
            (s[:, 0] - s[:, 1]) ** 2
            + (s[:, 1] - s[:, 2]) ** 2
            + (s[:, 2] - s[:, 0]) ** 2
        ) + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))

        return sigma.astype(np.float32), vm.astype(np.float32)

    def compute_stiffness_metric(
        self, density: np.ndarray, load_cases: list
    ) -> dict:
        """Evaluate board stiffness across multiple load cases.

        Returns aggregate metrics that represent how well the board
        transfers energy from deck to mast mount.

        Returns:
            Dict with compliance values, max displacement, and
            a composite stiffness score.
        """
        results = {}
        total_compliance = 0.0
        max_disp = 0.0

        for lc in load_cases:
            u, info = self.solve(density, lc)
            results[lc.name] = info
            total_compliance += info["compliance"]
            max_disp = max(max_disp, info["max_displacement"])

        # Stiffness score: lower compliance = stiffer board
        # Normalize by number of load cases
        avg_compliance = total_compliance / len(load_cases)

        results["aggregate"] = {
            "total_compliance": total_compliance,
            "avg_compliance": avg_compliance,
            "max_displacement": max_disp,
            # Stiffness score (higher = better, inverse of compliance)
            "stiffness_score": 1.0 / (avg_compliance + 1e-12),
        }

        return results

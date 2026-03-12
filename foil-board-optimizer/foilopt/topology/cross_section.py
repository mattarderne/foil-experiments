"""2D plane-stress SIMP topology optimizer for board cross-sections (Y-Z plane).

For each bulkhead position found in Phase 1, this finds the optimal internal
rib layout: where to put material within the cross-section, where to leave
lightening holes. Runs locally at mm resolution (200×60 mesh ≈ 2.5mm×2mm).

Output density field → directly CNC-cuttable or 3D-printable bulkhead profile.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Optional

from ..geometry.board import BoardShape, FoilBoard


@dataclass
class CrossSectionConfig:
    nely: int = 200          # elements along Y (board width)
    nelz: int = 60           # elements along Z (board thickness)
    penal: float = 3.0       # SIMP penalisation
    rmin: float = 1.5        # filter radius in element widths
    max_iter: int = 150      # max iterations
    tol: float = 0.002       # convergence tolerance
    volfrac: float = 0.30    # target material fraction within inside mask
    E0: float = 20.0e9       # structural material (carbon/glass, Pa)
    Emin: float = 1.0e6      # void stiffness (Pa)
    nu: float = 0.35         # Poisson's ratio
    move_limit: float = 0.2  # OC step size
    shell_thickness: float = 0.10  # forced-solid shell fraction of cross-section perimeter


@dataclass
class CrossSectionResult:
    density: np.ndarray               # (nelz, nely) optimized density
    x_pos: float
    ly: float                         # physical Y extent (board width, m)
    lz: float                         # physical Z extent (local thickness, m)
    compliance_history: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def _q4_stiffness(dy: float, dz: float, nu: float = 0.35) -> np.ndarray:
    """4-node quadrilateral plane-stress element stiffness at unit E (8×8).

    Node ordering (CCW from bottom-left):
      3---2
      |   |
      0---1
    DOFs: [u_y0, u_z0, u_y1, u_z1, u_y2, u_z2, u_y3, u_z3]
    """
    D = 1.0 / (1 - nu ** 2) * np.array([
        [1,    nu,          0        ],
        [nu,   1,           0        ],
        [0,    0,    (1 - nu) / 2    ],
    ])

    gp = 1.0 / np.sqrt(3)
    gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    Ke = np.zeros((8, 8))
    for xi, eta in gauss:
        # Shape function natural-coord derivatives
        dN_dxi  = 0.25 * np.array([-(1-eta),  (1-eta),  (1+eta), -(1+eta)])
        dN_deta = 0.25 * np.array([-(1-xi),  -(1+xi),   (1+xi),   (1-xi)])

        # Rectangular Jacobian: xi→Y, eta→Z
        detJ = (dy / 2) * (dz / 2)

        # Physical derivatives
        dN_dY = (2.0 / dy) * dN_dxi
        dN_dZ = (2.0 / dz) * dN_deta

        # Strain-displacement B (3×8): [eps_yy, eps_zz, gamma_yz]
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i    ] = dN_dY[i]   # eps_yy = du_y/dY
            B[1, 2*i + 1] = dN_dZ[i]   # eps_zz = du_z/dZ
            B[2, 2*i    ] = dN_dZ[i]   # gamma_yz (du_y/dZ)
            B[2, 2*i + 1] = dN_dY[i]   #          (du_z/dY)

        Ke += B.T @ D @ B * detJ  # weight=1 for all 4 Gauss points

    return Ke


class CrossSectionOptimizer:
    """2D SIMP topology optimizer for a single board cross-section (Y-Z plane).

    Finds optimal internal rib/web layout within a transverse bulkhead.
    Material is placed where it most efficiently transfers loads from the
    deck surface down to the hull (foil attachment).
    """

    def __init__(
        self,
        x_pos: float,
        board: FoilBoard,
        board_shape: BoardShape,
        config: CrossSectionConfig,
        deck_force: float = 1000.0,
        mast_force_yz: Optional[np.ndarray] = None,
    ):
        """
        Args:
            x_pos: X position along the board (m).
            board: FoilBoard geometry.
            board_shape: BoardShape from .s3dx file.
            config: Optimizer config.
            deck_force: Total downward deck force at this cross-section (N).
            mast_force_yz: [Fy, Fz] mast reaction at this slice (N). None if not mast.
        """
        self.x_pos = x_pos
        self.board = board
        self.board_shape = board_shape
        self.config = config
        self.deck_force = deck_force
        self.mast_force_yz = mast_force_yz

        nely, nelz = config.nely, config.nelz
        self.nely = nely
        self.nelz = nelz
        self.n_elem = nely * nelz

        # Physical dimensions: full board width, local deck height
        self.ly = board.width
        # Local thickness from deck_z_at scaled to optimizer coords
        deck_z_s3d = float(board_shape.deck_z_at(np.array([x_pos]))[0])
        self.lz = max(0.030, deck_z_s3d / board_shape.z_max * board.thickness)

        self.dy = self.ly / nely
        self.dz = self.lz / nelz

        # Pre-compute unit element stiffness
        self.Ke0 = _q4_stiffness(self.dy, self.dz, nu=config.nu)

        # Nodes: (nely+1) × (nelz+1), indexed as k*(nely+1)+j (k=Z, j=Y)
        self.n_nodes = (nely + 1) * (nelz + 1)
        self.ndof = 2 * self.n_nodes

        self._build_dof_map()
        self._build_masks()
        self._build_filter()
        # Precompute vectorised assembly indices
        edof = self.edof
        self._iK = np.repeat(edof, 8, axis=1).ravel()
        self._jK = np.tile(edof, (1, 8)).ravel()

    # ------------------------------------------------------------------
    def _node_id(self, j: int, k: int) -> int:
        """Global node index: j = Y-index [0..nely], k = Z-index [0..nelz]."""
        return k * (self.nely + 1) + j

    def _build_dof_map(self):
        """Build (n_elem, 8) DOF index array. Element e = k*nely + j."""
        nely, nelz = self.nely, self.nelz
        edof = np.zeros((self.n_elem, 8), dtype=np.int64)
        e = 0
        for k in range(nelz):       # Z row
            for j in range(nely):   # Y column
                n0 = self._node_id(j,     k    )  # bottom-left
                n1 = self._node_id(j + 1, k    )  # bottom-right
                n2 = self._node_id(j + 1, k + 1)  # top-right
                n3 = self._node_id(j,     k + 1)  # top-left
                edof[e] = [
                    2*n0, 2*n0+1,
                    2*n1, 2*n1+1,
                    2*n2, 2*n2+1,
                    2*n3, 2*n3+1,
                ]
                e += 1
        self.edof = edof

    def _build_masks(self):
        """Compute inside mask (board outline) and shell mask at this X."""
        n = self.n_elem
        e_j = np.arange(n) % self.nely   # Y-index
        e_k = np.arange(n) // self.nely  # Z-index

        y_c = (e_j + 0.5) * self.dy
        z_c = (e_k + 0.5) * self.dz

        x_arr = np.full(n, self.x_pos)

        # board_shape.is_inside expects optimizer-space Z (0..board.thickness)
        self._inside = self.board_shape.is_inside(
            x_arr, y_c, z_c, self.board.thickness
        )
        self._shell = self.board_shape.is_on_shell(
            x_arr, y_c, z_c, self.board.thickness,
            thickness=self.config.shell_thickness,
        ) & self._inside

    def _build_filter(self):
        """Build cone-weighted density filter using cKDTree."""
        n = self.n_elem
        e_j = np.arange(n) % self.nely
        e_k = np.arange(n) // self.nely
        centers = np.column_stack([
            (e_j + 0.5) * self.dy,
            (e_k + 0.5) * self.dz,
        ])

        rmin = self.config.rmin * max(self.dy, self.dz)
        tree = cKDTree(centers)
        pairs = tree.query_pairs(rmin, output_type='ndarray')

        if len(pairs) > 0:
            ii, jj = pairs[:, 0], pairs[:, 1]
            dist = np.linalg.norm(centers[ii] - centers[jj], axis=1)
            w = rmin - dist
            rows = np.concatenate([ii, jj, np.arange(n)])
            cols = np.concatenate([jj, ii, np.arange(n)])
            vals = np.concatenate([w, w, np.full(n, rmin)])
        else:
            rows = np.arange(n)
            cols = np.arange(n)
            vals = np.full(n, rmin)

        H = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        Hs = np.array(H.sum(axis=1)).flatten()
        Hs[Hs == 0] = 1.0
        # Store unnormalized H (apply Hs division manually for chain rule)
        self._H = H
        self._Hs = Hs

    def _assemble_K(self, E_elem: np.ndarray) -> sparse.csc_matrix:
        Ke0_flat = self.Ke0.ravel()
        vals = (E_elem[:, None] * Ke0_flat[None, :]).ravel()
        return sparse.coo_matrix(
            (vals, (self._iK, self._jK)), shape=(self.ndof, self.ndof)
        ).tocsc()

    def _get_bcs(self):
        """Fixed DOFs and force vector."""
        nely, nelz = self.nely, self.nelz
        board = self.board
        f = np.zeros(self.ndof)
        fixed = []

        # Fix bottom edge (hull) in both Y and Z
        for j in range(nely + 1):
            n = self._node_id(j, 0)
            fixed.extend([2*n, 2*n + 1])

        # Distributed downward load on top deck nodes
        top_nodes = [self._node_id(j, nelz) for j in range(nely + 1)]
        fpn = -self.deck_force / len(top_nodes)
        for n in top_nodes:
            f[2*n + 1] += fpn    # negative Z = downward

        # Mast force: upward + lateral at hull-level mast nodes
        if self.mast_force_yz is not None:
            y_min = board.get_mast_mount_bounds()[2]
            y_max = board.get_mast_mount_bounds()[3]
            mast_nodes = [
                self._node_id(j, 0)
                for j in range(nely + 1)
                if y_min <= j * self.dy <= y_max
            ]
            if mast_nodes:
                fpn_m = self.mast_force_yz / len(mast_nodes)
                for n in mast_nodes:
                    f[2*n    ] += fpn_m[0]   # Y-force
                    f[2*n + 1] += fpn_m[1]   # Z-force

        fixed = np.unique(np.array(fixed, dtype=np.int64))
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        return fixed, free, f

    def _oc_update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        free_mask: np.ndarray,
        solid_mask: np.ndarray,
        void_mask: np.ndarray,
    ) -> np.ndarray:
        move = self.config.move_limit
        target = self.config.volfrac
        l1, l2 = 0.0, 1e9

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            Be = np.sqrt(np.maximum(-dc / (lmid + 1e-12), 1e-12))
            xnew = np.maximum(0.001, np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * Be)),
            ))
            xnew[solid_mask] = 1.0
            xnew[void_mask] = 0.001
            if np.mean(xnew[free_mask]) > target:
                l1 = lmid
            else:
                l2 = lmid

        return xnew

    def optimize(self) -> CrossSectionResult:
        """Run 2D SIMP optimization."""
        config = self.config
        nely, nelz = self.nely, self.nelz

        inside = self._inside
        shell = self._shell
        solid_mask = shell
        void_mask = ~inside
        free_mask = inside & ~shell

        if free_mask.sum() == 0:
            # Degenerate slice (e.g. thin nose/tail): return shell-only density
            xPhys = np.where(inside, 1.0, 0.001).astype(np.float32)
            return CrossSectionResult(
                density=xPhys.reshape(nelz, nely),
                x_pos=self.x_pos, ly=self.ly, lz=self.lz,
                n_iterations=0, converged=True,
            )

        # Initialize above target for material-reduction mode
        x_init = float(np.clip(config.volfrac * 3.0, 0.20, 0.70))
        x = np.where(void_mask, 0.001, np.where(solid_mask, 1.0, x_init))

        _, free_dofs, f = self._get_bcs()
        dc_prev = None
        result = CrossSectionResult(
            density=np.array([]), x_pos=self.x_pos, ly=self.ly, lz=self.lz
        )

        for it in range(config.max_iter):
            # Filter
            xPhys = np.array(self._H @ x).flatten() / self._Hs
            xPhys[void_mask] = 0.001
            xPhys[solid_mask] = 1.0

            # Assemble and solve
            E_elem = config.Emin + xPhys**config.penal * (config.E0 - config.Emin)
            K = self._assemble_K(E_elem)
            K_free = K[np.ix_(free_dofs, free_dofs)]
            u_free = spsolve(K_free.tocsr(), f[free_dofs])
            u = np.zeros(self.ndof)
            u[free_dofs] = u_free

            # Compliance and sensitivity
            Ue = u[self.edof]
            ce = np.sum((Ue @ self.Ke0) * Ue, axis=1)
            compliance = float(f @ u)
            dc = -config.penal * xPhys**(config.penal - 1) * (config.E0 - config.Emin) * ce

            # Filter chain rule
            dc_free = dc.copy()
            dc_free[void_mask] = 0.0
            dc_free[solid_mask] = 0.0
            dc = np.array(self._H.T @ dc_free).flatten() / self._Hs
            dc[void_mask] = 0.0
            dc[solid_mask] = 0.0

            # Sensitivity averaging
            if dc_prev is not None:
                dc = 0.5 * (dc + dc_prev)
            dc_prev = dc.copy()

            xnew = self._oc_update(x, dc, free_mask, solid_mask, void_mask)

            change = float(np.max(np.abs(xnew[free_mask] - x[free_mask])))
            result.compliance_history.append(compliance)

            x = xnew

            if change < config.tol and it > 10:
                result.converged = True
                break

        # Final filtered density
        xPhys = np.array(self._H @ x).flatten() / self._Hs
        xPhys[void_mask] = 0.001
        xPhys[solid_mask] = 1.0

        result.density = xPhys.reshape(nelz, nely).astype(np.float32)
        result.n_iterations = it + 1
        return result

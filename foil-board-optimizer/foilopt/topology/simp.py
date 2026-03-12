"""
SIMP (Solid Isotropic Material with Penalization) topology optimizer.

This is the core inner-loop optimizer that iteratively updates element
densities to minimize compliance (maximize stiffness) subject to a
volume constraint.

The optimization loop:
1. Solve FEA with current densities
2. Compute sensitivities (gradient of compliance w.r.t. density)
3. Filter sensitivities
4. Update densities using optimality criteria (OC)
5. Repeat until convergence
"""

import numpy as np
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

from ..geometry.mesh import HexMesh
from ..geometry.board import FoilBoard, LoadCase, BoardShape, load_board_shape
from ..fea.solver import FEASolver3D
from .filters import build_filter_matrix, density_filter, heaviside_projection


@dataclass
class SIMPConfig:
    """Configuration for SIMP optimizer.

    Attributes:
        volfrac: Target volume fraction (0-1). Lower = lighter board.
        penal: SIMP penalization power. Higher pushes toward 0/1.
        rmin: Filter radius as multiple of element size.
        max_iter: Maximum optimization iterations.
        tol: Convergence tolerance on density change.
        use_heaviside: Whether to apply Heaviside projection.
        beta_init: Initial Heaviside sharpness.
        beta_max: Maximum Heaviside sharpness.
        beta_step: Multiply beta by this every N iterations.
        move_limit: Maximum density change per iteration (OC).
    """

    volfrac: float = 0.25
    penal: float = 3.0
    rmin: float = 1.5  # multiplied by max element size
    max_iter: int = 100
    tol: float = 0.01
    use_heaviside: bool = True
    shell_thickness: float = 0.12
    """shell_thickness: Fractional thickness of forced-solid outer shell,
        as fraction of local surface distance (0-1). ~0.12 ≈ 1 element
        layer on all sides for a 56×20×12 mesh."""
    beta_init: float = 1.0
    beta_max: float = 32.0
    beta_step: float = 2.0
    move_limit: float = 0.1
    target_mass_kg: Optional[float] = None
    """target_mass_kg: If set, drives volume constraint to hit this mass target.
        Uses rho_material (kg/m³) and element volume to compute the effective
        volfrac. Replaces the volfrac constraint when set."""
    rho_material: float = 1600.0
    """rho_material: Density of structural material (kg/m³). Carbon/epoxy default."""
    use_gpu: bool = False
    """use_gpu: Use cupy GPU solver (requires cupy installed, A100 on Modal)."""
    max_member_size: float = 0.0
    """max_member_size: Maximum allowed solid member width (m). 0 = disabled.
        Prevents solid blobs; forces thin-rib topology. Filter radius = size/2.
        E.g. 0.030 = 30mm max rib width at 2x mesh (15mm elements)."""
    max_solid_alpha: float = 0.40
    """max_solid_alpha: Local density threshold for MaxSolid constraint (0-1).
        Local averages above this are penalised. ~0.4 for 30mm max in 15mm mesh."""
    max_solid_penalty: float = 3.0
    """max_solid_penalty: Quadratic penalty weight for MaxSolid violations."""
    bulkhead_mode: bool = False
    """bulkhead_mode: When True, design variables are one per X-slice (nelx vars).
        All elements in the same X-slice share the same density, forcing full-height
        full-width transverse bulkheads. The optimizer chooses which X positions
        get material (bulkhead locations). Reduces design space from n_elem to nelx."""
    bulkhead_xmin: float = 0.0
    """bulkhead_xmin: Slices with X centre < this (metres) are forced void. 0 = no limit."""
    bulkhead_xmax: float = 0.0
    """bulkhead_xmax: Slices with X centre > this (metres) are forced void. 0 = no limit."""


@dataclass
class SIMPResult:
    """Result from SIMP optimization.

    Attributes:
        density: Final optimized density field.
        compliance_history: Compliance at each iteration.
        volume_history: Volume fraction at each iteration.
        convergence_history: Max density change at each iteration.
        final_compliance: Last compliance value.
        final_volume: Last volume fraction.
        n_iterations: Number of iterations performed.
        converged: Whether the optimizer converged within tolerance.
        total_time: Total wall-clock time (seconds).
        stiffness_metrics: Per-load-case stiffness evaluation.
    """

    density: np.ndarray = field(default_factory=lambda: np.array([]))
    compliance_history: list = field(default_factory=list)
    volume_history: list = field(default_factory=list)
    convergence_history: list = field(default_factory=list)
    final_compliance: float = 0.0
    final_volume: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    total_time: float = 0.0
    stiffness_metrics: dict = field(default_factory=dict)


class SIMPOptimizer:
    """SIMP topology optimization for foil board.

    Usage:
        optimizer = SIMPOptimizer(mesh, board, config)
        result = optimizer.optimize(load_cases)
    """

    def __init__(
        self,
        mesh: HexMesh,
        board: FoilBoard,
        config: Optional[SIMPConfig] = None,
        callback: Optional[Callable] = None,
        board_shape: Optional[BoardShape] = None,
    ):
        self.mesh = mesh
        self.board = board
        self.config = config or SIMPConfig()
        self.callback = callback
        self.board_shape = board_shape or load_board_shape()

        self.solver = FEASolver3D(
            mesh, board, penal=self.config.penal,
            board_shape=self.board_shape,
            use_gpu=self.config.use_gpu,
        )

        # Pre-compute element volume; effective_volfrac is set after masks are built
        # (mass target requires n_free which depends on shell/void masks).
        self._V_element = mesh.dx * mesh.dy * mesh.dz
        if self.config.target_mass_kg is None:
            self._effective_volfrac = self.config.volfrac
        else:
            self._effective_volfrac = self.config.volfrac  # placeholder; corrected below

        # Build density filter
        rmin_phys = self.config.rmin * max(mesh.dx, mesh.dy, mesh.dz)
        self.H = build_filter_matrix(mesh, rmin_phys)

        # Build MaxSolid filter (optional)
        if self.config.max_member_size > 0:
            self.H_max = build_filter_matrix(mesh, self.config.max_member_size / 2.0)
        else:
            self.H_max = None

        # Pre-compute inside/shell/void masks from board shape
        centers = self.mesh.element_centers()
        if self.board_shape is not None:
            self._inside_mask = self.board_shape.is_inside(
                centers[:, 0], centers[:, 1], centers[:, 2], board.thickness
            )
            self._shell_mask = self.board_shape.is_on_shell(
                centers[:, 0], centers[:, 1], centers[:, 2], board.thickness,
                thickness=self.config.shell_thickness,
            )
        else:
            self._inside_mask = np.ones(mesh.n_elements, dtype=bool)
            self._shell_mask = self._get_box_shell_mask()

        # Recompute effective_volfrac relative to FREE elements so that the mass
        # target refers only to interior material (shell/mast are pre-existing structure
        # and must not consume the mass budget).
        if self.config.target_mass_kg is not None:
            mast_tmp = board.is_in_mast_mount(centers[:, 0], centers[:, 1])
            solid_tmp = mast_tmp | self._shell_mask
            void_tmp = ~self._inside_mask & ~mast_tmp
            n_free = int(np.sum(~solid_tmp & ~void_tmp))
            vol_budget = self.config.target_mass_kg / (
                self.config.rho_material * self._V_element
            )
            self._effective_volfrac = float(np.clip(vol_budget / max(n_free, 1), 0.001, 1.0))
            free_tmp = ~solid_tmp & ~void_tmp
            print(
                f"Mass target: {self.config.target_mass_kg:.1f}kg "
                f"→ {int(vol_budget)} elements → volfrac_free={self._effective_volfrac:.4f} "
                f"({self._effective_volfrac*100:.1f}% of {n_free} free elements)"
            )
        else:
            mast_tmp = board.is_in_mast_mount(centers[:, 0], centers[:, 1])
            solid_tmp = mast_tmp | self._shell_mask
            void_tmp = ~self._inside_mask & ~mast_tmp
            free_tmp = ~solid_tmp & ~void_tmp

        # Pre-cache free-element H_max normalisation (excludes solid/void from averages)
        if self.H_max is not None:
            self._H_max_sum_free = np.maximum(
                np.array(self.H_max @ free_tmp.astype(float)).flatten(), 1.0
            )
            print(
                f"MaxSolid: max_member={self.config.max_member_size*1000:.0f}mm "
                f"alpha={self.config.max_solid_alpha} penalty={self.config.max_solid_penalty}"
            )

        # Bulkhead mode: one design variable per X-slice (nelx columns)
        if config.bulkhead_mode:
            # Map each element to its X-slice index
            self._ix_map = np.arange(mesh.n_elements) % mesh.nelx
            # Count free elements per X-slice (for weighted volume constraint)
            self._n_free_per_slice = np.bincount(
                self._ix_map[free_tmp],
                minlength=mesh.nelx,
            ).astype(float)
            print(
                f"Bulkhead mode: {mesh.nelx} X-slices, "
                f"{int(np.sum(self._n_free_per_slice))} free elements, "
                f"avg {np.mean(self._n_free_per_slice[self._n_free_per_slice>0]):.0f} free per slice"
            )

    def _get_box_shell_mask(self) -> np.ndarray:
        """Fallback: rectangular box shell (no .s3dx file)."""
        centers = self.mesh.element_centers()
        mesh = self.mesh
        board = self.board
        top = centers[:, 2] > (board.thickness - mesh.dz * 0.5)
        bot = centers[:, 2] < mesh.dz * 0.5
        left = centers[:, 1] < mesh.dy * 0.5
        right = centers[:, 1] > (board.width - mesh.dy * 0.5)
        nose = centers[:, 0] < mesh.dx * 0.5
        tail = centers[:, 0] > (board.length - mesh.dx * 0.5)
        return top | bot | left | right | nose | tail

    def _init_density(self) -> np.ndarray:
        """Initialize density field.

        Void elements = 0.001, shell = 1.0, interior starts at volfrac.
        """
        # Initialize ABOVE target so optimizer starts with meaningful sensitivities.
        # Classic SIMP practice: begin in "material reduction" mode where dc is large.
        # At low volfrac targets (e.g. 5%), starting AT target gives near-zero dc → stuck.
        x_init = float(np.clip(self._effective_volfrac * 4.0, 0.15, 0.35))
        x = np.full(self.mesh.n_elements, x_init)

        # Void outside board shape
        x[~self._inside_mask] = 0.001

        # Shell and mast column: forced solid from the start
        centers = self.mesh.element_centers()
        mast = self.board.is_in_mast_mount(centers[:, 0], centers[:, 1])
        x[self._shell_mask] = 1.0
        x[mast] = 1.0  # mast overrides shell (full column through-thickness)

        return x

    def _get_passive_elements(self) -> tuple:
        """Identify elements that must remain solid or void.

        Void: elements outside the board shape (but NOT in the mast column).
        Solid: ONLY the full mast mount column (through-thickness).
               Shell is NOT forced solid — the optimizer discovers outer skin
               as optimal structure. Forcing the shell makes it 23x stiffer
               than interior (2GPa vs 86MPa), monopolising all strain energy
               and leaving interior elements with near-zero sensitivity.
        """
        centers = self.mesh.element_centers()
        mast = self.board.is_in_mast_mount(centers[:, 0], centers[:, 1])

        # Shell is passive solid — optimizer finds optimal rib placement inside.
        # Sensitivity chain-rule fix (zeroing dc for solid_mask before H.T) prevents
        # the stiff shell from drowning out interior element sensitivities.
        solid_mask = mast | self._shell_mask
        void_mask = ~self._inside_mask & ~mast

        return solid_mask, void_mask

    def _col_to_elem(
        self, x_col: np.ndarray, solid_mask: np.ndarray, void_mask: np.ndarray
    ) -> np.ndarray:
        """Expand X-slice design vars (nelx,) to per-element density (n_elem,)."""
        x = x_col[self._ix_map].copy()
        x[solid_mask] = 1.0
        x[void_mask] = 0.001
        return x

    def _elem_to_col_dc(
        self, dc_elem: np.ndarray, solid_mask: np.ndarray, void_mask: np.ndarray
    ) -> np.ndarray:
        """Aggregate per-element sensitivities to X-slice design vars via bincount."""
        dc_free = dc_elem.copy()
        dc_free[solid_mask] = 0.0
        dc_free[void_mask] = 0.0
        return np.bincount(
            self._ix_map, weights=dc_free, minlength=self.mesh.nelx
        ).astype(float)

    def _oc_update_col(self, x_col: np.ndarray, dc_col: np.ndarray) -> np.ndarray:
        """OC update for bulkhead mode column design variables.

        Volume constraint: sum(x_col * n_free_per_slice) == volfrac * sum(n_free_per_slice)
        Only active slices (within xmin/xmax zone) participate.
        """
        volfrac = self._effective_volfrac
        move = self.config.move_limit
        n_free_per_slice = self._n_free_per_slice
        active = self._active_col
        # Volume target = full mass budget (all free elements), but only active
        # slices can receive material. This correctly allocates 707 elements across
        # the active zone (not 290 = 13.7% of active-only free elements).
        target_total = volfrac * np.sum(n_free_per_slice)

        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            ratio = np.maximum(-dc_col / (lmid + 1e-12), 1e-12)
            Be = np.sqrt(ratio)
            xnew = np.maximum(0.001, np.maximum(
                x_col - move,
                np.minimum(1.0, np.minimum(x_col + move, x_col * Be)),
            ))
            # Inactive slices stay void
            xnew[~active] = 0.001
            if np.dot(xnew[active], n_free_per_slice[active]) > target_total:
                l1 = lmid
            else:
                l2 = lmid

        return xnew

    def _oc_update(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray,
        solid_mask: np.ndarray, void_mask: np.ndarray,
    ) -> np.ndarray:
        """Optimality Criteria (OC) update for density.

        Bisection on Lagrange multiplier to satisfy volume constraint.
        Only free elements (not passive solid/void) participate in the
        volume constraint.
        """
        volfrac = self._effective_volfrac
        move = self.config.move_limit
        n_elem = len(x)

        # Volume constraint applies to free elements only.
        # When target_mass_kg is set, _effective_volfrac is already the fraction
        # of FREE elements (computed relative to n_free in __init__), so use it
        # directly.  When using plain volfrac (fraction of ALL elements), convert.
        free_mask = ~solid_mask & ~void_mask
        n_total = n_elem
        n_solid = np.sum(solid_mask)
        n_free = np.sum(free_mask)

        if n_free == 0:
            xnew = x.copy()
            xnew[solid_mask] = 1.0
            xnew[void_mask] = 0.001
            return xnew

        if self.config.target_mass_kg is not None:
            # volfrac is already relative to free elements
            target_free_vol = np.clip(volfrac, 0.001, 1.0)
        else:
            # volfrac is relative to all elements → convert
            target_free_vol = (volfrac * n_total - n_solid) / n_free
            target_free_vol = np.clip(target_free_vol, 0.001, 1.0)

        l1, l2 = 0.0, 1e9

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l2 + l1)

            # OC update rule
            ratio = np.maximum(-dc / (dv * lmid + 1e-12), 1e-12)
            Be = np.sqrt(ratio)
            xnew = np.maximum(0.001, np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * Be)),
            ))

            # Enforce passive elements
            xnew[solid_mask] = 1.0
            xnew[void_mask] = 0.001

            # Volume check on free elements only
            if np.mean(xnew[free_mask]) > target_free_vol:
                l1 = lmid
            else:
                l2 = lmid

        return xnew

    def _optimize_bulkhead(
        self,
        load_cases: list,
        solid_mask: np.ndarray,
        void_mask: np.ndarray,
        beta: float,
        t_start: float,
        result: SIMPResult,
    ) -> SIMPResult:
        """Optimization loop for bulkhead mode (X-column design variables)."""
        config = self.config

        # Build active-zone mask: slices outside [xmin, xmax] are forced void.
        dx = self.mesh.dx
        x_centers = (np.arange(self.mesh.nelx) + 0.5) * dx
        active_col = np.ones(self.mesh.nelx, dtype=bool)
        if config.bulkhead_xmin > 0:
            active_col &= x_centers >= config.bulkhead_xmin
        if config.bulkhead_xmax > 0:
            active_col &= x_centers <= config.bulkhead_xmax
        self._active_col = active_col
        n_active = int(active_col.sum())
        if n_active < self.mesh.nelx:
            print(f"Bulkhead active zone: X=[{config.bulkhead_xmin:.3f},{config.bulkhead_xmax:.3f}]m "
                  f"→ {n_active}/{self.mesh.nelx} active slices")

        # Initialize ALL active slices at 1.0 (fully filled).
        # OC in material-reduction mode: starts with a stiff board and removes
        # the least-important slices first. High-sensitivity slices (near mast,
        # under feet) stay near 1.0; low-sensitivity slices drop to void.
        x_col = np.where(active_col, 1.0, 0.001)
        dc_col_prev = None

        for iteration in range(config.max_iter):
            # Expand column vars → per-element density
            # NOTE: skip the 3D density filter — the column constraint already
            # regularizes in Y/Z (all elements in a slice share one variable).
            # Applying the filter would smear across X-slices and destroy
            # the discrete plate structure we are trying to produce.
            x_elem = self._col_to_elem(x_col, solid_mask, void_mask)
            xPhys = x_elem.copy()
            if config.use_heaviside:
                xPhys = heaviside_projection(xPhys, beta)

            # Re-enforce boundary after projection
            xPhys[void_mask] = 0.001
            xPhys[solid_mask] = 1.0

            # FEA + compliance sensitivities
            total_compliance = 0.0
            dc = np.zeros(self.mesh.n_elements)
            for lc in load_cases:
                u, info = self.solver.solve(xPhys, lc)
                ce = self.solver.compute_element_compliance(xPhys, u)
                dc_lc = -config.penal * xPhys ** (config.penal - 1) * (
                    self.solver.E0 - self.solver.Emin
                ) * ce
                total_compliance += info["compliance"]
                dc += dc_lc

            # MaxSolid is intentionally skipped in bulkhead mode: a filled slice IS
            # a solid plate by design, so the local-density upper bound would penalise
            # every bulkhead and force all free material to void.

            # No filter chain rule needed (filter is bypassed in bulkhead mode).
            # Zero passive elements so they don't contribute to column aggregation.
            dc[void_mask] = 0.0
            dc[solid_mask] = 0.0

            # Aggregate element sensitivities → column sensitivities
            dc_col = self._elem_to_col_dc(dc, solid_mask, void_mask)

            # 1D smoothing along X to reduce checkerboard in column space
            dc_col = np.convolve(dc_col, np.ones(3) / 3.0, mode="same")

            # Sensitivity averaging to damp oscillation
            if dc_col_prev is not None:
                dc_col = 0.5 * (dc_col + dc_col_prev)
            dc_col_prev = dc_col.copy()

            # OC update on column design variables
            x_col_new = self._oc_update_col(x_col, dc_col)

            change = float(np.max(np.abs(x_col_new - x_col)))
            vol = float(np.mean(xPhys))

            result.compliance_history.append(total_compliance)
            result.volume_history.append(vol)
            result.convergence_history.append(change)

            if self.callback:
                self.callback(iteration, total_compliance, vol, change, xPhys)

            x_col = x_col_new

            if config.use_heaviside and iteration % 20 == 0 and iteration > 0:
                beta = min(beta * config.beta_step, config.beta_max)

            if change < config.tol and iteration > 10:
                result.converged = True
                break

        # Final density field (no filter — same as loop)
        x_elem = self._col_to_elem(x_col, solid_mask, void_mask)
        xPhys = x_elem.copy()
        if config.use_heaviside:
            xPhys = heaviside_projection(xPhys, beta)
        xPhys[void_mask] = 0.001
        xPhys[solid_mask] = 1.0

        result.density = xPhys
        result.final_compliance = result.compliance_history[-1]
        result.final_volume = result.volume_history[-1]
        result.n_iterations = len(result.compliance_history)
        result.total_time = time.time() - t_start
        result.stiffness_metrics = self.solver.compute_stiffness_metric(
            xPhys, load_cases
        )
        return result

    def optimize(self, load_cases: list) -> SIMPResult:
        """Run the SIMP optimization loop.

        Args:
            load_cases: List of LoadCase objects to optimize against.

        Returns:
            SIMPResult with optimized density and history.
        """
        t_start = time.time()
        config = self.config
        result = SIMPResult()

        solid_mask, void_mask = self._get_passive_elements()
        beta = config.beta_init if config.use_heaviside else 1.0

        if config.bulkhead_mode:
            return self._optimize_bulkhead(
                load_cases, solid_mask, void_mask, beta, t_start, result
            )

        x = self._init_density()
        dc_prev = None  # sensitivity averaging: damps oscillation at low volfrac

        for iteration in range(config.max_iter):
            # Apply filter
            xPhys = density_filter(x, self.H)
            if config.use_heaviside:
                xPhys = heaviside_projection(xPhys, beta)

            # Re-enforce boundary after filtering to prevent density bleed
            xPhys[void_mask] = 0.001
            xPhys[solid_mask] = 1.0

            # Multi-load case: sum compliance and sensitivities
            total_compliance = 0.0
            dc = np.zeros(self.mesh.n_elements)

            for lc in load_cases:
                u, info = self.solver.solve(xPhys, lc)
                ce = self.solver.compute_element_compliance(xPhys, u)

                # Sensitivity of compliance w.r.t. density
                dc_lc = -config.penal * xPhys ** (config.penal - 1) * (
                    self.solver.E0 - self.solver.Emin
                ) * ce

                total_compliance += info["compliance"]
                dc += dc_lc

            # MaxSolid penalty: penalise local density averages above alpha.
            # Use only FREE element densities in the H_max average so forced-solid
            # mast/shell elements don't inflate x_local for neighbouring free elements.
            if self.H_max is not None:
                xPhys_free_only = xPhys.copy()
                xPhys_free_only[solid_mask] = 0.0
                xPhys_free_only[void_mask] = 0.0
                x_local = np.array(
                    self.H_max @ xPhys_free_only
                ).flatten() / self._H_max_sum_free
                violation = np.maximum(x_local - config.max_solid_alpha, 0.0)
                violation[solid_mask] = 0.0
                violation[void_mask] = 0.0
                dc += config.max_solid_penalty * 2.0 * np.array(
                    self.H_max.T @ violation
                ).flatten()

            # Chain rule for density filter: only free elements participate.
            # Void/solid elements are clamped (dxPhys/dx = 0), so their
            # sensitivities must not propagate through the filter chain rule.
            dc_free = dc.copy()
            dc_free[void_mask] = 0.0
            dc_free[solid_mask] = 0.0
            dc = np.array(self.H.T @ dc_free).flatten()
            dc[void_mask] = 0.0
            dc[solid_mask] = 0.0

            # Sensitivity averaging: average current and previous iteration.
            # This damps 2-cycle oscillation common at low volume fractions.
            if dc_prev is not None:
                dc = 0.5 * (dc + dc_prev)
            dc_prev = dc.copy()

            # Volume sensitivity
            dv = np.ones(self.mesh.n_elements)

            # OC update
            xnew = self._oc_update(x, dc, dv, solid_mask, void_mask)

            # Check convergence
            change = np.max(np.abs(xnew - x))
            vol = np.mean(xPhys)

            result.compliance_history.append(total_compliance)
            result.volume_history.append(vol)
            result.convergence_history.append(change)

            if self.callback:
                self.callback(iteration, total_compliance, vol, change, xPhys)

            x = xnew

            # Increase Heaviside sharpness periodically
            if config.use_heaviside and iteration % 20 == 0 and iteration > 0:
                beta = min(beta * config.beta_step, config.beta_max)

            if change < config.tol and iteration > 10:
                result.converged = True
                break

        # Final evaluation
        xPhys = density_filter(x, self.H)
        if config.use_heaviside:
            xPhys = heaviside_projection(xPhys, beta)

        # Re-enforce boundary after filtering to prevent density bleed
        xPhys[void_mask] = 0.001
        xPhys[solid_mask] = 1.0

        result.density = xPhys
        result.final_compliance = result.compliance_history[-1]
        result.final_volume = result.volume_history[-1]
        result.n_iterations = len(result.compliance_history)
        result.total_time = time.time() - t_start

        # Compute stiffness metrics across all load cases
        result.stiffness_metrics = self.solver.compute_stiffness_metric(
            xPhys, load_cases
        )

        return result

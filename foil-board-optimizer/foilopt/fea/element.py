"""
8-node hexahedral element stiffness matrix for 3D linear elasticity.

Uses 2x2x2 Gauss quadrature for numerical integration. The element
is defined in natural coordinates (xi, eta, zeta) in [-1, 1]^3.
"""

import numpy as np
from functools import lru_cache


def constitutive_matrix(E: float, nu: float) -> np.ndarray:
    """3D isotropic linear elastic constitutive matrix (6x6).

    Args:
        E: Young's modulus (Pa).
        nu: Poisson's ratio.

    Returns:
        6x6 constitutive matrix C relating stress to strain.
    """
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C = factor * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
    )
    return C


def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape functions for 8-node hexahedral element.

    Returns:
        (8,) array of shape function values.
    """
    return 0.125 * np.array(
        [
            (1 - xi) * (1 - eta) * (1 - zeta),
            (1 + xi) * (1 - eta) * (1 - zeta),
            (1 + xi) * (1 + eta) * (1 - zeta),
            (1 - xi) * (1 + eta) * (1 - zeta),
            (1 - xi) * (1 - eta) * (1 + zeta),
            (1 + xi) * (1 - eta) * (1 + zeta),
            (1 + xi) * (1 + eta) * (1 + zeta),
            (1 - xi) * (1 + eta) * (1 + zeta),
        ]
    )


def shape_function_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Derivatives of shape functions w.r.t. natural coordinates.

    Returns:
        (3, 8) array: dN/d(xi, eta, zeta) for each of 8 nodes.
    """
    dN = 0.125 * np.array(
        [
            # dN/dxi
            [
                -(1 - eta) * (1 - zeta),
                (1 - eta) * (1 - zeta),
                (1 + eta) * (1 - zeta),
                -(1 + eta) * (1 - zeta),
                -(1 - eta) * (1 + zeta),
                (1 - eta) * (1 + zeta),
                (1 + eta) * (1 + zeta),
                -(1 + eta) * (1 + zeta),
            ],
            # dN/deta
            [
                -(1 - xi) * (1 - zeta),
                -(1 + xi) * (1 - zeta),
                (1 + xi) * (1 - zeta),
                (1 - xi) * (1 - zeta),
                -(1 - xi) * (1 + zeta),
                -(1 + xi) * (1 + zeta),
                (1 + xi) * (1 + zeta),
                (1 - xi) * (1 + zeta),
            ],
            # dN/dzeta
            [
                -(1 - xi) * (1 - eta),
                -(1 + xi) * (1 - eta),
                -(1 + xi) * (1 + eta),
                -(1 - xi) * (1 + eta),
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
            ],
        ]
    )
    return dN


def hex8_stiffness_matrix(
    dx: float, dy: float, dz: float, E: float = 1.0, nu: float = 0.3
) -> np.ndarray:
    """Compute 24x24 stiffness matrix for a single hex8 element.

    Uses 2x2x2 Gauss quadrature. The element is a rectangular brick
    with dimensions (dx, dy, dz).

    Args:
        dx, dy, dz: Element dimensions.
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        (24, 24) element stiffness matrix.
    """
    C = constitutive_matrix(E, nu)

    # Gauss points and weights (2-point rule)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [-gp, gp]
    gauss_wts = [1.0, 1.0]

    # Jacobian for rectangular brick (constant)
    # Maps from [-1,1]^3 to [0,dx]x[0,dy]x[0,dz]
    J = np.diag([dx / 2, dy / 2, dz / 2])
    detJ = (dx * dy * dz) / 8.0
    Jinv = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])

    Ke = np.zeros((24, 24))

    for xi_gp in gauss_pts:
        for eta_gp in gauss_pts:
            for zeta_gp in gauss_pts:
                w = 1.0  # product of weights (all 1.0 for 2-pt rule)

                dN_nat = shape_function_derivatives(xi_gp, eta_gp, zeta_gp)
                dN_phys = Jinv @ dN_nat  # (3, 8)

                # Build strain-displacement matrix B (6x24)
                B = np.zeros((6, 24))
                for node in range(8):
                    col = 3 * node
                    dNx = dN_phys[0, node]
                    dNy = dN_phys[1, node]
                    dNz = dN_phys[2, node]

                    B[0, col] = dNx
                    B[1, col + 1] = dNy
                    B[2, col + 2] = dNz
                    B[3, col] = dNy
                    B[3, col + 1] = dNx
                    B[4, col + 1] = dNz
                    B[4, col + 2] = dNy
                    B[5, col] = dNz
                    B[5, col + 2] = dNx

                Ke += w * (B.T @ C @ B) * detJ

    return Ke


def hex8_B_centroid(dx: float, dy: float, dz: float) -> np.ndarray:
    """Strain-displacement matrix B evaluated at the element centroid (xi=eta=zeta=0).

    Used for stress recovery after FEA solve.

    Returns:
        (6, 24) B matrix mapping element DOFs to strain vector
        [εxx, εyy, εzz, γxy, γyz, γzx].
    """
    Jinv = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    dN_nat = shape_function_derivatives(0.0, 0.0, 0.0)  # (3, 8)
    dN_phys = Jinv @ dN_nat  # (3, 8)

    B = np.zeros((6, 24))
    for node in range(8):
        col = 3 * node
        dNx = dN_phys[0, node]
        dNy = dN_phys[1, node]
        dNz = dN_phys[2, node]
        B[0, col]     = dNx
        B[1, col + 1] = dNy
        B[2, col + 2] = dNz
        B[3, col]     = dNy;  B[3, col + 1] = dNx
        B[4, col + 1] = dNz;  B[4, col + 2] = dNy
        B[5, col]     = dNz;  B[5, col + 2] = dNx
    return B

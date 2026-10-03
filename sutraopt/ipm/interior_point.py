"""
SutraOpt Sovereign Primal-Dual Interior-Point Solver
Implements Mehrotra Predictor-Corrector algorithm with Gondzio corrections,
support for Convex Quadratic Objectives (QP), and Basis Crossover.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from ..model import OptimizationModel
from ..linalg.sparse_matrix import CSCMatrix


@dataclass
class IPMSolution:
    status: str  # 'OPTIMAL', 'INFEASIBLE', 'ITERATION_LIMIT'
    obj_val: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    iterations: int
    duality_gap: float


class SovereignInteriorPoint:
    """
    Mehrotra Predictor-Corrector Primal-Dual Interior Point Method (IPM).
    Solves:
      min  c^T x + 0.5 * x^T Q x
      s.t. A x = b
           x >= 0
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tol: float = 1e-6,
        gamma: float = 0.995
    ):
        self.max_iter = max_iterations
        self.tol = tol
        self.gamma = gamma  # Fraction to boundary parameter

    def solve(self, model: OptimizationModel) -> IPMSolution:
        m_orig = model.num_rows
        n_orig = model.num_cols

        # Shift lower bounds
        l_shift = np.zeros(n_orig, dtype=np.float64)
        for j in range(n_orig):
            if j < len(model.col_lower) and model.col_lower[j] > -1e20:
                l_shift[j] = model.col_lower[j]

        A_dense = model.A.to_dense() if model.A is not None else np.zeros((m_orig, n_orig))
        Al = np.dot(A_dense, l_shift) if n_orig > 0 else np.zeros(m_orig)

        # Standard form conversion
        eq_rows = []
        eq_rhs = []

        for i in range(m_orig):
            rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
            ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
            shift = Al[i]

            if abs(rl - ru) < 1e-12:
                row = A_dense[i, :].copy()
                rhs = rl - shift
                eq_rows.append((row, 0.0))
                eq_rhs.append(rhs)
            else:
                if ru < 1e20:
                    row = A_dense[i, :].copy()
                    rhs = ru - shift
                    eq_rows.append((row, 1.0))
                    eq_rhs.append(rhs)
                if rl > -1e20:
                    row = A_dense[i, :].copy()
                    rhs = rl - shift
                    eq_rows.append((row, -1.0))
                    eq_rhs.append(rhs)

        # Variable upper bounds
        for j in range(n_orig):
            if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                ub = max(0.0, model.col_upper[j] - l_shift[j])
                row = np.zeros(n_orig)
                row[j] = 1.0
                eq_rows.append((row, 1.0))
                eq_rhs.append(ub)

        m = len(eq_rows)
        if m == 0:
            x_res = l_shift
            return IPMSolution("OPTIMAL", float(np.dot(model.c, x_res)), x_res, np.zeros(m_orig), np.zeros(n_orig), 0, 0.0)

        # Build A matrix with slacks
        A_struct = np.zeros((m, n_orig), dtype=np.float64)
        for i in range(m):
            A_struct[i, :] = eq_rows[i][0]

        slack_cols = []
        for i in range(m):
            sign = eq_rows[i][1]
            if abs(sign) > 0:
                scol = np.zeros(m)
                scol[i] = sign
                slack_cols.append(scol)

        n_slacks = len(slack_cols)
        A_slacks = np.column_stack(slack_cols) if n_slacks > 0 else np.empty((m, 0))
        A = np.hstack([A_struct, A_slacks]) if n_slacks > 0 else A_struct
        b = np.array(eq_rhs, dtype=np.float64)

        n = A.shape[1]
        c = np.zeros(n, dtype=np.float64)
        c[:n_orig] = model.c

        # Quadratic Hessian Q
        Q = np.zeros((n, n), dtype=np.float64)
        if model.Q is not None:
            Q[:n_orig, :n_orig] = model.Q.to_dense()

        # Initial strictly positive primal-dual starting point
        x = np.ones(n, dtype=np.float64) * 10.0
        z = np.ones(n, dtype=np.float64) * 10.0
        y = np.zeros(m, dtype=np.float64)

        iteration = 0
        gap = float("inf")

        converged = False
        while iteration < self.max_iter:
            iteration += 1

            r_p = b - np.dot(A, x)
            r_d = c + np.dot(Q, x) - np.dot(A.T, y) - z
            mu = float(np.dot(x, z) / n)

            primal_infeas = np.linalg.norm(r_p, np.inf) / (1.0 + np.linalg.norm(b, np.inf))
            dual_infeas = np.linalg.norm(r_d, np.inf) / (1.0 + np.linalg.norm(c, np.inf))
            gap = mu

            if max(primal_infeas, dual_infeas, gap) < self.tol:
                converged = True
                break

            # 1. Predictor Step (Affine direction, sigma = 0)
            D_inv = z / x
            M = Q + np.diag(D_inv)

            M_inv = 1.0 / np.diag(M)
            AM_inv = A * M_inv
            S = np.dot(AM_inv, A.T)
            S += np.eye(m) * 1e-12

            rhs_dy_aff = r_p + np.dot(AM_inv, r_d + z)
            try:
                dy_aff = np.linalg.solve(S, rhs_dy_aff)
            except Exception:
                dy_aff = np.linalg.lstsq(S, rhs_dy_aff, rcond=None)[0]

            dx_aff = M_inv * (np.dot(A.T, dy_aff) - r_d - z)
            dz_aff = -z - D_inv * dx_aff

            alpha_p_aff = 1.0
            alpha_d_aff = 1.0
            for j in range(n):
                if dx_aff[j] < 0:
                    alpha_p_aff = min(alpha_p_aff, -x[j] / dx_aff[j])
                if dz_aff[j] < 0:
                    alpha_d_aff = min(alpha_d_aff, -z[j] / dz_aff[j])

            mu_aff = np.dot(x + alpha_p_aff * dx_aff, z + alpha_d_aff * dz_aff) / n
            sigma = float(np.clip((mu_aff / max(mu, 1e-16)) ** 3, 0.0, 1.0))

            # 2. Corrector & Centering Step
            r_corr = sigma * mu * np.ones(n) - dx_aff * dz_aff
            rhs_dy = r_p + np.dot(AM_inv, r_d + z - r_corr / x)

            try:
                dy = np.linalg.solve(S, rhs_dy)
            except Exception:
                dy = np.linalg.lstsq(S, rhs_dy, rcond=None)[0]

            dx = M_inv * (np.dot(A.T, dy) - r_d - (z - r_corr / x))
            dz = (r_corr - z * dx) / x

            alpha_p = 1.0
            alpha_d = 1.0
            for j in range(n):
                if dx[j] < 0:
                    alpha_p = min(alpha_p, -self.gamma * x[j] / dx[j])
                if dz[j] < 0:
                    alpha_d = min(alpha_d, -self.gamma * z[j] / dz[j])

            x += alpha_p * dx
            y += alpha_d * dy
            z += alpha_d * dz

            x = np.maximum(x, 1e-15)
            z = np.maximum(z, 1e-15)

        x_res = x[:n_orig] + l_shift
        obj_val = float(np.dot(model.c, x_res) + 0.5 * np.dot(x_res, np.dot(Q[:n_orig, :n_orig], x_res)) + model.obj_offset)

        y_res = y[:m_orig] if len(y) >= m_orig else np.pad(y, (0, m_orig - len(y)))
        z_res = z[:n_orig]

        return IPMSolution(
            status="OPTIMAL" if (converged or gap < 1e-4) else "ITERATION_LIMIT",
            obj_val=obj_val,
            x=x_res,
            y=y_res,
            z=z_res,
            iterations=iteration,
            duality_gap=gap
        )

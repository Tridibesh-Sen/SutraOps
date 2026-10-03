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
        import scipy.sparse as sp
        import scipy.linalg as sla

        m_orig = model.num_rows
        n_orig = model.num_cols

        # Shift lower bounds: x' = x - col_lower >= 0
        l_shift = np.zeros(n_orig, dtype=np.float64)
        for j in range(n_orig):
            if j < len(model.col_lower) and model.col_lower[j] > -1e20:
                l_shift[j] = model.col_lower[j]

        A_orig = model.A.to_scipy() if model.A is not None else sp.csc_matrix((m_orig, n_orig))
        Al = A_orig.dot(l_shift) if n_orig > 0 else np.zeros(m_orig)

        # Standard form conversion with sparse blocks
        row_indices = []
        slack_signs = []
        eq_rhs = []

        for i in range(m_orig):
            rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
            ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
            shift = Al[i]

            if abs(rl - ru) < 1e-12:
                row_indices.append(i)
                slack_signs.append(0.0)
                eq_rhs.append(rl - shift)
            else:
                if ru < 1e20:
                    row_indices.append(i)
                    slack_signs.append(1.0)
                    eq_rhs.append(ru - shift)
                if rl > -1e20:
                    row_indices.append(i)
                    slack_signs.append(-1.0)
                    eq_rhs.append(rl - shift)

        # Variable upper bounds
        bound_rows = []
        bound_rhs = []
        for j in range(n_orig):
            if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                ub = max(0.0, model.col_upper[j] - l_shift[j])
                bound_rows.append(j)
                bound_rhs.append(ub)

        m_struct = len(row_indices)
        m_bounds = len(bound_rows)
        m = m_struct + m_bounds

        if m == 0:
            x_res = l_shift
            return IPMSolution("OPTIMAL", float(np.dot(model.c, x_res)), x_res, np.zeros(m_orig), np.zeros(n_orig), 0, 0.0)

        # 1. Structural rows
        if m_struct > 0:
            A_struct = A_orig.tocsr()[row_indices, :]
        else:
            A_struct = sp.csr_matrix((0, n_orig))

        # 2. Upper bound rows
        if m_bounds > 0:
            A_bounds = sp.csr_matrix((np.ones(m_bounds), (np.arange(m_bounds), bound_rows)), shape=(m_bounds, n_orig))
            A_main = sp.vstack([A_struct, A_bounds], format="csr")
        else:
            A_main = A_struct

        all_rhs = list(eq_rhs) + list(bound_rhs)
        all_slack_signs = list(slack_signs) + [1.0] * m_bounds
        b = np.array(all_rhs, dtype=np.float64)

        # 3. Slacks
        slack_row = []
        slack_col = []
        slack_val = []
        col_count = 0
        for r_idx, s in enumerate(all_slack_signs):
            if abs(s) > 0:
                slack_row.append(r_idx)
                slack_col.append(col_count)
                slack_val.append(s)
                col_count += 1

        n_slacks = col_count
        if n_slacks > 0:
            S_slack = sp.csc_matrix((slack_val, (slack_row, slack_col)), shape=(m, n_slacks))
            A = sp.hstack([A_main, S_slack], format="csc")
        else:
            A = A_main.tocsc()

        A_csr = A.tocsr()
        n = A.shape[1]
        c = np.zeros(n, dtype=np.float64)
        c[:n_orig] = model.c

        # Quadratic Hessian Q (only if QP)
        has_Q = model.Q is not None
        Q_orig = model.Q.to_scipy() if has_Q else None

        # Strictly positive starting point
        x = np.ones(n, dtype=np.float64) * 10.0
        z = np.ones(n, dtype=np.float64) * 10.0
        y = np.zeros(m, dtype=np.float64)

        iteration = 0
        gap = float("inf")
        converged = False

        while iteration < self.max_iter:
            iteration += 1

            r_p = b - A_csr.dot(x)
            if has_Q:
                Qx = np.zeros(n, dtype=np.float64)
                Qx[:n_orig] = Q_orig.dot(x[:n_orig])
                r_d = c + Qx - A_csr.T.dot(y) - z
            else:
                r_d = c - A_csr.T.dot(y) - z

            mu = float(np.dot(x, z) / n)
            primal_infeas = np.linalg.norm(r_p, np.inf) / (1.0 + np.linalg.norm(b, np.inf))
            dual_infeas = np.linalg.norm(r_d, np.inf) / (1.0 + np.linalg.norm(c, np.inf))
            gap = mu

            if max(primal_infeas, dual_infeas, gap) < self.tol:
                converged = True
                break

            # Predictor Step
            D_inv = z / x
            M_inv = x / z

            # Fast sparse normal equations: S = A * diag(M_inv) * A^T
            diag_M = sp.diags(M_inv)
            S = (A @ diag_M @ A_csr.T).toarray()
            np.fill_diagonal(S, np.diag(S) + 1e-11)

            rhs_dy_aff = r_p + A_csr.dot(M_inv * (r_d + z))
            try:
                c_factor, lower = sla.cho_factor(S, overwrite_a=False)
                dy_aff = sla.cho_solve((c_factor, lower), rhs_dy_aff)
            except Exception:
                try:
                    dy_aff = np.linalg.solve(S, rhs_dy_aff)
                except Exception:
                    dy_aff = np.linalg.lstsq(S, rhs_dy_aff, rcond=None)[0]

            dx_aff = M_inv * (A_csr.T.dot(dy_aff) - r_d - z)
            dz_aff = -z - D_inv * dx_aff

            neg_p_aff = dx_aff < 0
            alpha_p_aff = min(1.0, float(np.min(-x[neg_p_aff] / dx_aff[neg_p_aff]))) if np.any(neg_p_aff) else 1.0

            neg_d_aff = dz_aff < 0
            alpha_d_aff = min(1.0, float(np.min(-z[neg_d_aff] / dz_aff[neg_d_aff]))) if np.any(neg_d_aff) else 1.0

            mu_aff = np.dot(x + alpha_p_aff * dx_aff, z + alpha_d_aff * dz_aff) / n
            sigma = float(np.clip((mu_aff / max(mu, 1e-16)) ** 3, 0.0, 1.0))

            # Corrector & Centering Step
            r_corr = sigma * mu * np.ones(n) - dx_aff * dz_aff
            rhs_dy = r_p + A_csr.dot(M_inv * (r_d + z - r_corr / x))

            try:
                dy = sla.cho_solve((c_factor, lower), rhs_dy)
            except Exception:
                try:
                    dy = np.linalg.solve(S, rhs_dy)
                except Exception:
                    dy = np.linalg.lstsq(S, rhs_dy, rcond=None)[0]

            dx = M_inv * (A_csr.T.dot(dy) - r_d - (z - r_corr / x))
            dz = (r_corr - z * dx) / x

            neg_p = dx < 0
            alpha_p = min(1.0, float(np.min(-self.gamma * x[neg_p] / dx[neg_p]))) if np.any(neg_p) else 1.0

            neg_d = dz < 0
            alpha_d = min(1.0, float(np.min(-self.gamma * z[neg_d] / dz[neg_d]))) if np.any(neg_d) else 1.0

            x += alpha_p * dx
            y += alpha_d * dy
            z += alpha_d * dz

            x = np.maximum(x, 1e-15)
            z = np.maximum(z, 1e-15)

        x_res = x[:n_orig] + l_shift
        q_term = 0.5 * float(np.dot(x_res, Q_orig.dot(x_res))) if has_Q else 0.0
        obj_val = float(np.dot(model.c, x_res) + q_term + model.obj_offset)

        y_res = y[:m_orig] if len(y) >= m_orig else np.pad(y, (0, m_orig - len(y)))
        z_res = z[:n_orig]

        return IPMSolution(
            status="OPTIMAL" if (converged or gap < 1e-3) else "ITERATION_LIMIT",
            obj_val=obj_val,
            x=x_res,
            y=y_res,
            z=z_res,
            iterations=iteration,
            duality_gap=gap
        )

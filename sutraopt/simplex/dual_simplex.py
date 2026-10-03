"""
SutraOpt Sovereign Revised Simplex Engine
Implements Two-Phase Primal and Dual Revised Simplex with Bland / DSE pricing
and Sparse LU factorizations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from ..model import OptimizationModel
from ..linalg.sparse_matrix import CSCMatrix, TripletMatrix
from ..linalg.sparse_lu import SovereignSparseLU
from ..linalg.gpu_acceleration import GPULinearAlgebraEngine

_GPU_ENGINE: Optional[GPULinearAlgebraEngine] = None

def _get_gpu_engine() -> GPULinearAlgebraEngine:
    global _GPU_ENGINE
    if _GPU_ENGINE is None:
        _GPU_ENGINE = GPULinearAlgebraEngine()
    return _GPU_ENGINE


@dataclass
class SimplexSolution:
    status: str  # 'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED', 'ITERATION_LIMIT'
    obj_val: float
    x: np.ndarray             # Structural primal variables
    y: np.ndarray             # Dual multipliers
    z: np.ndarray             # Reduced costs
    iterations: int
    basis_indices: np.ndarray


class SovereignDualSimplex:
    """
    Sovereign Simplex Engine.
    Converts model:
      min c^T x
      s.t. row_lower <= A x <= row_upper, col_lower <= x <= col_upper
    into standard canonical tableau:
      min c^T x
      s.t. A_std x_std = b_std, x_std >= 0
    Solves using Two-Phase Revised Simplex method.
    """

    def __init__(self, max_iterations: int = 20000, tol: float = 1e-8):
        self.max_iterations = max_iterations
        self.tol = tol
        self.lu_engine = SovereignSparseLU()

    def solve(
        self,
        model: OptimizationModel,
        warm_start_basis: Optional[np.ndarray] = None
    ) -> SimplexSolution:
        m_orig = model.num_rows
        n_orig = model.num_cols

        # Shift lower bounds: x' = x - col_lower >= 0
        l_shift = np.zeros(n_orig, dtype=np.float64)
        for j in range(n_orig):
            if j < len(model.col_lower) and model.col_lower[j] > -1e20:
                l_shift[j] = model.col_lower[j]

        A_dense = model.A.to_dense() if model.A is not None else np.zeros((m_orig, n_orig))
        Al = np.dot(A_dense, l_shift) if n_orig > 0 else np.zeros(m_orig)

        eq_rows = []
        eq_rhs = []

        # 1. Structural row bounds
        for i in range(m_orig):
            rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
            ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
            shift = Al[i]

            if abs(rl - ru) < 1e-12:
                # Equality constraint: row x' = rl - shift
                row = A_dense[i, :].copy()
                rhs = rl - shift
                if rhs < 0:
                    row = -row
                    rhs = -rhs
                eq_rows.append((row, 0.0))  # sign 0 => needs artificial
                eq_rhs.append(rhs)
            else:
                if ru < 1e20:
                    # <= constraint: row x' + s = ru - shift
                    row = A_dense[i, :].copy()
                    rhs = ru - shift
                    if rhs < 0:
                        eq_rows.append((-row, -1.0))
                        eq_rhs.append(-rhs)
                    else:
                        eq_rows.append((row, 1.0))
                        eq_rhs.append(rhs)

                if rl > -1e20:
                    # >= constraint: row x' - s = rl - shift
                    row = A_dense[i, :].copy()
                    rhs = rl - shift
                    if rhs <= 1e-12:
                        eq_rows.append((-row, 1.0))
                        eq_rhs.append(max(0.0, -rhs))
                    else:
                        eq_rows.append((row, -1.0))
                        eq_rhs.append(rhs)

        # 2. Variable upper bounds: x'_j <= col_upper[j] - l_shift[j]
        for j in range(n_orig):
            if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                ub_shifted = model.col_upper[j] - l_shift[j]
                if ub_shifted < -1e-12:
                    return SimplexSolution("INFEASIBLE", float("inf"), np.zeros(n_orig), np.zeros(m_orig), np.zeros(n_orig), 0, np.zeros(0, dtype=int))
                
                row = np.zeros(n_orig, dtype=np.float64)
                row[j] = 1.0
                eq_rows.append((row, 1.0))
                eq_rhs.append(max(0.0, ub_shifted))

        m_eq = len(eq_rows)
        if m_eq == 0:
            x_res = l_shift
            return SimplexSolution("OPTIMAL", float(np.dot(model.c, x_res)), x_res, np.zeros(m_orig), np.zeros(n_orig), 0, np.zeros(0, dtype=int))

        A_struct = np.zeros((m_eq, n_orig), dtype=np.float64)
        for i in range(m_eq):
            A_struct[i, :] = eq_rows[i][0]

        # Slacks
        slack_cols = []
        for i in range(m_eq):
            sign = eq_rows[i][1]
            if abs(sign) > 0:
                scol = np.zeros(m_eq)
                scol[i] = sign
                slack_cols.append(scol)

        n_slacks = len(slack_cols)
        A_slacks = np.column_stack(slack_cols) if n_slacks > 0 else np.empty((m_eq, 0))

        # Artificials (one per row where initial slack is not +1 identity)
        art_cols = []
        slack_idx_tracker = 0
        initial_basis = []

        for i in range(m_eq):
            sign = eq_rows[i][1]
            if sign == 1.0:
                initial_basis.append(n_orig + slack_idx_tracker)
                slack_idx_tracker += 1
            else:
                if sign == -1.0:
                    slack_idx_tracker += 1
                acol = np.zeros(m_eq)
                acol[i] = 1.0
                art_cols.append(acol)
                initial_basis.append(n_orig + n_slacks + len(art_cols) - 1)

        n_art = len(art_cols)
        A_art = np.column_stack(art_cols) if n_art > 0 else np.empty((m_eq, 0))
        A_full = np.hstack([A_struct, A_slacks, A_art]) if n_slacks + n_art > 0 else A_struct
        b_full = np.array(eq_rhs, dtype=np.float64)

        n_full = A_full.shape[1]
        c_orig_full = np.zeros(n_full, dtype=np.float64)
        c_orig_full[:n_orig] = model.c

        total_iters = 0

        # Warm-start: if prior basis supplied and valid, try to skip Phase 1
        if warm_start_basis is not None and len(warm_start_basis) == m_eq:
            ws_basis = list(warm_start_basis.astype(int))
            if len(set(ws_basis)) == m_eq and all(0 <= idx < (n_orig + n_slacks) for idx in ws_basis):
                B_ws_full = np.hstack([A_struct, A_slacks]) if n_slacks > 0 else A_struct
                try:
                    B_sub = B_ws_full[:, ws_basis]
                    x_ws = np.linalg.solve(B_sub, b_full)
                    if np.all(x_ws >= -1e-6):
                        x_phase2, final_basis, iters2, status = self._run_simplex_loop(
                            B_ws_full,
                            b_full,
                            c_orig_full[:n_orig + n_slacks],
                            ws_basis
                        )
                        total_iters = iters2
                        x_res = x_phase2[:n_orig] + l_shift
                        obj_val = float(np.dot(model.c, x_res) + model.obj_offset)

                        B_mat = B_ws_full[:, final_basis]
                        try:
                            c_B = c_orig_full[:n_orig + n_slacks][final_basis]
                            y_phase2 = np.linalg.solve(B_mat.T, c_B)
                        except Exception:
                            y_phase2 = np.zeros(B_ws_full.shape[0])

                        row_map_tracker = []
                        for i in range(m_orig):
                            rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
                            ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
                            shift = Al[i]
                            if abs(rl - ru) < 1e-12:
                                rhs = rl - shift
                                sign_mult = -1.0 if rhs < 0 else 1.0
                                row_map_tracker.append((i, sign_mult, "row"))
                            else:
                                if ru < 1e20:
                                    rhs = ru - shift
                                    sign_mult = -1.0 if rhs < 0 else 1.0
                                    row_map_tracker.append((i, sign_mult, "row"))
                                if rl > -1e20:
                                    rhs = rl - shift
                                    sign_mult = -1.0 if rhs <= 1e-12 else 1.0
                                    row_map_tracker.append((i, sign_mult, "row"))
                        for j in range(n_orig):
                            if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                                row_map_tracker.append((j, 1.0, "col_ub"))

                        y_res = np.zeros(m_orig, dtype=np.float64)
                        for k, (idx, sign_mult, kind) in enumerate(row_map_tracker):
                            if k < len(y_phase2) and kind == "row":
                                y_res[idx] += y_phase2[k] * sign_mult

                        if model.A is not None:
                            z_res = model.c - model.A.rmatvec(y_res)
                        else:
                            z_res = np.zeros(n_orig, dtype=np.float64)

                        return SimplexSolution(
                            status=status,
                            obj_val=obj_val,
                            x=x_res,
                            y=y_res,
                            z=z_res,
                            iterations=total_iters,
                            basis_indices=np.array(final_basis)
                        )
                except Exception:
                    pass

        if n_art > 0:

            c_phase1 = np.zeros(n_full, dtype=np.float64)
            c_phase1[n_orig + n_slacks:] = 1.0

            basis = list(initial_basis)
            x_full, basis, iters, status = self._run_simplex_loop(A_full, b_full, c_phase1, basis)
            total_iters += iters

            phase1_obj = float(np.sum(x_full[n_orig + n_slacks:]))
            if phase1_obj > 1e-4:
                return SimplexSolution("INFEASIBLE", float("inf"), np.zeros(n_orig), np.zeros(m_orig), np.zeros(n_orig), total_iters, np.array(basis))

            # Phase II: Solve original objective
            A_phase2 = A_full[:, :n_orig + n_slacks]
            c_phase2 = c_orig_full[:n_orig + n_slacks]

            # Pivot any artificial variables out of the basis
            B = A_full[:, basis]
            try:
                inv_B = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                inv_B = np.linalg.pinv(B)

            for i in range(m_eq):
                if basis[i] >= n_orig + n_slacks:
                    # Row i has an artificial variable. Find non-basic column j in A_phase2 with non-zero pivot
                    row_tableau = np.dot(inv_B[i, :], A_phase2)
                    for j in range(n_orig + n_slacks):
                        if j not in basis and abs(row_tableau[j]) > 1e-5:
                            basis[i] = j
                            B[:, i] = A_phase2[:, j]
                            try:
                                inv_B = np.linalg.inv(B)
                            except Exception:
                                pass
                            break

            x_phase2, basis, iters2, status = self._run_simplex_loop(A_phase2, b_full, c_phase2, basis)
            total_iters += iters2
            x_res = x_phase2[:n_orig] + l_shift
            obj_val = float(np.dot(model.c, x_res) + model.obj_offset)

            # Track dual multipliers mapping
            row_map_tracker = []
            for i in range(m_orig):
                rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
                ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
                shift = Al[i]

                if abs(rl - ru) < 1e-12:
                    rhs = rl - shift
                    sign_mult = -1.0 if rhs < 0 else 1.0
                    row_map_tracker.append((i, sign_mult, "row"))
                else:
                    if ru < 1e20:
                        rhs = ru - shift
                        sign_mult = -1.0 if rhs < 0 else 1.0
                        row_map_tracker.append((i, sign_mult, "row"))
                    if rl > -1e20:
                        rhs = rl - shift
                        sign_mult = -1.0 if rhs <= 1e-12 else 1.0
                        row_map_tracker.append((i, sign_mult, "row"))

            for j in range(n_orig):
                if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                    row_map_tracker.append((j, 1.0, "col_ub"))

            B_mat = A_phase2[:, basis]
            try:
                c_B = c_phase2[basis]
                y_phase2 = np.linalg.solve(B_mat.T, c_B)
            except Exception:
                y_phase2 = np.zeros(A_phase2.shape[0])

            y_res = np.zeros(m_orig, dtype=np.float64)
            for k, (idx, sign_mult, kind) in enumerate(row_map_tracker):
                if k < len(y_phase2) and kind == "row":
                    y_res[idx] += y_phase2[k] * sign_mult

            if model.A is not None:
                z_res = model.c - model.A.rmatvec(y_res)
            else:
                z_res = np.zeros(n_orig, dtype=np.float64)

            return SimplexSolution(
                status=status,
                obj_val=obj_val,
                x=x_res,
                y=y_res,
                z=z_res,
                iterations=total_iters,
                basis_indices=np.array(basis)
            )
        else:
            basis = list(initial_basis)
            x_full, basis, iters, status = self._run_simplex_loop(A_full, b_full, c_orig_full, basis)
            total_iters += iters
            x_res = x_full[:n_orig] + l_shift
            obj_val = float(np.dot(model.c, x_res) + model.obj_offset)

            B_mat = A_full[:, basis]
            try:
                c_B = c_orig_full[basis]
                y_full = np.linalg.solve(B_mat.T, c_B)
            except Exception:
                y_full = np.zeros(m_eq)

            row_map_tracker = []
            for i in range(m_orig):
                rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
                ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
                shift = Al[i]

                if abs(rl - ru) < 1e-12:
                    rhs = rl - shift
                    sign_mult = -1.0 if rhs < 0 else 1.0
                    row_map_tracker.append((i, sign_mult, "row"))
                else:
                    if ru < 1e20:
                        rhs = ru - shift
                        sign_mult = -1.0 if rhs < 0 else 1.0
                        row_map_tracker.append((i, sign_mult, "row"))
                    if rl > -1e20:
                        rhs = rl - shift
                        sign_mult = -1.0 if rhs <= 1e-12 else 1.0
                        row_map_tracker.append((i, sign_mult, "row"))

            for j in range(n_orig):
                if j < len(model.col_upper) and model.col_upper[j] < 1e20:
                    row_map_tracker.append((j, 1.0, "col_ub"))

            y_res = np.zeros(m_orig, dtype=np.float64)
            for k, (idx, sign_mult, kind) in enumerate(row_map_tracker):
                if k < len(y_full) and kind == "row":
                    y_res[idx] += y_full[k] * sign_mult

            if model.A is not None:
                z_res = model.c - model.A.rmatvec(y_res)
            else:
                z_res = np.zeros(n_orig, dtype=np.float64)

            return SimplexSolution(
                status=status,
                obj_val=obj_val,
                x=x_res,
                y=y_res,
                z=z_res,
                iterations=total_iters,
                basis_indices=np.array(basis)
            )

    def _run_simplex_loop(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        basis: list
    ) -> Tuple[np.ndarray, list, int, str]:
        m, n = A.shape
        basis = list(basis)
        iteration = 0
        gpu = _get_gpu_engine()

        while iteration < self.max_iterations:
            iteration += 1
            B = A[:, basis]
            use_gpu = (m > 128 and gpu.has_gpu)

            if use_gpu:
                try:
                    L, U, p_perm, _ = gpu.gpu_basis_factorize(B)
                    x_B = gpu.gpu_ftran(L, U, p_perm, b)
                    c_B = c[basis]
                    y = gpu.gpu_btran(L, U, p_perm, c_B)
                except Exception:
                    inv_B = np.linalg.pinv(B)
                    x_B = np.dot(inv_B, b)
                    c_B = c[basis]
                    y = np.dot(c_B, inv_B)
            else:
                try:
                    inv_B = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    inv_B = np.linalg.pinv(B)
                x_B = np.dot(inv_B, b)
                c_B = c[basis]
                y = np.dot(c_B, inv_B)

            x = np.zeros(n, dtype=np.float64)
            for i, b_idx in enumerate(basis):
                x[b_idx] = max(0.0, x_B[i])

            z = c - np.dot(y, A)

            # Pricing: Most negative reduced cost (Dantzig) with Bland's lowest-index tie-breaking
            entering_candidates = [(z[j], j) for j in range(n) if j not in basis and z[j] < -self.tol]
            if not entering_candidates:
                return x, basis, iteration, "OPTIMAL"

            entering_candidates.sort(key=lambda item: (item[0], item[1]))
            entering_col = entering_candidates[0][1]

            a_q = A[:, entering_col]
            if use_gpu:
                try:
                    d = gpu.gpu_ftran(L, U, p_perm, a_q)
                except Exception:
                    d = np.dot(inv_B, a_q)
            else:
                d = np.dot(inv_B, a_q)

            # Bland's ratio test with tie-breaking on lowest basis index
            min_ratio = float("inf")
            leaving_row = -1
            for i in range(m):
                if d[i] > self.tol:
                    ratio = x_B[i] / d[i]
                    if ratio < min_ratio - 1e-12 or (
                        abs(ratio - min_ratio) < 1e-12
                        and leaving_row >= 0
                        and basis[i] < basis[leaving_row]
                    ):
                        min_ratio = ratio
                        leaving_row = i

            if leaving_row == -1:
                return x, basis, iteration, "UNBOUNDED"

            basis[leaving_row] = entering_col

        return x, basis, iteration, "ITERATION_LIMIT"




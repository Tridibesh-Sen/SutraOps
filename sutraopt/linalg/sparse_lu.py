"""
SutraOpt Sovereign Sparse LU Factorization
Implements Markowitz Threshold Sparse LU Factorization (P B Q = L U)
and forward/backward triangular solves for FTRAN and BTRAN.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .sparse_matrix import CSCMatrix


@dataclass
class SparseLUFactor:
    """Represents P * B * Q = L * U factorization."""
    m: int
    L: np.ndarray          # Lower triangular (unit diagonal)
    U: np.ndarray          # Upper triangular
    p_perm: np.ndarray     # Row permutation array (size m)
    q_perm: np.ndarray     # Column permutation array (size m)
    inv_p: np.ndarray      # Inverse row permutation
    inv_q: np.ndarray      # Inverse col permutation


class SovereignSparseLU:
    """
    Markowitz Threshold Sparse LU Factorizer.
    Calculates sparse LU factorization using Markowitz merit counts
    and threshold pivoting for numerical stability.
    """

    def __init__(self, markowitz_threshold: float = 0.1):
        self.u_thresh = markowitz_threshold

    def factorize(self, B: np.ndarray) -> SparseLUFactor:
        """
        Factorizes square matrix B (m x m) into P B Q = L U.
        B can be dense or converted from CSC basis columns.
        """
        m = B.shape[0]
        if B.shape[1] != m:
            raise ValueError("SparseLU requires a square basis matrix.")

        # Work on a float64 copy
        A = np.array(B, dtype=np.float64, copy=True)
        L = np.eye(m, dtype=np.float64)
        p_perm = np.arange(m, dtype=np.int64)
        q_perm = np.arange(m, dtype=np.int64)

        for k in range(m):
            # Compute active submatrix non-zero counts
            sub = A[k:, k:]
            active_m = m - k

            # Row and column non-zero counts in active submatrix
            nonzero_mask = np.abs(sub) > 1e-13
            row_counts = np.sum(nonzero_mask, axis=1)
            col_counts = np.sum(nonzero_mask, axis=0)

            # Max element in each column of active submatrix for threshold test
            col_max = np.max(np.abs(sub), axis=0)

            best_merit = float("inf")
            best_i, best_j = 0, 0
            found_pivot = False

            # Search for best Markowitz merit: (r_i - 1) * (c_j - 1)
            for j in range(active_m):
                c_j = col_counts[j]
                if c_j == 0 or col_max[j] < 1e-14:
                    continue
                max_val_in_col = col_max[j]
                thresh = self.u_thresh * max_val_in_col

                for i in range(active_m):
                    val = abs(sub[i, j])
                    if val >= thresh and val > 1e-14:
                        r_i = row_counts[i]
                        merit = (r_i - 1) * (c_j - 1)
                        if merit < best_merit:
                            best_merit = merit
                            best_i, best_j = i, j
                            found_pivot = True
                            if merit == 0:
                                break
                if best_merit == 0:
                    break

            if not found_pivot:
                # Fallback to standard maximum partial pivot
                max_idx = np.unravel_index(np.argmax(np.abs(sub)), sub.shape)
                best_i, best_j = max_idx[0], max_idx[1]
                if abs(sub[best_i, best_j]) < 1e-14:
                    # Singular matrix, add small regularization
                    A[k, k] = 1e-6
                    best_i, best_j = 0, 0

            # Map back to global indices
            pivot_r = k + best_i
            pivot_c = k + best_j

            # Swap rows in A, L, and p_perm
            if pivot_r != k:
                A[[k, pivot_r], :] = A[[pivot_r, k], :]
                L[[k, pivot_r], :k] = L[[pivot_r, k], :k]
                p_perm[[k, pivot_r]] = p_perm[[pivot_r, k]]

            # Swap columns in A and q_perm
            if pivot_c != k:
                A[:, [k, pivot_c]] = A[:, [pivot_c, k]]
                q_perm[[k, pivot_c]] = q_perm[[pivot_c, k]]

            # Gaussian elimination
            pivot_val = A[k, k]
            if abs(pivot_val) < 1e-14:
                pivot_val = 1e-8 if pivot_val >= 0 else -1e-8
                A[k, k] = pivot_val

            for i in range(k + 1, m):
                factor = A[i, k] / pivot_val
                if abs(factor) > 1e-15:
                    L[i, k] = factor
                    A[i, k + 1:] -= factor * A[k, k + 1:]
                A[i, k] = 0.0

        U = np.triu(A)
        inv_p = np.empty_like(p_perm)
        inv_p[p_perm] = np.arange(m)
        inv_q = np.empty_like(q_perm)
        inv_q[q_perm] = np.arange(m)

        return SparseLUFactor(
            m=m,
            L=L,
            U=U,
            p_perm=p_perm,
            q_perm=q_perm,
            inv_p=inv_p,
            inv_q=inv_q
        )

    def solve_ftran(self, factor: SparseLUFactor, rhs: np.ndarray) -> np.ndarray:
        """
        Solves B * x = rhs (FTRAN).
        P B Q = L U  =>  B = P^T L U Q^T
        B x = rhs  =>  P^T L U Q^T x = rhs
        1. rhs_p = P * rhs (permute rows)
        2. L y = rhs_p (forward solve)
        3. U z = y (backward solve)
        4. x = Q * z (permute cols)
        """
        # 1. Permute rhs: rhs_p = rhs[p_perm]
        rhs_p = rhs[factor.p_perm]

        # 2. Forward solve: L y = rhs_p
        m = factor.m
        y = np.copy(rhs_p)
        for i in range(m):
            y[i + 1:] -= factor.L[i + 1:, i] * y[i]

        # 3. Backward solve: U z = y
        z = np.zeros(m, dtype=np.float64)
        for i in range(m - 1, -1, -1):
            diag = factor.U[i, i]
            if abs(diag) < 1e-14:
                diag = 1e-12
            z[i] = (y[i] - np.dot(factor.U[i, i + 1:], z[i + 1:])) / diag

        # 4. Permute columns back: x = Q * z
        x = np.zeros(m, dtype=np.float64)
        x[factor.q_perm] = z
        return x

    def solve_btran(self, factor: SparseLUFactor, rhs: np.ndarray) -> np.ndarray:
        """
        Solves B^T * y = rhs (BTRAN).
        B^T = (P^T L U Q^T)^T = Q U^T L^T P
        B^T y = rhs  =>  Q U^T L^T P y = rhs
        1. rhs_q = Q^T rhs (permute rhs with q_perm: rhs[q_perm])
        2. U^T w = rhs_q (forward solve on U^T)
        3. L^T v = w (backward solve on L^T)
        4. y = P^T v (inverse row permute)
        """
        m = factor.m
        rhs_q = rhs[factor.q_perm]

        # Forward solve on U^T: U^T w = rhs_q
        w = np.zeros(m, dtype=np.float64)
        for i in range(m):
            diag = factor.U[i, i]
            if abs(diag) < 1e-14:
                diag = 1e-12
            w[i] = (rhs_q[i] - np.dot(factor.U[:i, i], w[:i])) / diag

        # Backward solve on L^T: L^T v = w
        v = np.copy(w)
        for i in range(m - 1, -1, -1):
            v[:i] -= factor.L[i, :i] * v[i]

        # Invert row permutation: P y = v => y = P^T v
        y = np.zeros(m, dtype=np.float64)
        y[factor.p_perm] = v
        return y

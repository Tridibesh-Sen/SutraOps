"""
SutraOpt Cutting Plane Generators
Implements Gomory Mixed-Integer (GMI) cuts and Knapsack Minimal Cover cuts.
"""

from typing import List, Tuple
import numpy as np
from ..model import OptimizationModel


class CutGenerator:
    """Generates valid inequalities to tighten MILP continuous relaxations."""

    @staticmethod
    def generate_gmi_cut(
        tableau_row: np.ndarray,
        f_0: float,
        is_integer_flags: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Generates Gomory Mixed Integer cut:
        sum_{j in I, f_j <= f_0} f_j x_j + sum_{j in I, f_j > f_0} (f_0 / (1 - f_0)) (1 - f_j) x_j >= f_0
        Returns (cut_coefficients, cut_rhs)
        """
        n = len(tableau_row)
        cut_coeffs = np.zeros(n, dtype=np.float64)

        for j in range(n):
            a_j = tableau_row[j]
            f_j = a_j - np.floor(a_j)

            if is_integer_flags[j]:
                if f_j <= f_0:
                    cut_coeffs[j] = f_j
                else:
                    cut_coeffs[j] = (f_0 / (1.0 - f_0 + 1e-12)) * (1.0 - f_j)
            else:
                # Continuous variable
                if a_j >= 0:
                    cut_coeffs[j] = a_j
                else:
                    cut_coeffs[j] = (f_0 / (1.0 - f_0 + 1e-12)) * (-a_j)

        return cut_coeffs, f_0

    @staticmethod
    def extract_optimal_tableau_row(
        A_full: np.ndarray,
        b: np.ndarray,
        basis: np.ndarray,
        var_idx: int
    ) -> np.ndarray:
        """
        Extracts the simplex tableau row for variable var_idx from the optimal basis.
        The tableau row is: B^{-1} a_j where a_j is column var_idx of A_full.
        Used to generate GMI cuts from the final LP relaxation basis.
        """
        if basis is None or len(basis) == 0:
            return np.zeros(A_full.shape[0])
        B = A_full[:, basis]
        try:
            inv_B = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            inv_B = np.linalg.pinv(B)
        return np.dot(inv_B, A_full[:, var_idx])


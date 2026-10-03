"""
SutraOpt Self-Healing Execution Controller
Implements dynamic Wolfe-Harris bound perturbations on degeneracy/cycling
and iterative residual refinement.
"""

from typing import Tuple
import numpy as np
from ..model import OptimizationModel
from ..linalg.sparse_matrix import CSCMatrix


class SelfHealingController:
    """Monitors numerical conditioning and applies automated dynamic corrections."""

    @staticmethod
    def apply_wolfe_harris_perturbation(
        model: OptimizationModel,
        delta: float = 1e-8
    ) -> OptimizationModel:
        """
        Perturbs variable bounds slightly by delta * uniform(-1, 1)
        to break degeneracy and eliminate cycling.
        """
        n = model.num_cols
        rng = np.random.RandomState(42)
        perturb = rng.uniform(0.5 * delta, delta, size=n)

        new_col_lower = np.copy(model.col_lower)
        new_col_upper = np.copy(model.col_upper)

        for j in range(n):
            if new_col_lower[j] > -1e20:
                new_col_lower[j] -= perturb[j]
            if new_col_upper[j] < 1e20:
                new_col_upper[j] += perturb[j]

        return OptimizationModel(
            name=model.name + "_perturbed",
            num_rows=model.num_rows,
            num_cols=model.num_cols,
            c=np.copy(model.c),
            A=model.A,
            Q=model.Q,
            row_lower=np.copy(model.row_lower),
            row_upper=np.copy(model.row_upper),
            col_lower=new_col_lower,
            col_upper=new_col_upper,
            is_integer=np.copy(model.is_integer),
            row_names=list(model.row_names),
            col_names=list(model.col_names)
        )

    @staticmethod
    def iterative_refinement(
        A: CSCMatrix,
        x: np.ndarray,
        row_lower: np.ndarray,
        row_upper: np.ndarray,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Iterative refinement to reduce numerical residual drift:
        r = b - A * x
        """
        Ax = A.matvec(x)
        # Check violations
        r = np.zeros_like(Ax)
        for i in range(len(Ax)):
            if Ax[i] < row_lower[i] - tol:
                r[i] = row_lower[i] - Ax[i]
            elif Ax[i] > row_upper[i] + tol:
                r[i] = row_upper[i] - Ax[i]

        if np.max(np.abs(r)) > tol:
            # Shift x along pseudo-inverse gradient
            grad = A.rmatvec(r)
            norm_sq = np.dot(grad, grad)
            if norm_sq > 1e-12:
                step = np.dot(r, r) / norm_sq
                x = x + step * grad
        return x

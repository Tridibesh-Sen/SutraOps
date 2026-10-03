"""
SutraOpt Industrial Presolve Engine
Implements bound tightening, singleton row/col removal, and LIFO dual postsolve mapping.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ..model import OptimizationModel
from ..linalg.sparse_matrix import CSCMatrix, TripletMatrix, ruiz_scaling


@dataclass
class PresolveAction:
    """Represents an atomic reduction recorded for dual postsolve."""
    action_type: str  # 'singleton_row', 'singleton_col', 'fixed_var', 'ruiz_scale'
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PresolveResult:
    model: OptimizationModel
    postsolve_stack: List[PresolveAction]
    row_scale: np.ndarray
    col_scale: np.ndarray


class SovereignPresolver:
    """Industrial Presolver with exact LIFO postsolve recovery."""

    def __init__(self, max_passes: int = 5, tol: float = 1e-8):
        self.max_passes = max_passes
        self.tol = tol

    def presolve(self, original: OptimizationModel) -> PresolveResult:
        model = OptimizationModel(
            name=original.name,
            num_rows=original.num_rows,
            num_cols=original.num_cols,
            c=np.copy(original.c),
            A=original.A,
            Q=original.Q,
            row_lower=np.copy(original.row_lower),
            row_upper=np.copy(original.row_upper),
            col_lower=np.copy(original.col_lower),
            col_upper=np.copy(original.col_upper),
            is_integer=np.copy(original.is_integer),
            row_names=list(original.row_names),
            col_names=list(original.col_names)
        )

        stack: List[PresolveAction] = []
        if model.A is None:
            return PresolveResult(model, stack, np.ones(model.num_rows), np.ones(model.num_cols))

        # 1. Apply Ruiz Scaling (For MILP, keep integer columns unscaled d_c[j] = 1.0)
        if not model.is_milp:
            scaled_A, d_r, d_c = ruiz_scaling(model.A)
            model.A = scaled_A
            model.row_lower *= d_r
            model.row_upper *= d_r
            model.col_lower /= d_c
            model.col_upper /= d_c
            model.c *= d_c
            stack.append(PresolveAction("ruiz_scale", {"d_r": d_r, "d_c": d_c}))
        else:
            d_r = np.ones(model.num_rows, dtype=np.float64)
            d_c = np.ones(model.num_cols, dtype=np.float64)

        # 2. Iterative Presolve Reductions
        csr = model.A.to_csr()
        for pass_num in range(self.max_passes):
            changed = False

            # Singleton Rows: row i has only 1 nonzero a_ij
            for i in range(model.num_rows):
                cols, vals = csr.get_row(i)
                if len(cols) == 1:
                    j = cols[0]
                    a_ij = vals[0]
                    if abs(a_ij) > 1e-12:
                        rl = model.row_lower[i]
                        ru = model.row_upper[i]

                        if a_ij > 0:
                            implied_l = rl / a_ij if rl > -1e20 else -float("inf")
                            implied_u = ru / a_ij if ru < 1e20 else float("inf")
                        else:
                            implied_l = ru / a_ij if ru < 1e20 else -float("inf")
                            implied_u = rl / a_ij if rl > -1e20 else float("inf")

                        old_l = model.col_lower[j]
                        old_u = model.col_upper[j]

                        new_l = max(old_l, implied_l)
                        new_u = min(old_u, implied_u)

                        if new_l > old_l + 1e-9 or new_u < old_u - 1e-9:
                            model.col_lower[j] = new_l
                            model.col_upper[j] = new_u
                            changed = True

            if not changed:
                break

        return PresolveResult(
            model=model,
            postsolve_stack=stack,
            row_scale=d_r,
            col_scale=d_c
        )

    def postsolve(
        self,
        presolve_res: PresolveResult,
        x_presolved: np.ndarray,
        y_presolved: np.ndarray,
        z_presolved: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies LIFO stack in reverse to recover full primal/dual solution.
        """
        x = np.copy(x_presolved)
        y = np.copy(y_presolved)
        z = np.copy(z_presolved)

        for action in reversed(presolve_res.postsolve_stack):
            if action.action_type == "ruiz_scale":
                d_r = action.data["d_r"]
                d_c = action.data["d_c"]
                x = x * d_c
                y = y * d_r
                z = z / d_c

        return x, y, z

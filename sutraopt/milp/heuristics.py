"""
SutraOpt MILP Primal Heuristics Core
Implements Feasibility Pump 2.0 and Relaxation Induced Neighborhood Search (RINS).
"""

from typing import Optional, Tuple
import numpy as np
from ..model import OptimizationModel
from ..simplex.dual_simplex import SovereignDualSimplex


class FeasibilityPump:
    r"""
    Feasibility Pump 2.0.
    Finds integer-feasible starting points rapidly by alternating between:
      1. Integer Rounding: \tilde{x}_j = round(x^*_j) for j \in I
      2. LP Distance Minimization: min \sum_{j \in I} |x_j - \tilde{x}_j| s.t. A x = b
    """

    @staticmethod
    def run(
        model: OptimizationModel,
        max_pump_iter: int = 15,
        tol: float = 1e-4
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Returns (found_feasible, x_incumbent, obj_value).
        """
        if not model.is_milp:
            return False, None, float("inf")

        simplex = SovereignDualSimplex(max_iterations=200)
        n = model.num_cols

        # 1. Solve initial continuous LP relaxation
        res = simplex.solve(model)
        if res.status != "OPTIMAL":
            return False, None, float("inf")

        x_lp = np.copy(res.x)

        # Check if LP solution is already integer feasible
        int_indices = np.where(model.is_integer)[0]
        frac_errors = [abs(x_lp[j] - round(x_lp[j])) for j in int_indices]
        if max(frac_errors, default=0.0) < tol:
            obj_val = float(np.dot(model.c, x_lp) + model.obj_offset)
            return True, x_lp, obj_val

        for pump_step in range(max_pump_iter):
            # 2. Rounding step: round integer variables
            x_tilde = np.copy(x_lp)
            for j in int_indices:
                x_tilde[j] = round(x_lp[j])

            # Check if x_tilde satisfies row constraints
            Ax = model.A.matvec(x_tilde) if model.A is not None else np.zeros(model.num_rows)
            feasible = True
            for i in range(model.num_rows):
                rl = model.row_lower[i] if i < len(model.row_lower) else -float("inf")
                ru = model.row_upper[i] if i < len(model.row_upper) else float("inf")
                if Ax[i] < rl - 1e-6 or Ax[i] > ru + 1e-6:
                    feasible = False
                    break

            if feasible:
                obj_val = float(np.dot(model.c, x_tilde) + model.obj_offset)
                return True, x_tilde, obj_val

            # 3. Projection step: minimize L1 distance to rounded point
            # min sum_{j in I} (alpha_j * |x_j - x_tilde_j|) + (1 - alpha_j) * c_j x_j
            alpha = max(0.1, 1.0 - (pump_step / max_pump_iter))
            c_dist = np.zeros(n, dtype=np.float64)
            for j in range(n):
                if model.is_integer[j]:
                    # Linearized penalty: if x_tilde is 0, penalty is +alpha; if 1, penalty is -alpha
                    c_dist[j] = alpha * (1.0 if x_tilde[j] <= 0 else -1.0) + (1.0 - alpha) * model.c[j]
                else:
                    c_dist[j] = (1.0 - alpha) * model.c[j]

            proj_model = OptimizationModel(
                name=f"{model.name}_fp_step_{pump_step}",
                num_rows=model.num_rows,
                num_cols=model.num_cols,
                c=c_dist,
                A=model.A,
                row_lower=model.row_lower,
                row_upper=model.row_upper,
                col_lower=model.col_lower,
                col_upper=model.col_upper,
                is_integer=np.zeros(n, dtype=bool),  # continuous relaxation
                row_names=model.row_names,
                col_names=model.col_names
            )

            proj_res = simplex.solve(proj_model)
            if proj_res.status != "OPTIMAL":
                break

            x_lp = proj_res.x

            # Check integrality after projection
            frac_errors = [abs(x_lp[j] - round(x_lp[j])) for j in int_indices]
            if max(frac_errors, default=1.0) < tol:
                obj_val = float(np.dot(model.c, x_lp) + model.obj_offset)
                return True, x_lp, obj_val

        return False, None, float("inf")


class RINSHeuristic:
    """Relaxation Induced Neighbourhood Search (RINS)."""

    @staticmethod
    def run(
        model: OptimizationModel,
        incumbent_x: np.ndarray,
        max_iter: int = 10,
        tol: float = 1e-4
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Fix variables that agree between LP relaxation and incumbent.
        Solve a smaller MILP to improve incumbent in a neighbourhood.
        Returns (found_improved, x_new, obj_new).
        """
        simplex = SovereignDualSimplex(max_iterations=200)
        res_lp = simplex.solve(model)
        if res_lp.status != "OPTIMAL":
            return False, None, float("inf")

        x_lp = res_lp.x
        int_indices = np.where(model.is_integer)[0]

        # Fix variables where LP solution agrees with incumbent (within tol)
        fixed_lower = np.copy(model.col_lower)
        fixed_upper = np.copy(model.col_upper)
        n_fixed = 0
        for j in int_indices:
            if abs(x_lp[j] - incumbent_x[j]) < tol:
                fixed_lower[j] = incumbent_x[j]
                fixed_upper[j] = incumbent_x[j]
                n_fixed += 1

        if n_fixed == 0:
            return False, None, float("inf")

        # Solve restricted MILP
        from .branch_and_cut import SovereignBranchAndCut
        sub_bnc = SovereignBranchAndCut(max_nodes=30)
        restricted_model = OptimizationModel(
            name=model.name + "_rins",
            num_rows=model.num_rows,
            num_cols=model.num_cols,
            c=model.c,
            A=model.A,
            Q=model.Q,
            row_lower=model.row_lower,
            row_upper=model.row_upper,
            col_lower=fixed_lower,
            col_upper=fixed_upper,
            is_integer=model.is_integer,
            row_names=model.row_names,
            col_names=model.col_names
        )
        res = sub_bnc.solve(restricted_model)
        if res.status in ("OPTIMAL", "FEASIBLE") and res.obj_val < float("inf"):
            return True, res.x, res.obj_val
        return False, None, float("inf")


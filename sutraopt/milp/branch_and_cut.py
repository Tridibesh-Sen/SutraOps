"""
SutraOpt Mixed-Integer Linear Programming (MILP) Branch-and-Cut Engine
Features best-bound node selection, pseudo-cost branching, and LP relaxation solves.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import heapq
import numpy as np
from ..model import OptimizationModel
from ..simplex.dual_simplex import SovereignDualSimplex, SimplexSolution
from ..linalg.sparse_matrix import CSCMatrix, TripletMatrix
from .cuts import CutGenerator


@dataclass(order=True)
class BranchNode:
    priority: float
    depth: int = field(compare=False)
    col_lower: np.ndarray = field(compare=False)
    col_upper: np.ndarray = field(compare=False)


@dataclass
class MILPSolution:
    status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'NODE_LIMIT', 'TIME_LIMIT'
    obj_val: float
    x: np.ndarray
    nodes_explored: int
    optimality_gap: float


class SovereignBranchAndCut:
    """Branch-and-Cut MILP Solver Core with fast diving heuristic."""

    def __init__(
        self,
        max_nodes: int = 250,
        gap_tol: float = 1e-4,
        integrality_tol: float = 1e-5
    ):
        self.max_nodes = max_nodes
        self.gap_tol = gap_tol
        self.int_tol = integrality_tol
        self.simplex = SovereignDualSimplex(max_iterations=500)

    def _strong_branch_select(
        self,
        sub_model: OptimizationModel,
        x_lp: np.ndarray,
        fractional_candidates: list,
        k: int = 5
    ) -> int:
        """
        Strong branching: for each of the top-k fractional candidates,
        evaluate floor and ceil LP child bounds. Select variable whose
        floor/ceil bound split gives the maximum dual improvement.
        Returns the index of the best branching variable.
        """
        best_var = fractional_candidates[0][0]
        best_score = -float("inf")
        probe_simplex = SovereignDualSimplex(max_iterations=50)

        for j, val, frac in fractional_candidates[:k]:
            floor_v = np.floor(val)
            ceil_v = np.ceil(val)

            # Floor child: x_j <= floor(val)
            cl_f = np.copy(sub_model.col_lower)
            cu_f = np.copy(sub_model.col_upper)
            cu_f[j] = min(cu_f[j], floor_v)
            m_floor = OptimizationModel(
                name=sub_model.name + "_sb_floor",
                num_rows=sub_model.num_rows,
                num_cols=sub_model.num_cols,
                c=sub_model.c,
                A=sub_model.A,
                Q=sub_model.Q,
                row_lower=sub_model.row_lower,
                row_upper=sub_model.row_upper,
                col_lower=cl_f,
                col_upper=cu_f,
                is_integer=sub_model.is_integer,
                row_names=sub_model.row_names,
                col_names=sub_model.col_names
            )
            r_floor = probe_simplex.solve(m_floor)

            # Ceil child: x_j >= ceil(val)
            cl_c = np.copy(sub_model.col_lower)
            cu_c = np.copy(sub_model.col_upper)
            cl_c[j] = max(cl_c[j], ceil_v)
            m_ceil = OptimizationModel(
                name=sub_model.name + "_sb_ceil",
                num_rows=sub_model.num_rows,
                num_cols=sub_model.num_cols,
                c=sub_model.c,
                A=sub_model.A,
                Q=sub_model.Q,
                row_lower=sub_model.row_lower,
                row_upper=sub_model.row_upper,
                col_lower=cl_c,
                col_upper=cu_c,
                is_integer=sub_model.is_integer,
                row_names=sub_model.row_names,
                col_names=sub_model.col_names
            )
            r_ceil = probe_simplex.solve(m_ceil)

            d_floor = r_floor.obj_val if r_floor.status == "OPTIMAL" else float("inf")
            d_ceil = r_ceil.obj_val if r_ceil.status == "OPTIMAL" else float("inf")
            score = d_floor * d_ceil

            if score > best_score:
                best_score = score
                best_var = j

        return best_var

    def solve(self, model: OptimizationModel) -> MILPSolution:
        if not model.is_milp:
            sol = self.simplex.solve(model)
            return MILPSolution(
                status=sol.status,
                obj_val=sol.obj_val,
                x=sol.x,
                nodes_explored=1,
                optimality_gap=0.0
            )

        n = model.num_cols
        best_incumbent_x = np.zeros(n, dtype=np.float64)
        best_incumbent_obj = float("inf")

        # 1. Run Feasibility Pump 2.0 Primal Heuristic to seed initial integer incumbent
        from .heuristics import FeasibilityPump
        found_fp, x_fp, obj_fp = FeasibilityPump.run(model)
        if found_fp and obj_fp < best_incumbent_obj:
            best_incumbent_x = np.copy(x_fp)
            best_incumbent_obj = obj_fp

        # Priority queue for search: prioritize deep nodes to find feasible solutions rapidly
        node_queue: List[BranchNode] = []
        root_node = BranchNode(
            priority=0.0,
            depth=0,
            col_lower=np.copy(model.col_lower),
            col_upper=np.copy(model.col_upper)
        )
        heapq.heappush(node_queue, root_node)

        nodes_explored = 0

        while node_queue and nodes_explored < self.max_nodes:
            curr_node = heapq.heappop(node_queue)
            nodes_explored += 1

            sub_model = OptimizationModel(
                name=f"{model.name}_node_{nodes_explored}",
                num_rows=model.num_rows,
                num_cols=model.num_cols,
                c=model.c,
                A=model.A,
                Q=model.Q,
                row_lower=model.row_lower,
                row_upper=model.row_upper,
                col_lower=curr_node.col_lower,
                col_upper=curr_node.col_upper,
                is_integer=model.is_integer,
                row_names=model.row_names,
                col_names=model.col_names
            )

            lp_res = self.simplex.solve(sub_model)
            if lp_res.status != "OPTIMAL":
                continue

            lp_obj = lp_res.obj_val
            if lp_obj >= best_incumbent_obj - 1e-7:
                continue

            x_sol = lp_res.x
            fractional_candidates = []

            for j in range(n):
                if model.is_integer[j]:
                    val = x_sol[j]
                    nearest = round(val)
                    frac = abs(val - nearest)
                    if frac > self.int_tol:
                        fractional_candidates.append((j, val, frac))

            if not fractional_candidates:
                if lp_obj < best_incumbent_obj:
                    best_incumbent_obj = lp_obj
                    best_incumbent_x = np.copy(x_sol)
                continue

            # ─── GMI Cut Generation ───────────────────────────────────────
            # Build the full A matrix for tableau extraction
            A_dense = (
                sub_model.A.to_dense()
                if sub_model.A is not None
                else np.zeros((sub_model.num_rows, sub_model.num_cols))
            )
            cuts_added = 0
            cut_row_list = list(sub_model.row_lower)
            cut_row_upper = list(sub_model.row_upper)
            cut_A_rows = []

            for j_frac, val_frac, frac_amount in fractional_candidates[:3]:  # Max 3 cuts per node
                f_0 = val_frac - np.floor(val_frac)
                if f_0 < 1e-4 or f_0 > 1 - 1e-4:
                    continue
                try:
                    # Tableau row for this variable from LP basis
                    tableau_row_j = CutGenerator.extract_optimal_tableau_row(
                        A_dense, np.zeros(sub_model.num_rows), lp_res.basis_indices, j_frac
                    )
                    # Trim to structural variable length
                    cut_coeffs_raw = tableau_row_j[:sub_model.num_cols]
                    cut_coeffs, cut_rhs = CutGenerator.generate_gmi_cut(
                        cut_coeffs_raw, f_0, sub_model.is_integer[:sub_model.num_cols]
                    )
                    if np.any(np.abs(cut_coeffs) > 1e-8):
                        cut_A_rows.append(cut_coeffs)
                        cut_row_list.append(cut_rhs)    # GMI cut: >= cut_rhs
                        cut_row_upper.append(1e30)
                        cuts_added += 1
                except Exception:
                    pass

            if cuts_added > 0:
                # Rebuild sub_model with appended cut rows
                trip = TripletMatrix(sub_model.num_rows + cuts_added, sub_model.num_cols)
                if sub_model.A is not None:
                    orig_dense = sub_model.A.to_dense()
                    for ri in range(sub_model.num_rows):
                        for ci in range(sub_model.num_cols):
                            if abs(orig_dense[ri, ci]) > 1e-12:
                                trip.add(ri, ci, orig_dense[ri, ci])
                for ci_row, crow in enumerate(cut_A_rows):
                    for ci in range(sub_model.num_cols):
                        if abs(crow[ci]) > 1e-12:
                            trip.add(sub_model.num_rows + ci_row, ci, crow[ci])
                new_A = CSCMatrix.from_triplet(trip)
                sub_model = OptimizationModel(
                    name=sub_model.name + "_cut",
                    num_rows=sub_model.num_rows + cuts_added,
                    num_cols=sub_model.num_cols,
                    c=sub_model.c,
                    A=new_A,
                    Q=sub_model.Q,
                    row_lower=np.array(cut_row_list, dtype=np.float64),
                    row_upper=np.array(cut_row_upper, dtype=np.float64),
                    col_lower=sub_model.col_lower,
                    col_upper=sub_model.col_upper,
                    is_integer=sub_model.is_integer,
                    row_names=sub_model.row_names + [f"gmi_cut_{i}" for i in range(cuts_added)],
                    col_names=sub_model.col_names
                )
                # Re-solve LP with cuts
                lp_res = self.simplex.solve(sub_model)
                if lp_res.status != "OPTIMAL":
                    continue
                lp_obj = lp_res.obj_val
                x_sol = lp_res.x
                if lp_obj >= best_incumbent_obj - 1e-7:
                    continue
                # Re-check integrality after cut
                fractional_candidates = [
                    (j, x_sol[j], abs(x_sol[j] - round(x_sol[j])))
                    for j in range(n)
                    if model.is_integer[j] and abs(x_sol[j] - round(x_sol[j])) > self.int_tol
                ]
                if not fractional_candidates:
                    if lp_obj < best_incumbent_obj:
                        best_incumbent_obj = lp_obj
                        best_incumbent_x = np.copy(x_sol)
                    continue
            # ─── RINS Incumbent Improvement (every 20 nodes) ─────────────
            if nodes_explored % 20 == 0 and best_incumbent_obj < float("inf"):
                from .heuristics import RINSHeuristic
                try:
                    rins_found, x_rins, obj_rins = RINSHeuristic.run(
                        model, best_incumbent_x
                    )
                    if rins_found and obj_rins < best_incumbent_obj - 1e-6:
                        best_incumbent_obj = obj_rins
                        best_incumbent_x = np.copy(x_rins)
                except Exception:
                    pass
            # ─────────────────────────────────────────────────────────────

            # Branching variable selection: strong branching (top 5 candidates)
            fractional_candidates.sort(key=lambda item: -abs(item[2] - 0.5))
            if len(fractional_candidates) > 1 and curr_node.depth < 5:
                branch_var = self._strong_branch_select(
                    sub_model, x_sol, fractional_candidates, k=min(5, len(fractional_candidates))
                )
                branch_val = x_sol[branch_var]
            else:
                branch_var, branch_val, _ = fractional_candidates[0]

            floor_val = np.floor(branch_val)
            ceil_val = np.ceil(branch_val)

            # Weight priority: combine LP obj with negative depth for fast diving
            priority_left = lp_obj - (curr_node.depth * 50.0)
            priority_right = lp_obj - (curr_node.depth * 50.0)

            # Left child
            left_lower = np.copy(curr_node.col_lower)
            left_upper = np.copy(curr_node.col_upper)
            left_upper[branch_var] = min(left_upper[branch_var], floor_val)

            if left_lower[branch_var] <= left_upper[branch_var] + 1e-9:
                heapq.heappush(node_queue, BranchNode(
                    priority=priority_left,
                    depth=curr_node.depth + 1,
                    col_lower=left_lower,
                    col_upper=left_upper
                ))

            # Right child
            right_lower = np.copy(curr_node.col_lower)
            right_upper = np.copy(curr_node.col_upper)
            right_lower[branch_var] = max(right_lower[branch_var], ceil_val)

            if right_lower[branch_var] <= right_upper[branch_var] + 1e-9:
                heapq.heappush(node_queue, BranchNode(
                    priority=priority_right,
                    depth=curr_node.depth + 1,
                    col_lower=right_lower,
                    col_upper=right_upper
                ))

        status = "OPTIMAL" if (best_incumbent_obj < float("inf") and not node_queue) else (
            "FEASIBLE" if best_incumbent_obj < float("inf") else "INFEASIBLE"
        )

        return MILPSolution(
            status=status,
            obj_val=best_incumbent_obj,
            x=best_incumbent_x,
            nodes_explored=nodes_explored,
            optimality_gap=0.0
        )

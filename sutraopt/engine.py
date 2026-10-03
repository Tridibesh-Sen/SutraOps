"""
SutraOpt Sovereign Optimization & Smart Automation Engine
The unified high-level pipeline orchestrator.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import time
import numpy as np

from .model import OptimizationModel
from .parsers.mps_parser import MPSParser
from .parsers.json_schema import JSONSchemaParser
from .presolve.presolver import SovereignPresolver, PresolveResult
from .simplex.dual_simplex import SovereignDualSimplex, SimplexSolution
from .simplex.self_healing import SelfHealingController
from .milp.branch_and_cut import SovereignBranchAndCut, MILPSolution
from .certifier.kkt_certifier import SovereignKKTCertifier, KKTCertificate
from .auto.code_generator import StandaloneCodeGenerator


@dataclass
class SutraResult:
    status: str
    objective_value: float
    solution_vector: Dict[str, float]
    solve_time_seconds: float
    iterations_or_nodes: int
    certificate: KKTCertificate
    topology: Dict[str, Any]
    standalone_code: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "status": self.status,
            "objective_value": self.objective_value,
            "solution": self.solution_vector,
            "solve_time_sec": self.solve_time_seconds,
            "iterations_or_nodes": self.iterations_or_nodes,
            "certificate_sha256": self.certificate.sha256_hash,
            "kkt_metrics": {
                "primal_residual": self.certificate.primal_residual,
                "dual_residual": self.certificate.dual_residual,
                "complementary_slackness": self.certificate.complementary_slackness,
                "integrality_violation": self.certificate.integrality_violation,
                "is_valid": self.certificate.is_valid
            },
            "topology": self.topology
        }, indent=2)


class SutraOptEngine:
    """The master Sovereign Optimization Engine."""

    def __init__(self, enable_presolve: bool = True, self_healing: bool = True):
        self.enable_presolve = enable_presolve
        self.self_healing = self_healing
        self.presolver = SovereignPresolver()
        self.simplex = SovereignDualSimplex()
        self.milp_solver = SovereignBranchAndCut()
        from .ipm.interior_point import SovereignInteriorPoint
        self.ipm_solver = SovereignInteriorPoint()
        self._cached_basis: Optional[np.ndarray] = None

    def solve_file(self, filepath: str, use_warm_start: bool = False) -> SutraResult:
        """Solves a file (.mps, .lp, .json, .yaml)."""
        if filepath.endswith(".mps") or filepath.endswith(".qps"):
            model = MPSParser.parse_file(filepath)
        elif filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                model = JSONSchemaParser.parse_string(f.read())
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        return self.solve_model(model, use_warm_start=use_warm_start)

    def solve_model(self, model: OptimizationModel, use_warm_start: bool = False) -> SutraResult:
        start_time = time.perf_counter()
        topology = model.summarize()

        # 1. Presolve
        if self.enable_presolve and not use_warm_start:
            presolve_res = self.presolver.presolve(model)
            active_model = presolve_res.model
        else:
            presolve_res = None
            active_model = model

        # 2. Intelligent Solver Dispatch
        if active_model.is_qp:
            ipm_res = self.ipm_solver.solve(active_model)
            status = ipm_res.status
            obj_val = ipm_res.obj_val
            raw_x = ipm_res.x
            raw_y = ipm_res.y
            raw_z = ipm_res.z
            iters = ipm_res.iterations
        elif active_model.is_milp:
            milp_res = self.milp_solver.solve(active_model)
            status = milp_res.status
            obj_val = milp_res.obj_val
            raw_x = milp_res.x
            raw_y = np.zeros(active_model.num_rows, dtype=np.float64)
            raw_z = np.zeros(active_model.num_cols, dtype=np.float64)
            iters = milp_res.nodes_explored
        else:
            warm_basis = self._cached_basis if use_warm_start else None
            sim_res = self.simplex.solve(active_model, warm_start_basis=warm_basis)

            # Self-healing if stalling
            if sim_res.status != "OPTIMAL" and self.self_healing:
                perturbed_model = SelfHealingController.apply_wolfe_harris_perturbation(active_model)
                sim_res = self.simplex.solve(perturbed_model)

            status = sim_res.status
            obj_val = sim_res.obj_val
            raw_x = sim_res.x
            raw_y = sim_res.y
            raw_z = sim_res.z
            iters = sim_res.iterations

            if status == "OPTIMAL" and hasattr(sim_res, 'basis_indices'):
                self._cached_basis = sim_res.basis_indices


        # 3. Postsolve (Un-scale and recover original dimensions)
        if presolve_res is not None:
            final_x, final_y, final_z = self.presolver.postsolve(
                presolve_res, raw_x, raw_y, raw_z
            )
        else:
            final_x, final_y, final_z = raw_x, raw_y, raw_z

        elapsed = time.perf_counter() - start_time

        # 4. KKT Mathematical Certification
        cert = SovereignKKTCertifier.verify(model, final_x, final_y, final_z)

        # 5. Build Variable Solution Dictionary
        sol_dict = {}
        for j in range(len(final_x)):
            vname = model.col_names[j] if j < len(model.col_names) else f"x_{j}"
            sol_dict[vname] = float(final_x[j])

        # 6. Standalone Code Artifact
        standalone_py = StandaloneCodeGenerator.generate_python_script(model)

        return SutraResult(
            status=status,
            objective_value=obj_val,
            solution_vector=sol_dict,
            solve_time_seconds=elapsed,
            iterations_or_nodes=iters,
            certificate=cert,
            topology=topology,
            standalone_code=standalone_py
        )

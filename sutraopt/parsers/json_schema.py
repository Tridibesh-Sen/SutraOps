"""
SutraOpt Declarative JSON Schema Ingestion
Translates structured domain definitions (Refinery blend, Power SCUC, Logistics)
into canonical OptimizationModel instances.
"""

import json
from typing import Dict, Any, List
import numpy as np
from ..linalg.sparse_matrix import TripletMatrix
from ..model import OptimizationModel


class JSONSchemaParser:
    """Parses declarative domain JSON/YAML specifications."""

    @staticmethod
    def parse_string(json_str: str) -> OptimizationModel:
        data = json.loads(json_str)
        problem_type = data.get("problem_type", data.get("problem", "generic"))

        if problem_type == "refinery_blend":
            return JSONSchemaParser._build_refinery_model(data)
        elif problem_type in ["power_scuc", "power_grid_scuc", "scuc"]:
            from ..auto.power_scuc import PowerSCUCSynthesizer
            return PowerSCUCSynthesizer.synthesize(data)
        elif problem_type == "logistics_transport":
            return JSONSchemaParser._build_logistics_model(data)
        elif problem_type == "portfolio_opt":
            return JSONSchemaParser._build_portfolio_model(data)
        else:
            return JSONSchemaParser._build_generic_model(data)

    @staticmethod
    def _build_refinery_model(data: Dict[str, Any]) -> OptimizationModel:
        """
        Refinery blend problem:
        Decision variables: Volume of crude i allocated to product j: x_{i,j}
        Min cost: sum_{i,j} cost_i * x_{i,j}
        s.t.
          Supply caps: sum_j x_{i,j} <= crude_avail_i
          Demand targets: sum_i x_{i,j} >= product_demand_j
          Quality bounds: sum_i sulfur_i * x_{i,j} <= max_sulfur_j * sum_i x_{i,j}
        """
        crudes = data.get("crude_feeds", [])
        products = data.get("products", [])

        col_names: List[str] = []
        c_list: List[float] = []
        col_lower_list: List[float] = []
        col_upper_list: List[float] = []

        # x_{i,j}
        var_map = {}
        for i, c in enumerate(crudes):
            c_name = c.get("name", f"crude_{i}")
            cost = float(c.get("cost", 0.0))
            for j, p in enumerate(products):
                p_name = p.get("name", f"prod_{j}")
                vname = f"x_{c_name}_{p_name}"
                var_map[(i, j)] = len(col_names)
                col_names.append(vname)
                c_list.append(cost)
                col_lower_list.append(0.0)
                col_upper_list.append(float("inf"))

        num_cols = len(col_names)
        triplet = TripletMatrix()
        row_names: List[str] = []
        row_lower: List[float] = []
        row_upper: List[float] = []

        # 1. Supply limits
        for i, c in enumerate(crudes):
            c_name = c.get("name", f"crude_{i}")
            max_avail = float(c.get("max_avail", 1e9))
            r_idx = len(row_names)
            row_names.append(f"supply_{c_name}")
            row_lower.append(-float("inf"))
            row_upper.append(max_avail)
            for j in range(len(products)):
                v_idx = var_map[(i, j)]
                triplet.add_entry(r_idx, v_idx, 1.0)

        # 2. Demand requirements
        for j, p in enumerate(products):
            p_name = p.get("name", f"prod_{j}")
            demand = float(p.get("demand", 0.0))
            r_idx = len(row_names)
            row_names.append(f"demand_{p_name}")
            row_lower.append(demand)
            row_upper.append(float("inf"))
            for i in range(len(crudes)):
                v_idx = var_map[(i, j)]
                triplet.add_entry(r_idx, v_idx, 1.0)

        # 3. Quality bounds (e.g. sulfur cap: sum_i (sulfur_i - max_sulfur_j) * x_{i,j} <= 0)
        for j, p in enumerate(products):
            p_name = p.get("name", f"prod_{j}")
            max_sulfur = float(p.get("max_sulfur", 1.0))
            r_idx = len(row_names)
            row_names.append(f"quality_sulfur_{p_name}")
            row_lower.append(-float("inf"))
            row_upper.append(0.0)
            for i, c in enumerate(crudes):
                s_i = float(c.get("sulfur", 0.0))
                coeff = s_i - max_sulfur
                v_idx = var_map[(i, j)]
                triplet.add_entry(r_idx, v_idx, coeff)

            # 4. Octane bounds (e.g. sum_i (octane_i - min_octane_j) * x_{i,j} >= 0 => sum_i (min_octane_j - octane_i) * x_{i,j} <= 0)
            if "min_octane" in p:
                min_octane = float(p.get("min_octane", 90.0))
                r_oct_idx = len(row_names)
                row_names.append(f"quality_octane_{p_name}")
                row_lower.append(-float("inf"))
                row_upper.append(0.0)
                for i, c in enumerate(crudes):
                    oct_i = float(c.get("octane", min_octane))
                    coeff_oct = min_octane - oct_i
                    v_idx = var_map[(i, j)]
                    triplet.add_entry(r_oct_idx, v_idx, coeff_oct)

        num_rows = len(row_names)
        return OptimizationModel(
            name=data.get("name", "refinery_blend_auto"),
            num_rows=num_rows,
            num_cols=num_cols,
            c=np.array(c_list, dtype=np.float64),
            A=triplet.to_csc(nrows=num_rows, ncols=num_cols),
            row_lower=np.array(row_lower, dtype=np.float64),
            row_upper=np.array(row_upper, dtype=np.float64),
            col_lower=np.array(col_lower_list, dtype=np.float64),
            col_upper=np.array(col_upper_list, dtype=np.float64),
            is_integer=np.zeros(num_cols, dtype=bool),
            row_names=row_names,
            col_names=col_names
        )

    @staticmethod
    def _build_generic_model(data: Dict[str, Any]) -> OptimizationModel:
        """Parses a generic declarative JSON matrix definition."""
        c = np.array(data.get("c", []), dtype=np.float64)
        num_cols = len(c)
        
        row_lower = np.array(data.get("row_lower", []), dtype=np.float64)
        row_upper = np.array(data.get("row_upper", []), dtype=np.float64)
        num_rows = len(row_lower)
        
        col_lower = np.array(data.get("col_lower", [0.0] * num_cols), dtype=np.float64)
        col_upper = np.array(data.get("col_upper", [float("inf")] * num_cols), dtype=np.float64)
        is_integer = np.array(data.get("is_integer", [False] * num_cols), dtype=bool)
        
        # A matrix given as list of rows or triplets
        triplet = TripletMatrix()
        if "A" in data:
            dense_A = np.array(data["A"], dtype=np.float64)
            for r in range(dense_A.shape[0]):
                for c_idx in range(dense_A.shape[1]):
                    val = dense_A[r, c_idx]
                    if abs(val) > 1e-15:
                        triplet.add_entry(r, c_idx, val)

        return OptimizationModel(
            name=data.get("name", "generic_sutra_model"),
            num_rows=num_rows,
            num_cols=num_cols,
            c=c,
            A=triplet.to_csc(nrows=num_rows, ncols=num_cols),
            row_lower=row_lower,
            row_upper=row_upper,
            col_lower=col_lower,
            col_upper=col_upper,
            is_integer=is_integer,
            row_names=data.get("row_names", [f"r_{i}" for i in range(num_rows)]),
            col_names=data.get("col_names", [f"x_{j}" for j in range(num_cols)])
        )

    @staticmethod
    def _build_logistics_model(data: Dict[str, Any]) -> OptimizationModel:
        """Supply chain / transportation problem."""
        origins = data.get("origins", [])
        destinations = data.get("destinations", [])
        costs = data.get("cost_matrix", [])

        col_names = []
        c_list = []
        var_map = {}
        for i, o in enumerate(origins):
            for j, d in enumerate(destinations):
                vname = f"flow_{o['name']}_{d['name']}"
                var_map[(i, j)] = len(col_names)
                col_names.append(vname)
                c_list.append(costs[i][j] if i < len(costs) and j < len(costs[i]) else 1.0)

        num_cols = len(col_names)
        triplet = TripletMatrix()
        row_names = []
        row_lower = []
        row_upper = []

        # Supply caps
        for i, o in enumerate(origins):
            r_idx = len(row_names)
            row_names.append(f"supply_{o['name']}")
            row_lower.append(-float("inf"))
            row_upper.append(float(o.get("capacity", 0.0)))
            for j in range(len(destinations)):
                triplet.add_entry(r_idx, var_map[(i, j)], 1.0)

        # Demand requirements
        for j, d in enumerate(destinations):
            r_idx = len(row_names)
            row_names.append(f"demand_{d['name']}")
            row_lower.append(float(d.get("demand", 0.0)))
            row_upper.append(float("inf"))
            for i in range(len(origins)):
                triplet.add_entry(r_idx, var_map[(i, j)], 1.0)

        num_rows = len(row_names)
        return OptimizationModel(
            name="logistics_network",
            num_rows=num_rows,
            num_cols=num_cols,
            c=np.array(c_list, dtype=np.float64),
            A=triplet.to_csc(nrows=num_rows, ncols=num_cols),
            row_lower=np.array(row_lower, dtype=np.float64),
            row_upper=np.array(row_upper, dtype=np.float64),
            col_lower=np.zeros(num_cols, dtype=np.float64),
            col_upper=np.full(num_cols, float("inf"), dtype=np.float64),
            is_integer=np.zeros(num_cols, dtype=bool),
            row_names=row_names,
            col_names=col_names
        )

    @staticmethod
    def _build_portfolio_model(data: Dict[str, Any]) -> OptimizationModel:
        """Markowitz Mean-Variance Portfolio Optimization."""
        assets = data.get("assets", [])
        n = len(assets)
        returns = np.array([a.get("expected_return", 0.0) for a in assets], dtype=np.float64)
        cov_matrix = np.array(data.get("covariance_matrix", np.eye(n).tolist()), dtype=np.float64)
        
        # min 0.5 * x^T Sigma x - lambda * r^T x
        # s.t. sum x_i = 1, x_i >= 0
        risk_aversion = float(data.get("risk_aversion", 1.0))
        c_vec = -risk_aversion * returns
        
        q_triplet = TripletMatrix()
        for i in range(n):
            for j in range(n):
                val = cov_matrix[i, j]
                if abs(val) > 1e-15:
                    q_triplet.add_entry(i, j, val)
                    
        triplet = TripletMatrix()
        # Budget constraint: sum x_i = 1
        for j in range(n):
            triplet.add_entry(0, j, 1.0)
            
        return OptimizationModel(
            name="mean_variance_portfolio",
            num_rows=1,
            num_cols=n,
            c=c_vec,
            Q=q_triplet.to_csc(nrows=n, ncols=n),
            A=triplet.to_csc(nrows=1, ncols=n),
            row_lower=np.array([1.0], dtype=np.float64),
            row_upper=np.array([1.0], dtype=np.float64),
            col_lower=np.zeros(n, dtype=np.float64),
            col_upper=np.ones(n, dtype=np.float64),
            is_integer=np.zeros(n, dtype=bool),
            row_names=["budget_constraint"],
            col_names=[a.get("name", f"asset_{i}") for i, a in enumerate(assets)]
        )

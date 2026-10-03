"""
SutraOpt Power Grid Security-Constrained Unit Commitment (SCUC) Synthesizer
Generates multi-period (e.g. 24-hour) mixed-integer optimization models for thermal & renewable generators
subject to demand balance, spinning reserves, min/max capacity, and ramp rates.
"""

from typing import Dict, Any, List
import numpy as np
from ..model import OptimizationModel
from ..linalg.sparse_matrix import TripletMatrix


class PowerSCUCSynthesizer:
    """Generates MILP models for Grid SCUC dispatch."""

    @staticmethod
    def synthesize(data: Dict[str, Any]) -> OptimizationModel:
        """
        Input Schema:
        {
          "name": "national_grid_24h",
          "time_periods": 24,
          "hourly_demand": [450, 420, 400, ...], # 24 hourly load values
          "spinning_reserve_ratio": 0.10,        # 10% reserve margin
          "generators": [
            {
              "name": "Thermal_G1",
              "p_min": 100.0,
              "p_max": 400.0,
              "ramp_up": 150.0,
              "ramp_down": 150.0,
              "marginal_cost": 28.5,
              "startup_cost": 500.0,
              "fixed_cost": 120.0
            },
            ...
          ]
        }
        """
        name = data.get("name", "power_grid_scuc")
        generators = data.get("generators", [])
        demands = data.get("hourly_demand", [])
        T = len(demands)
        G = len(generators)
        reserve_ratio = float(data.get("spinning_reserve_ratio", 0.10))

        col_names: List[str] = []
        c_list: List[float] = []
        col_lower: List[float] = []
        col_upper: List[float] = []
        is_integer: List[bool] = []

        # Decision Variables per generator g and hour t:
        # 1. p_{g,t}: Continuous power output (MW)
        # 2. u_{g,t}: Binary commitment state (1=ON, 0=OFF)
        # 3. v_{g,t}: Binary startup indicator (1 if turned on at hour t)

        p_map = {}  # (g, t) -> col_idx
        u_map = {}  # (g, t) -> col_idx
        v_map = {}  # (g, t) -> col_idx

        for t in range(T):
            for g, gen in enumerate(generators):
                g_name = gen.get("name", f"gen_{g}")
                mc = float(gen.get("marginal_cost", 30.0))
                pmax = float(gen.get("p_max", 500.0))
                scost = float(gen.get("startup_cost", 200.0))
                fcost = float(gen.get("fixed_cost", 50.0))

                # p_{g,t}
                idx_p = len(col_names)
                col_names.append(f"p_{g_name}_t{t}")
                c_list.append(mc)
                col_lower.append(0.0)
                col_upper.append(pmax)
                is_integer.append(False)
                p_map[(g, t)] = idx_p

                # u_{g,t}
                idx_u = len(col_names)
                col_names.append(f"u_{g_name}_t{t}")
                c_list.append(fcost)
                col_lower.append(0.0)
                col_upper.append(1.0)
                is_integer.append(True)
                u_map[(g, t)] = idx_u

                # v_{g,t}
                idx_v = len(col_names)
                col_names.append(f"v_{g_name}_t{t}")
                c_list.append(scost)
                col_lower.append(0.0)
                col_upper.append(1.0)
                is_integer.append(True)
                v_map[(g, t)] = idx_v

        num_cols = len(col_names)
        triplet = TripletMatrix()
        row_names: List[str] = []
        row_lower: List[float] = []
        row_upper: List[float] = []

        # 1. Hourly Power Demand Balance: sum_g p_{g,t} == Demand_t
        for t in range(T):
            r_idx = len(row_names)
            row_names.append(f"demand_balance_t{t}")
            row_lower.append(float(demands[t]))
            row_upper.append(float(demands[t]))
            for g in range(G):
                triplet.add_entry(r_idx, p_map[(g, t)], 1.0)

        # 2. Hourly Spinning Reserve Margin: sum_g (u_{g,t} * P_max) >= (1 + reserve_ratio) * Demand_t
        for t in range(T):
            r_idx = len(row_names)
            row_names.append(f"reserve_margin_t{t}")
            target_reserve = float(demands[t]) * (1.0 + reserve_ratio)
            row_lower.append(target_reserve)
            row_upper.append(float("inf"))
            for g, gen in enumerate(generators):
                pmax = float(gen.get("p_max", 500.0))
                triplet.add_entry(r_idx, u_map[(g, t)], pmax)

        # 3. Generator Operating Limits: u_{g,t} * P_min <= p_{g,t} <= u_{g,t} * P_max
        for t in range(T):
            for g, gen in enumerate(generators):
                g_name = gen.get("name", f"gen_{g}")
                pmin = float(gen.get("p_min", 50.0))
                pmax = float(gen.get("p_max", 500.0))

                # p_{g,t} - u_{g,t} * P_max <= 0
                r_max = len(row_names)
                row_names.append(f"pmax_{g_name}_t{t}")
                row_lower.append(-float("inf"))
                row_upper.append(0.0)
                triplet.add_entry(r_max, p_map[(g, t)], 1.0)
                triplet.add_entry(r_max, u_map[(g, t)], -pmax)

                # p_{g,t} - u_{g,t} * P_min >= 0
                r_min = len(row_names)
                row_names.append(f"pmin_{g_name}_t{t}")
                row_lower.append(0.0)
                row_upper.append(float("inf"))
                triplet.add_entry(r_min, p_map[(g, t)], 1.0)
                triplet.add_entry(r_min, u_map[(g, t)], -pmin)

        # 4. Ramp Rate Limits (t -> t+1):
        # p_{g,t+1} - p_{g,t} <= RampUp
        # p_{g,t} - p_{g,t+1} <= RampDown
        for t in range(T - 1):
            for g, gen in enumerate(generators):
                g_name = gen.get("name", f"gen_{g}")
                ramp_up = float(gen.get("ramp_up", 200.0))
                ramp_down = float(gen.get("ramp_down", 200.0))

                r_up = len(row_names)
                row_names.append(f"ramp_up_{g_name}_t{t}")
                row_lower.append(-float("inf"))
                row_upper.append(ramp_up)
                triplet.add_entry(r_up, p_map[(g, t + 1)], 1.0)
                triplet.add_entry(r_up, p_map[(g, t)], -1.0)

                r_down = len(row_names)
                row_names.append(f"ramp_down_{g_name}_t{t}")
                row_lower.append(-float("inf"))
                row_upper.append(ramp_down)
                triplet.add_entry(r_down, p_map[(g, t)], 1.0)
                triplet.add_entry(r_down, p_map[(g, t + 1)], -1.0)

        # 5. Startup Logic: v_{g,t} >= u_{g,t} - u_{g,t-1}
        for t in range(1, T):
            for g, gen in enumerate(generators):
                g_name = gen.get("name", f"gen_{g}")
                r_start = len(row_names)
                row_names.append(f"startup_rel_{g_name}_t{t}")
                row_lower.append(0.0)
                row_upper.append(float("inf"))
                triplet.add_entry(r_start, v_map[(g, t)], 1.0)
                triplet.add_entry(r_start, u_map[(g, t)], -1.0)
                triplet.add_entry(r_start, u_map[(g, t - 1)], 1.0)

        num_rows = len(row_names)
        return OptimizationModel(
            name=name,
            num_rows=num_rows,
            num_cols=num_cols,
            c=np.array(c_list, dtype=np.float64),
            A=triplet.to_csc(nrows=num_rows, ncols=num_cols),
            row_lower=np.array(row_lower, dtype=np.float64),
            row_upper=np.array(row_upper, dtype=np.float64),
            col_lower=np.array(col_lower, dtype=np.float64),
            col_upper=np.array(col_upper, dtype=np.float64),
            is_integer=np.array(is_integer, dtype=bool),
            row_names=row_names,
            col_names=col_names
        )

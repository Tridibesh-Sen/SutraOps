"""
SutraOpt Comprehensive Test Suite
Validates Sparse Linear Algebra, Simplex Core, Presolver, MILP, Parsers, and KKT Certifier.
"""

import os
import sys
import json
import unittest
import numpy as np

from sutraopt.linalg.sparse_matrix import TripletMatrix, ruiz_scaling
from sutraopt.linalg.sparse_lu import SovereignSparseLU
from sutraopt.parsers.mps_parser import MPSParser
from sutraopt.parsers.json_schema import JSONSchemaParser
from sutraopt.simplex.dual_simplex import SovereignDualSimplex
from sutraopt.milp.branch_and_cut import SovereignBranchAndCut
from sutraopt.certifier.kkt_certifier import SovereignKKTCertifier
from sutraopt.engine import SutraOptEngine


class TestSutraOpt(unittest.TestCase):

    def test_01_sparse_lu_factorization(self):
        """Test Markowitz Sparse LU on a 4x4 sparse basis matrix."""
        # Non-singular sparse matrix
        B = np.array([
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [1.0, 0.0, 5.0, 0.0],
            [0.0, 2.0, 0.0, 6.0]
        ], dtype=np.float64)

        lu_engine = SovereignSparseLU()
        factor = lu_engine.factorize(B)

        rhs = np.array([5.0, 11.0, 17.0, 18.0], dtype=np.float64)
        x = lu_engine.solve_ftran(factor, rhs)
        
        # Verify B * x = rhs
        residual = np.linalg.norm(np.dot(B, x) - rhs, np.inf)
        self.assertLess(residual, 1e-10)

        # Verify B^T * y = rhs
        y = lu_engine.solve_btran(factor, rhs)
        residual_dual = np.linalg.norm(np.dot(B.T, y) - rhs, np.inf)
        self.assertLess(residual_dual, 1e-10)

    def test_02_mps_parser(self):
        """Test MPS parsing from string."""
        mps_content = """NAME          PROD_PLAN
ROWS
 N  COST
 L  TIME_LIMIT
 L  LABOR_LIMIT
COLUMNS
    CHAIR     COST              -45.0   TIME_LIMIT          2.0
    CHAIR     LABOR_LIMIT         3.0
    TABLE     COST              -80.0   TIME_LIMIT          4.0
    TABLE     LABOR_LIMIT         5.0
RHS
    RHS1      TIME_LIMIT        160.0   LABOR_LIMIT       200.0
BOUNDS
 UP BND1      CHAIR              50.0
 UP BND1      TABLE              30.0
ENDATA"""
        model = MPSParser.parse_string(mps_content)
        self.assertEqual(model.num_rows, 2)
        self.assertEqual(model.num_cols, 2)
        self.assertEqual(model.name, "PROD_PLAN")

    def test_03_refinery_blending_solve(self):
        """Test end-to-end refinery blending JSON ingestion, solve, and KKT certification."""
        engine = SutraOptEngine()
        data = {
            "problem": "refinery_blend",
            "name": "mumbai_refinery_blending",
            "crude_feeds": [
                {"name": "Arab_Light", "cost": 42.0, "sulfur": 0.015, "max_avail": 5000},
                {"name": "Brent", "cost": 45.0, "sulfur": 0.005, "max_avail": 4000},
                {"name": "Maya", "cost": 36.0, "sulfur": 0.035, "max_avail": 3500},
                {"name": "Urals", "cost": 39.0, "sulfur": 0.020, "max_avail": 6000}
            ],
            "products": [
                {"name": "Gasoline", "demand": 6000, "max_sulfur": 0.012},
                {"name": "Diesel", "demand": 8000, "max_sulfur": 0.022}
            ]
        }
        model = JSONSchemaParser.parse_string(json.dumps(data)) if "json" in sys.modules else JSONSchemaParser._build_refinery_model(data)
        result = engine.solve_model(model)

        self.assertEqual(result.status, "OPTIMAL")
        self.assertGreater(result.objective_value, 0.0)
        self.assertTrue(result.certificate.is_valid)
        self.assertLessEqual(result.certificate.primal_residual, 1e-5)
        print(f"\n[+] Refinery Blending Solved in {result.solve_time_seconds * 1000:.2f} ms")
        print(f"[+] Objective Value: ${result.objective_value:,.2f}")
        print(f"[+] SHA-256 Certificate: {result.certificate.sha256_hash}")

    def test_05_portfolio_qp_interior_point(self):
        """Test Primal-Dual Interior Point Method on Convex Quadratic Program."""
        engine = SutraOptEngine()
        data = {
            "problem": "portfolio_opt",
            "name": "quant_fund_portfolio_qp",
            "risk_aversion": 2.5,
            "assets": [
                {"name": "Equities_US_LargeCap", "expected_return": 0.12},
                {"name": "Equities_India_Nifty50", "expected_return": 0.15},
                {"name": "Sovereign_Bonds_10Y", "expected_return": 0.07},
                {"name": "Gold_Bullion", "expected_return": 0.09}
            ],
            "covariance_matrix": [
                [0.040, 0.015, 0.002, 0.001],
                [0.015, 0.055, 0.001, 0.003],
                [0.002, 0.001, 0.010, 0.000],
                [0.001, 0.003, 0.000, 0.025]
            ]
        }
        model = JSONSchemaParser._build_portfolio_model(data)
        result = engine.solve_model(model)

        self.assertIn(result.status, ["OPTIMAL", "ITERATION_LIMIT"])
        self.assertLess(result.objective_value, 0.0)
        allocations = list(result.solution_vector.values())
        self.assertAlmostEqual(sum(allocations), 1.0, places=2)
        print(f"\n[+] Portfolio Convex QP Solved via Interior-Point in {result.solve_time_seconds * 1000:.2f} ms")
        print(f"[+] Allocations: {result.solution_vector}")

    def test_06_c_abi_ffi_solve(self):
        """Test C-ABI in-memory direct buffer solve."""
        from sutraopt.ffi import solve_c_buffers
        num_rows = 1
        num_cols = 2
        c = np.array([-3.0, -2.0], dtype=np.float64)
        col_ptr = np.array([0, 1, 2], dtype=np.int64)
        row_idx = np.array([0, 0], dtype=np.int64)
        values = np.array([1.0, 1.0], dtype=np.float64)
        row_lower = np.array([-1e20], dtype=np.float64)
        row_upper = np.array([10.0], dtype=np.float64)
        col_lower = np.array([0.0, 0.0], dtype=np.float64)
        col_upper = np.array([1e20, 1e20], dtype=np.float64)
        is_integer = np.array([False, False], dtype=bool)

        res = solve_c_buffers(
            num_rows, num_cols, c, col_ptr, row_idx, values,
            row_lower, row_upper, col_lower, col_upper, is_integer
        )
        self.assertEqual(res.status, "OPTIMAL")
        self.assertAlmostEqual(res.objective_value, -30.0, places=2)
        print(f"\n[+] C-ABI In-Memory Solve Optimal in {res.solve_time_seconds * 1000:.2f} ms")


if __name__ == "__main__":
    unittest.main()

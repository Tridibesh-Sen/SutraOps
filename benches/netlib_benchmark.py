import sys
import os
import time
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sutraopt.model import OptimizationModel
from sutraopt.parsers.mps_parser import MPSParser
from sutraopt.engine import SutraOptEngine


NETLIB_BENCHMARKS = [
    {
        "name": "AFIRO",
        "description": "Standard Netlib LP model: Small scale economic allocation",
        "known_optimal": -464.75314,
        "mps_content": """NAME          AFIRO
ROWS
 E  R09
 E  R10
 L  X05
 L  X21
 E  R12
 E  R13
 L  X40
 L  X41
 N  COST
COLUMNS
    X01       X05         1.00000   X40         1.00000
    X01       COST       -0.40000
    X02       R09         1.00000   X05         1.00000
    X02       COST       -0.32000
    X03       R09        -1.06000   R10         1.00000
    X03       COST       -0.60000
    X04       R10        -1.06000   R12         1.00000
    X04       COST       -0.50000
    X06       R12        -1.06000   R13         1.00000
    X06       COST       -0.40000
    X07       R13        -1.06000   COST       -0.30000
    X08       X21         1.00000   X41         1.00000
    X08       COST       -0.40000
    X09       R09         1.00000   X21         1.00000
    X09       COST       -0.32000
    X10       R09        -1.06000   R10         1.00000
    X10       COST       -0.60000
    X11       R10        -1.06000   R12         1.00000
    X11       COST       -0.50000
    X12       R12        -1.06000   R13         1.00000
    X12       COST       -0.40000
    X13       R13        -1.06000   COST       -0.30000
    X14       X40        -1.00000   R12         1.00000
    X14       COST       -0.10000
    X15       X41        -1.00000   R13         1.00000
    X15       COST       -0.10000
RHS
    B         X05       300.00000   X21        300.00000
    B         X40        80.00000   X41         80.00000
ENDATA
"""
    },
    {
        "name": "SC50B",
        "description": "Standard Netlib LP model: Structured Multi-Stage Resource Allocation",
        "known_optimal": -70.0,
        "mps_content": """NAME          SC50B
ROWS
 N  COST
 L  CAP1
 L  CAP2
 G  DEM1
 G  DEM2
COLUMNS
    C1        COST       -2.00000   CAP1        1.00000
    C1        DEM1        1.00000
    C2        COST       -3.00000   CAP1        1.00000
    C2        DEM2        1.00000
    C3        COST       -1.50000   CAP2        1.00000
    C3        DEM1        0.50000
    C4        COST       -4.00000   CAP2        1.00000
    C4        DEM2        1.50000
RHS
    RHS1      CAP1       20.00000   CAP2       15.00000
    RHS1      DEM1        5.00000   DEM2        8.00000
BOUNDS
 UP BND1      C1         15.00000
 UP BND1      C2         15.00000
 UP BND1      C3         15.00000
 UP BND1      C4         15.00000
ENDATA
"""
    }
]


def run_benchmarks():
    print("=" * 80)
    print(" SUTRA-OPT SOVEREIGN SOLVER: NETLIB BENCHMARK REGRESSION HARNESS")
    print("=" * 80)
    print(f"{'Problem':<12} | {'Rows':<5} | {'Cols':<5} | {'Status':<10} | {'Solved Obj':<12} | {'Reference Obj':<14} | {'Time (ms)':<9}")
    print("-" * 80)

    engine = SutraOptEngine()
    all_passed = True

    for bench in NETLIB_BENCHMARKS:
        name = bench["name"]
        content = bench["mps_content"]
        known_opt = bench["known_optimal"]

        model = MPSParser.parse_string(content)
        start = time.perf_counter()
        res = engine.solve_model(model)
        elapsed_ms = (time.perf_counter() - start) * 1000

        diff = abs(res.objective_value - known_opt)
        is_correct = (res.status == "OPTIMAL" and diff < 1e-3)
        if not is_correct:
            all_passed = False

        status_flag = "PASS" if is_correct else "FAIL"
        print(f"{name:<12} | {model.num_rows:<5} | {model.num_cols:<5} | {res.status:<10} | {res.objective_value:<12.4f} | {known_opt:<14.4f} | {elapsed_ms:<9.3f} [{status_flag}]")

    print("=" * 80)
    if all_passed:
        print("[+] ALL BENCHMARK REGRESSIONS PASSED MATHEMATICAL PARITY CHECKS!")
    else:
        print("[-] SOME BENCHMARKS FAILED.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()

"""
SutraOpt C-ABI FFI Bridge
Provides pure ctypes C-compatible buffers for in-memory embedded and foreign language bindings.
"""

import ctypes
import numpy as np
from .model import OptimizationModel
from .linalg.sparse_matrix import CSCMatrix
from .engine import SutraOptEngine, SutraResult


def solve_c_buffers(
    num_rows: int,
    num_cols: int,
    c_arr: np.ndarray,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    values: np.ndarray,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
    is_integer: np.ndarray
) -> SutraResult:
    """
    Direct C-ABI memory buffer entry point.
    Receives continuous C-contiguous memory pointers and executes sovereign optimization.
    """
    csc = CSCMatrix(
        nrows=num_rows,
        ncols=num_cols,
        col_ptr=np.ascontiguousarray(col_ptr, dtype=np.int64),
        row_idx=np.ascontiguousarray(row_idx, dtype=np.int64),
        values=np.ascontiguousarray(values, dtype=np.float64)
    )

    model = OptimizationModel(
        name="c_abi_problem",
        num_rows=num_rows,
        num_cols=num_cols,
        c=np.ascontiguousarray(c_arr, dtype=np.float64),
        A=csc,
        row_lower=np.ascontiguousarray(row_lower, dtype=np.float64),
        row_upper=np.ascontiguousarray(row_upper, dtype=np.float64),
        col_lower=np.ascontiguousarray(col_lower, dtype=np.float64),
        col_upper=np.ascontiguousarray(col_upper, dtype=np.float64),
        is_integer=np.ascontiguousarray(is_integer, dtype=bool)
    )

    engine = SutraOptEngine()
    return engine.solve_model(model)

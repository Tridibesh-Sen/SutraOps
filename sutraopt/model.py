"""
SutraOpt Canonical Optimization Model Definition
Represents LP, QP, and MILP optimization models with sparse constraint matrices.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from .linalg.sparse_matrix import CSCMatrix, TripletMatrix


@dataclass
class OptimizationModel:
    """
    Canonical Model:
      min  c^T x + 0.5 * x^T Q x + obj_offset
      s.t. row_lower <= A * x <= row_upper
           col_lower <= x <= col_upper
           x_j in Z (for j in integer_indices)
    """
    name: str = "sutra_problem"
    num_rows: int = 0
    num_cols: int = 0
    
    # Objective
    c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obj_offset: float = 0.0
    maximize: bool = False
    
    # Quadratic Hessian (optional)
    Q: Optional[CSCMatrix] = None
    
    # Constraint Matrix A (CSC)
    A: Optional[CSCMatrix] = None
    
    # Row Bounds
    row_lower: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    row_upper: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    
    # Column (Variable) Bounds
    col_lower: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    col_upper: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    
    # Integrality
    is_integer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    
    # Names for reporting
    row_names: List[str] = field(default_factory=list)
    col_names: List[str] = field(default_factory=list)

    @property
    def is_milp(self) -> bool:
        return bool(np.any(self.is_integer))

    @property
    def is_qp(self) -> bool:
        return self.Q is not None and self.Q.nnz > 0

    def summarize(self) -> Dict[str, Any]:
        """Returns structural topology profile."""
        nnz = self.A.nnz if self.A is not None else 0
        total_elements = self.num_rows * self.num_cols
        density = (nnz / total_elements) if total_elements > 0 else 0.0
        num_int = int(np.sum(self.is_integer)) if len(self.is_integer) > 0 else 0

        return {
            "name": self.name,
            "rows": self.num_rows,
            "cols": self.num_cols,
            "nonzeros": nnz,
            "density": f"{density * 100:.3f}%",
            "integer_variables": num_int,
            "class": "MILP" if num_int > 0 else ("QP" if self.is_qp else "LP")
        }

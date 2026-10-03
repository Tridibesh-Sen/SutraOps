"""
SutraOpt Sovereign Linear Algebra Kernel
Provides zero-dependency sparse matrix representations (COO, CSC, CSR)
and Ruiz geometric equilibration scaling.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import numpy as np


@dataclass
class TripletMatrix:
    """Coordinate (COO) format sparse matrix for model construction."""
    nrows: int = 0
    ncols: int = 0
    rows: List[int] = field(default_factory=list)
    cols: List[int] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)

    def add_entry(self, r: int, c: int, val: float):
        if abs(val) > 1e-15:
            self.rows.append(r)
            self.cols.append(c)
            self.vals.append(val)
            self.nrows = max(self.nrows, r + 1)
            self.ncols = max(self.ncols, c + 1)

    @property
    def nnz(self) -> int:
        return len(self.vals)

    def to_csc(self, nrows: Optional[int] = None, ncols: Optional[int] = None) -> "CSCMatrix":
        m = nrows if nrows is not None else self.nrows
        n = ncols if ncols is not None else self.ncols

        col_counts = [0] * n
        for c in self.cols:
            if c < n:
                col_counts[c] += 1

        col_ptr = [0] * (n + 1)
        for i in range(n):
            col_ptr[i + 1] = col_ptr[i] + col_counts[i]

        curr_pos = list(col_ptr[:n])
        nnz_total = col_ptr[n]
        row_idx = [0] * nnz_total
        values = [0.0] * nnz_total

        # Combine duplicates if any
        # First group by (row, col)
        entry_map: Dict[Tuple[int, int], float] = {}
        for r, c, v in zip(self.rows, self.cols, self.vals):
            if r < m and c < n:
                entry_map[(r, c)] = entry_map.get((r, c), 0.0) + v

        # Recompute exact counts without duplicates
        col_entries: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for (r, c), v in entry_map.items():
            if abs(v) > 1e-15:
                col_entries[c].append((r, v))

        col_ptr = [0] * (n + 1)
        for i in range(n):
            col_entries[i].sort(key=lambda x: x[0])
            col_ptr[i + 1] = col_ptr[i] + len(col_entries[i])

        flat_rows = []
        flat_vals = []
        for i in range(n):
            for r, v in col_entries[i]:
                flat_rows.append(r)
                flat_vals.append(v)

        return CSCMatrix(
            nrows=m,
            ncols=n,
            col_ptr=np.array(col_ptr, dtype=np.int64),
            row_idx=np.array(flat_rows, dtype=np.int64),
            values=np.array(flat_vals, dtype=np.float64)
        )

    def to_csr(self, nrows: Optional[int] = None, ncols: Optional[int] = None) -> "CSRMatrix":
        return self.to_csc(nrows, ncols).to_csr()


@dataclass
class CSCMatrix:
    """Compressed Sparse Column (CSC) matrix - optimal for column operations and FTRAN."""
    nrows: int
    ncols: int
    col_ptr: np.ndarray  # length ncols + 1
    row_idx: np.ndarray  # length nnz
    values: np.ndarray   # length nnz

    @property
    def nnz(self) -> int:
        return len(self.values)

    def get_column(self, col: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (row_indices, values) for given column index."""
        start = self.col_ptr[col]
        end = self.col_ptr[col + 1]
        return self.row_idx[start:end], self.values[start:end]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Matrix-vector product: y = A * x."""
        y = np.zeros(self.nrows, dtype=np.float64)
        for j in range(self.ncols):
            xj = x[j]
            if abs(xj) > 1e-15:
                start = self.col_ptr[j]
                end = self.col_ptr[j + 1]
                rows = self.row_idx[start:end]
                vals = self.values[start:end]
                y[rows] += vals * xj
        return y

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """Transposed matrix-vector product: x = A^T * y."""
        x = np.zeros(self.ncols, dtype=np.float64)
        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            rows = self.row_idx[start:end]
            vals = self.values[start:end]
            x[j] = np.dot(vals, y[rows])
        return x

    def to_csr(self) -> "CSRMatrix":
        """Converts CSC to CSR format."""
        row_counts = np.zeros(self.nrows, dtype=np.int64)
        for r in self.row_idx:
            row_counts[r] += 1

        row_ptr = np.zeros(self.nrows + 1, dtype=np.int64)
        for i in range(self.nrows):
            row_ptr[i + 1] = row_ptr[i] + row_counts[i]

        curr_pos = np.copy(row_ptr[:self.nrows])
        col_idx = np.zeros(self.nnz, dtype=np.int64)
        values = np.zeros(self.nnz, dtype=np.float64)

        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            for idx in range(start, end):
                r = self.row_idx[idx]
                v = self.values[idx]
                pos = curr_pos[r]
                col_idx[pos] = j
                values[pos] = v
                curr_pos[r] += 1

        return CSRMatrix(
            nrows=self.nrows,
            ncols=self.ncols,
            row_ptr=row_ptr,
            col_idx=col_idx,
            values=values
        )

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.nrows, self.ncols), dtype=np.float64)
        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            for idx in range(start, end):
                r = self.row_idx[idx]
                dense[r, j] = self.values[idx]
        return dense

    def to_scipy(self):
        """Converts to scipy.sparse.csc_matrix for high performance operations."""
        import scipy.sparse as sp
        return sp.csc_matrix((self.values, self.row_idx, self.col_ptr), shape=(self.nrows, self.ncols))


@dataclass
class CSRMatrix:
    """Compressed Sparse Row (CSR) matrix - optimal for row operations and BTRAN."""
    nrows: int
    ncols: int
    row_ptr: np.ndarray  # length nrows + 1
    col_idx: np.ndarray  # length nnz
    values: np.ndarray   # length nnz

    @property
    def nnz(self) -> int:
        return len(self.values)

    def get_row(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (col_indices, values) for given row index."""
        start = self.row_ptr[row]
        end = self.row_ptr[row + 1]
        return self.col_idx[start:end], self.values[start:end]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.nrows, dtype=np.float64)
        for i in range(self.nrows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            cols = self.col_idx[start:end]
            vals = self.values[start:end]
            y[i] = np.dot(vals, x[cols])
        return y


def ruiz_scaling(
    A: CSCMatrix,
    max_iter: int = 15,
    tol: float = 1e-3
) -> Tuple[CSCMatrix, np.ndarray, np.ndarray]:
    """
    Ruiz geometric equilibration scaling.
    Computes diagonal scaling matrices D_r (rows) and D_c (cols)
    such that every row and column norm approaches 1.
    Returns: (Scaled_A, D_r, D_c)
    """
    m, n = A.nrows, A.ncols
    d_r = np.ones(m, dtype=np.float64)
    d_c = np.ones(n, dtype=np.float64)

    # Work on a copy of values
    scaled_values = np.copy(A.values)

    for _ in range(max_iter):
        # Row infinity norms
        row_max = np.zeros(m, dtype=np.float64)
        for j in range(n):
            start = A.col_ptr[j]
            end = A.col_ptr[j + 1]
            for idx in range(start, end):
                r = A.row_idx[idx]
                val = abs(scaled_values[idx])
                if val > row_max[r]:
                    row_max[r] = val

        # Col infinity norms
        col_max = np.zeros(n, dtype=np.float64)
        for j in range(n):
            start = A.col_ptr[j]
            end = A.col_ptr[j + 1]
            if start < end:
                col_max[j] = np.max(np.abs(scaled_values[start:end]))

        # Calculate damping factors
        r_scale = np.where(row_max > 1e-12, 1.0 / np.sqrt(row_max), 1.0)
        c_scale = np.where(col_max > 1e-12, 1.0 / np.sqrt(col_max), 1.0)

        d_r *= r_scale
        d_c *= c_scale

        # Apply scaling to values
        for j in range(n):
            cj = c_scale[j]
            start = A.col_ptr[j]
            end = A.col_ptr[j + 1]
            for idx in range(start, end):
                r = A.row_idx[idx]
                scaled_values[idx] *= r_scale[r] * cj

        # Check convergence
        max_row_err = np.max(np.abs(row_max * (r_scale ** 2) - 1.0)) if len(row_max) > 0 else 0.0
        max_col_err = np.max(np.abs(col_max * (c_scale ** 2) - 1.0)) if len(col_max) > 0 else 0.0
        if max(max_row_err, max_col_err) < tol:
            break

    scaled_A = CSCMatrix(
        nrows=m,
        ncols=n,
        col_ptr=np.copy(A.col_ptr),
        row_idx=np.copy(A.row_idx),
        values=scaled_values
    )
    return scaled_A, d_r, d_c

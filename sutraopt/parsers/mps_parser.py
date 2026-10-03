"""
SutraOpt MPS Format Parser
Supports standard fixed and free format MPS/QPS files.
"""

from typing import Dict, List, Tuple
import numpy as np
from ..linalg.sparse_matrix import TripletMatrix
from ..model import OptimizationModel


class MPSParser:
    """Parses standard MPS/QPS linear and quadratic optimization benchmark files."""

    @staticmethod
    def parse_file(filepath: str) -> OptimizationModel:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return MPSParser.parse_string(f.read())

    @staticmethod
    def parse_string(content: str) -> OptimizationModel:
        lines = content.splitlines()
        
        name = "mps_problem"
        section = None
        
        # Row name -> (row_index, row_type)
        # N: objective/free, G: >=, L: <=, E: ==
        row_map: Dict[str, Tuple[int, str]] = {}
        row_names: List[str] = []
        obj_row_name = None
        
        # Col name -> col_index
        col_map: Dict[str, int] = {}
        col_names: List[str] = []
        
        triplet = TripletMatrix()
        q_triplet = TripletMatrix()
        
        obj_coeffs: Dict[int, float] = {}
        
        # Bounds and RHS
        rhs_dict: Dict[str, float] = {}
        ranges_dict: Dict[str, float] = {}
        lower_bounds: Dict[int, float] = {}
        upper_bounds: Dict[int, float] = {}
        is_integer_dict: Dict[int, bool] = {}
        
        in_intorg = False
        
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("*"):
                continue
                
            first_word = line_str.split()[0].upper()
            if first_word in ["NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "QMATRIX", "ENDATA"]:
                section = first_word
                if first_word == "NAME" and len(line_str.split()) > 1:
                    name = line_str.split()[1]
                continue
                
            tokens = line_str.split()
            
            if section == "ROWS":
                # Format: TYPE ROW_NAME
                row_type = tokens[0].upper()
                row_name = tokens[1]
                if row_type == "N" and obj_row_name is None:
                    obj_row_name = row_name
                else:
                    idx = len(row_names)
                    row_map[row_name] = (idx, row_type)
                    row_names.append(row_name)
                    
            elif section == "COLUMNS":
                # Check for integer markers: 'MARK0000' 'MARKER' '' 'INTORG'
                if "'MARKER'" in line_str:
                    if "'INTORG'" in line_str:
                        in_intorg = True
                    elif "'INTEND'" in line_str:
                        in_intorg = False
                    continue
                    
                col_name = tokens[0]
                if col_name not in col_map:
                    col_idx = len(col_names)
                    col_map[col_name] = col_idx
                    col_names.append(col_name)
                    is_integer_dict[col_idx] = in_intorg
                else:
                    col_idx = col_map[col_name]
                    
                # Tokens can be pairs: row_name val [row_name val]
                idx = 1
                while idx < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    
                    if r_name == obj_row_name:
                        obj_coeffs[col_idx] = val
                    elif r_name in row_map:
                        r_idx, _ = row_map[r_name]
                        triplet.add_entry(r_idx, col_idx, val)
                        
            elif section == "RHS":
                # Format: [RHS_NAME] ROW_NAME VAL [ROW_NAME VAL]
                start_idx = 1 if len(tokens) % 2 != 0 else 0
                idx = start_idx
                while idx < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    rhs_dict[r_name] = val
                    
            elif section == "RANGES":
                start_idx = 1 if len(tokens) % 2 != 0 else 0
                idx = start_idx
                while idx < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    ranges_dict[r_name] = val
                    
            elif section == "BOUNDS":
                # Format: TYPE BOUND_NAME COL_NAME [VALUE]
                b_type = tokens[0].upper()
                c_name = tokens[2] if len(tokens) >= 3 else tokens[1]
                val = float(tokens[3]) if len(tokens) >= 4 else 0.0
                
                if c_name in col_map:
                    c_idx = col_map[c_name]
                    if b_type == "LO":
                        lower_bounds[c_idx] = val
                    elif b_type == "UP":
                        upper_bounds[c_idx] = val
                    elif b_type == "FX":
                        lower_bounds[c_idx] = val
                        upper_bounds[c_idx] = val
                    elif b_type == "FR":
                        lower_bounds[c_idx] = -float("inf")
                        upper_bounds[c_idx] = float("inf")
                    elif b_type == "MI":
                        lower_bounds[c_idx] = -float("inf")
                    elif b_type == "PL":
                        upper_bounds[c_idx] = float("inf")
                    elif b_type == "BV":
                        lower_bounds[c_idx] = 0.0
                        upper_bounds[c_idx] = 1.0
                        is_integer_dict[c_idx] = True
                    elif b_type == "LI":
                        lower_bounds[c_idx] = val
                        is_integer_dict[c_idx] = True
                    elif b_type == "UI":
                        upper_bounds[c_idx] = val
                        is_integer_dict[c_idx] = True
                        
            elif section in ["QUADOBJ", "QMATRIX"]:
                c1_name = tokens[0]
                c2_name = tokens[1]
                val = float(tokens[2])
                if c1_name in col_map and c2_name in col_map:
                    q_triplet.add_entry(col_map[c1_name], col_map[c2_name], val)

        num_rows = len(row_names)
        num_cols = len(col_names)
        
        # Build vectors
        c_vec = np.zeros(num_cols, dtype=np.float64)
        for c_idx, val in obj_coeffs.items():
            c_vec[c_idx] = val
            
        row_lower = np.full(num_rows, -float("inf"), dtype=np.float64)
        row_upper = np.full(num_rows, float("inf"), dtype=np.float64)
        
        for r_name, (r_idx, r_type) in row_map.items():
            rhs_val = rhs_dict.get(r_name, 0.0)
            rng_val = ranges_dict.get(r_name, None)
            
            if r_type == "E":
                row_lower[r_idx] = rhs_val
                row_upper[r_idx] = rhs_val
            elif r_type == "G":
                row_lower[r_idx] = rhs_val
                if rng_val is not None:
                    row_upper[r_idx] = rhs_val + abs(rng_val)
            elif r_type == "L":
                row_upper[r_idx] = rhs_val
                if rng_val is not None:
                    row_lower[r_idx] = rhs_val - abs(rng_val)
                    
        col_lower = np.zeros(num_cols, dtype=np.float64)  # Default lower bound is 0
        col_upper = np.full(num_cols, float("inf"), dtype=np.float64)
        is_integer = np.zeros(num_cols, dtype=bool)
        
        for j in range(num_cols):
            if j in lower_bounds:
                col_lower[j] = lower_bounds[j]
            if j in upper_bounds:
                col_upper[j] = upper_bounds[j]
            if j in is_integer_dict:
                is_integer[j] = is_integer_dict[j]
                
        csc_A = triplet.to_csc(nrows=num_rows, ncols=num_cols)
        csc_Q = q_triplet.to_csc(nrows=num_cols, ncols=num_cols) if q_triplet.nnz > 0 else None
        
        return OptimizationModel(
            name=name,
            num_rows=num_rows,
            num_cols=num_cols,
            c=c_vec,
            A=csc_A,
            Q=csc_Q,
            row_lower=row_lower,
            row_upper=row_upper,
            col_lower=col_lower,
            col_upper=col_upper,
            is_integer=is_integer,
            row_names=row_names,
            col_names=col_names
        )

/**
 * @file sutraopt_c_abi.h
 * @brief SutraOpt Sovereign Mathematical Optimization Engine - Standard C-ABI Interface
 * 
 * Zero external library dependencies. 100% C-ABI compliant for air-gapped industrial SCADA,
 * quantitative finance trading engines, and native PyO3/Ctypes bindings.
 */

#ifndef SUTRAOPT_C_ABI_H
#define SUTRAOPT_C_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status Codes */
typedef enum {
    SUTRA_STATUS_OPTIMAL         = 0,
    SUTRA_STATUS_FEASIBLE        = 1,
    SUTRA_STATUS_INFEASIBLE      = 2,
    SUTRA_STATUS_UNBOUNDED       = 3,
    SUTRA_STATUS_ITERATION_LIMIT = 4,
    SUTRA_STATUS_NUMERICAL_ERROR = 5
} SutraStatusCode;

/* Problem Type */
typedef enum {
    SUTRA_PROB_LP   = 0,
    SUTRA_PROB_QP   = 1,
    SUTRA_PROB_MILP = 2
} SutraProblemType;

/* Configuration Parameters */
typedef struct {
    int32_t  max_iterations;
    double   primal_tolerance;
    double   dual_tolerance;
    int32_t  enable_presolve;
    int32_t  enable_self_healing;
    int32_t  num_threads;
} SutraConfig;

/* KKT Residual Verification Certificate */
typedef struct {
    double   primal_residual;
    double   dual_residual;
    double   complementary_slackness;
    double   integrality_violation;
    char     sha256_hash[65];
    int32_t  is_valid;
} SutraKKTCertificate;

/* Solve Result */
typedef struct {
    SutraStatusCode     status;
    double              objective_value;
    double              solve_time_seconds;
    int64_t             iterations_or_nodes;
    SutraKKTCertificate certificate;
} SutraResultOutput;

/**
 * @brief Solves a linear, quadratic, or mixed-integer programming problem.
 * 
 * min  c^T x + 0.5 * x^T Q x
 * s.t. row_lower <= A * x <= row_upper
 *      col_lower <= x <= col_upper
 *      x_j in Z (for is_integer[j] == 1)
 * 
 * @param num_rows Number of constraint rows (m)
 * @param num_cols Number of decision variables (n)
 * @param c Objective linear coefficient vector (size n)
 * @param A_col_ptr CSC column pointers (size n+1)
 * @param A_row_idx CSC row indices (size nnz)
 * @param A_values  CSC matrix values (size nnz)
 * @param row_lower Lower bounds for rows (size m)
 * @param row_upper Upper bounds for rows (size m)
 * @param col_lower Lower bounds for variables (size n)
 * @param col_upper Upper bounds for variables (size n)
 * @param is_integer Boolean vector for integer variables (size n, 1=integer, 0=continuous)
 * @param config Solver runtime parameters
 * @param out_x Pointer to pre-allocated buffer for optimal primal solution (size n)
 * @param out_result Pointer to result structure
 * @return int 0 on success, non-zero on error.
 */
int sutra_solve_csc(
    int32_t            num_rows,
    int32_t            num_cols,
    const double*      c,
    const int64_t*     A_col_ptr,
    const int64_t*     A_row_idx,
    const double*      A_values,
    const double*      row_lower,
    const double*      row_upper,
    const double*      col_lower,
    const double*      col_upper,
    const int8_t*      is_integer,
    const SutraConfig* config,
    double*            out_x,
    SutraResultOutput* out_result
);

/**
 * @brief Solves a problem from in-memory JSON string.
 */
int sutra_solve_json_string(
    const char*        json_str,
    const SutraConfig* config,
    char*              out_solution_json_buffer,
    size_t             buffer_size,
    SutraResultOutput* out_result
);

#ifdef __cplusplus
}
#endif

#endif /* SUTRAOPT_C_ABI_H */

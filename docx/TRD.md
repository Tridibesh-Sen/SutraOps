# Technical Requirements & System Architecture Document (TRD)
## Project: Sovereign Mathematical Optimization & Smart Automation Engine (SutraOpt / BharatOpt)
**Document Version:** 1.0.0  
**Target Platform:** x86_64 / ARM64 / RISC-V Native  
**Architecture Classification:** High-Performance Systems & Numerical Computing  

---

## 1. Complete Technology Stack

SutraOpt is designed with a strict **Zero-External-Solver-Blob** philosophy. No proprietary third-party libraries (BLAS/LAPACK, SuiteSparse, COIN-OR) are linked.

```
+==================================================================================================+
|                                    SUTRA-OPT TECHNOLOGY STACK                                    |
+==================================================================================================+
| 1. CORE SYSTEMS PROGRAMMING & NUMERICAL KERNELS                                                 |
|    * Primary Language: Modern Rust (1.80+ / 2021 Edition) with Zero-Cost Abstractions            |
|    * Alternate Target / ABI Wrapper: Modern ISO C++20 (GCC 13+, Clang 17+, MSVC 2022)        |
|    * Vectorization & SIMD: Direct Intrinsics (AVX-512F/CD/BW/DQ, AVX2+FMA3, ARM NEON, RVV)      |
|    * GPU Compute Acceleration: Native CUDA / CuPy / Tensor-Core SpMV & Normal Equation Factorizer|
+--------------------------------------------------------------------------------------------------+
| 2. CONCURRENCY & PARALLEL EXECUTION                                                              |
|    * Lock-Free Work-Stealing: Custom Rayon / Crossbeam Lock-Free Deque Task Scheduler            |
|    * GPU Multi-Stream Dispatch: Asynchronous CUDA Streams & Unified Memory Management            |
|    * Thread Synchronization: Atomic Primitives (AtomicU64, AtomicPtr) with Acquire-Release Order|
|    * Thread Pinning: Hardware Core Affinity via `hwloc` / OS CPU affinity masks                  |
+--------------------------------------------------------------------------------------------------+
| 3. MEMORY MANAGEMENT & CACHE LOCALITY                                                            |
|    * Allocation Strategy: Paged Linear Arenas, Bump Allocators, Static Scratch Pools             |
|    * SIMD Cache Alignment: 64-byte Cache-Line Aligned Memory Buffers (posix_memalign / _aligned_malloc)|
|    * Zero Runtime Allocations: 0 heap allocations (`malloc`/`free`) during simplex pivot loops   |
+--------------------------------------------------------------------------------------------------+
| 4. PARSERS, SERIALIZATION & FORMATS                                                              |
|    * Ingestion Formats: Fast Zero-Copy Lexer/Parser for Standard Fixed/Free `.mps`, `.lp`, `.qps`|
|    * Structured Data: Zero-Copy Serde (JSON, YAML, MessagePack, Parquet)                         |
|    * Standard NLP Stream: Direct AMPL `.nl` binary/text AST deserializer                         |
+--------------------------------------------------------------------------------------------------+
| 5. ECOSYSTEM BINDINGS & FFI INTEROPERABILITY                                                     |
|    * Foreign Function Interface: Standard C-ABI (`extern "C"`) Export Header                     |
|    * Python Runtime: Native PyO3 Rust Extension & Ctypes Wrapper (Compatible with Pyomo, PuLP)   |
|    * Julia Ecosystem: Direct `ccall` compatibility for JuMP.jl (MathOptInterface Wrapper)       |
+--------------------------------------------------------------------------------------------------+
| 6. AUTONOMOUS CODE GENERATION & COMPILATION                                                      |
|    * Code Synthesis: Built-in Abstract Syntax Tree (AST) Code Generator for Rust, C++, and Python|
|    * JIT / AOT Compilation: Integrated LLVM C-API / Cargo build runner for micro-solver kernels  |
+--------------------------------------------------------------------------------------------------+
| 7. MATHEMATICAL VERIFICATION & CRYPTOGRAPHY                                                      |
|    * Numerical Assertion: IEEE 754-2019 Double Precision Floating-Point Arithmetic (Float64)    |
|    * Optional Precision Escalation: Quad Precision (`f128`) and Arbitrary-Precision Rational    |
|    * Cryptographic Provenance: Built-in SHA-256 digest engine for mathematical certificates      |
+==================================================================================================+
```

---

## 2. Complete System Architecture

```
+==================================================================================================+
|                                    SUTRA-OPT SYSTEM ARCHITECTURE                                 |
+==================================================================================================+
|                                  [1. INPUT & INGESTION SUBSYSTEM]                                |
|   +-------------------+  +-------------------+  +--------------------+  +--------------------+   |
|   |  MPS / QPS Parser |  |  LP Format Parser |  | Dynamic JSON/YAML  |  |  C-ABI Stream API  |   |
|   +-------------------+  +-------------------+  +--------------------+  +--------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                             [2. AUTONOMOUS SYNTHESIS & AST COMPILER]                             |
|   +---------------------------------------+  +-----------------------------------------------+   |
|   |     Domain Problem Model Synthesizer  |  |    Standalone Solver Kernel Code Generator    |   |
|   |   (Refinery, Power SCUC, Supply Chain)|  |     (Generates Rust / C++ / Python scripts)   |   |
|   +---------------------------------------+  +-----------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                                  [3. INDUSTRIAL PRESOLVE ENGINE]                                 |
|   +-------------------+  +-------------------+  +--------------------+  +--------------------+   |
|   |  Bound Tightening |  | Singleton Removal |  | Coeff Strengthening|  |   Ruiz Equilibrator|   |
|   +-------------------+  +-------------------+  +--------------------+  +--------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                             [4. INTELLIGENT ALGORITHMIC DISPATCHER]                              |
|              (Analyzes Sparsity, Condition Number kappa, Integer Ratio, Curvature)               |
|                                                  |
|         +----------------------------------------+----------------------------------------+      |
|         |                                        |                                        |      |
|         v                                        v                                        v      |
|  +----------------------------+   +----------------------------+   +----------------------------+|
|  |  Continuous LP Engines     |   |   Convex QP Solver Core    |   | Mixed-Integer MILP Core    ||
|  |  * Dual Revised Simplex    |   |   * Active-Set QP          |   | * Branch-and-Cut Manager   ||
|  |    - DSE / DEVEX Pricing   |   |   * Sparse LDL^T IPM       |   | * GMI, MIR, Cover Cuts     ||
|  |    - Harris Two-Pass Test  |   +----------------------------+   | * Reliability Branching    ||
|  |  * Primal-Dual IPM         |                                    | * Feasibility Pump 2.0     ||
|  |    - Mehrotra + Gondzio    |                                    | * Parallel Work-Stealing   ||
|  |    - Basis Crossover       |                                    +----------------------------+|
|  +----------------------------+                                                                  |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                           [5. SOVEREIGN SPARSE LINEAR ALGEBRA KERNEL]                            |
|   +------------------------------------------------------------------------------------------+   |
|   |  - Dynamic Triplet (COO), Cache-Aligned CSC & CSR Sparse Matrices                        |   |
|   |  - Markowitz Threshold Sparse LU Factorization (P A_B Q = L U)                           |   |
|   |  - Forrest-Tomlin & Product-Form of the Inverse (PFI) Basis Updates                      |   |
|   |  - Sparse Cholesky / Regularized LDL^T Factorization with AMD Ordering                   |   |
|   |  - Hyper-Sparse Graph Reachability FTRAN (B d = a_q) and BTRAN (B^T y = c_B)             |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                          [6. SELF-HEALING RUNTIME & EXECUTION CONTROLLER]                        |
|   +------------------------------------------------------------------------------------------+   |
|   |  - Dynamic Bound Perturbation (Wolfe-Harris) for Simplex Stalls & Degeneracy Cycling     |   |
|   |  - Iterative Refinement & Precision Escalation for Ill-Conditioned Factorizations        |   |
|   |  - Adaptive Dynamic Cut Filtering (Orthogonality & Coefficient Dynamism)                 |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
|                             [7. POSTSOLVE, CERTIFICATION & ARTIFACTS]                            |
|   +------------------------------------+  +--------------------------------------------------+   |
|   |          Dual Postsolver           |  |        Mathematical Certificate Generator        |   |
|   | (Reconstructs original dimensions) |  |   (||Ax-b|| <= 1e-6, ||A^Ty+z-c|| <= 1e-6, SHA-256)|   |
|   +------------------------------------+  +--------------------------------------------------+   |
|   +------------------------------------------------------------------------------------------+   |
|   |  Output Delivery: solution.json, certificate.md, model.mps, telemetry.log, Web Dashboard |   |
|   +------------------------------------------------------------------------------------------+   |
+==================================================================================================+
```

---

## 3. Detailed Component Technical Specifications

### 3.1 Sovereign Sparse Linear Algebra Kernel

```
                           +--------------------------+
                           |  Raw Triplet (COO) Model |
                           +--------------------------+
                                        |
                                        v
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
         +--------------------+                   +--------------------+
         | Static Column CSC  |                   | Static Row CSR     |
         | (FTRAN Operations) |                   | (BTRAN Operations) |
         +--------------------+                   +--------------------+
                    |                                       |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |  Markowitz Threshold Sparse LU Core   |
                    |            P A_B Q = L U              |
                    +---------------------------------------+
                                        |
                         +--------------+--------------+
                         |                             |
                         v                             v
              +--------------------+        +--------------------+
              | Forrest-Tomlin (FT)|        | Product Form (PFI) |
              | Basis Updates      |        | Eta-Matrix Updates |
              +--------------------+        +--------------------+
```

#### 3.1.1 Sparse Matrix Storage Layout
* **CSC (Compressed Sparse Column)**:
  - `col_ptr: Vec<usize>` of size $n+1$.
  - `row_idx: Vec<usize>` of size $\text{nnz}$.
  - `values: Vec<f64>` of size $\text{nnz}$ (64-byte aligned).
* **CSR (Compressed Sparse Row)**:
  - `row_ptr: Vec<usize>` of size $m+1$.
  - `col_idx: Vec<usize>` of size $\text{nnz}$.
  - `values: Vec<f64>` of size $\text{nnz}$.

#### 3.1.2 Markowitz Threshold Sparse LU Factorization
Factorizes basis $B \in \mathbb{R}^{m \times m}$ into $P B Q = L U$:
1. **Markowitz Count Calculation**: For each non-zero entry $a_{ij}$, compute cost $M_{ij} = (r_i - 1)(c_j - 1)$, where $r_i, c_j$ are active row and column degrees.
2. **Threshold Pivoting Condition**: Candidate $a_{ij}$ must satisfy:
   $$|a_{ij}| \ge u \cdot \max_{k} |a_{kj}|, \quad u = 0.05$$
3. **Sparse Gaussian Elimination**: Updates active submatrix using sparse accumulation vectors (scatter/gather) with zero dynamic allocation.

#### 3.1.3 Basis Update Mechanics
* **Forrest-Tomlin (FT) Update**: Eliminates the leaving variable's column from $U$, creating an upper Hessenberg form. It permutes rows and applies Gauss transformations to restore triangular structure in $O(\text{nnz})$ time.
* **Refactorization Threshold**: Refactorization is triggered when number of updates $k \ge 200$ or when residual $\|B \beta - b\|_\infty > 10^{-7}$.

#### 3.1.4 Matrix Scaling & Equilibration
* **Ruiz Equilibration Algorithm**:
  ```
  Initialize D_R = I_m, D_C = I_n
  For iter = 1 to 20:
      r_i = sqrt(||row_i(A)||_inf) for i in 1..m
      c_j = sqrt(||col_j(A)||_inf) for j in 1..n
      D_R = D_R * diag(1/r_i)
      D_C = D_C * diag(1/c_j)
      A = diag(1/r_i) * A * diag(1/c_j)
      Break if max(|1 - r_i|, |1 - c_j|) < 1e-3
  ```

---

### 3.2 Industrial Presolve Subsystem

```
[Raw Problem Instance]
         |
         v
[Pass 1: Empty / Singleton Filter] ----> Eliminate zero rows/cols & fixed variables
         |
         v
[Pass 2: Constraint Propagation]  ----> Compute L_i, U_i, tighten l_j, u_j bounds
         |
         v
[Pass 3: Probing & Disjunction]   ----> Strengthen big-M coefficients, detect cliques
         |
         v
[Pass 4: Dual Reduction Filter]   ----> Detect dominated rows/columns & free variables
         |
         v
[Presolved Reduced Model] + [Dual Postsolve Stack]
```

1. **Bound Propagation Equations**:
   $$L_i = \sum_{j \in P_i} a_{ij} l_j + \sum_{j \in N_i} a_{ij} u_j, \quad U_i = \sum_{j \in P_i} a_{ij} u_j + \sum_{j \in N_i} a_{ij} l_j$$
   For constraint $b_i^{\text{low}} \le a_i^T x \le b_i^{\text{up}}$:
   - For $a_{ij} > 0$: $x_j \le u_j' = \min\left(u_j, \frac{b_i^{\text{up}} - L_i + a_{ij} l_j}{a_{ij}}\right)$
   - For $a_{ij} < 0$: $x_j \le u_j' = \min\left(u_j, \frac{b_i^{\text{low}} - U_i + a_{ij} l_j}{a_{ij}}\right)$
2. **Dual Postsolver Stack**: Records each reduction in a Last-In-First-Out (LIFO) undo log. During postsolve, recovers original primal vector $x$, dual vector $y$, reduced costs $z$, and slack variables.

---

### 3.3 Continuous LP Engines

#### 3.3.1 Hyper-Sparse Dual Revised Simplex
1. **Pricing (Dual Steepest Edge)**:
   * Edge weights $\gamma_i = \|B^{-1} e_i\|_2^2$ are tracked.
   * Pivot row $p$ is selected as:
     $$p = \arg\max_{i} \left( \frac{\xi_i^2}{\gamma_i} \right)$$
     where $\xi = B^{-1} b$ is the primal basic vector and $\xi_i < l_{B(i)}$ or $\xi_i > u_{B(i)}$.
2. **BTRAN (Compute Pricing Vector)**:
   * Solve $B^T v = e_p$ via hyper-sparse graph-reachability topological sort.
3. **Ratio Test (Harris Two-Pass)**:
   * Pass 1: Determine step length $\theta = \min_{j} \left\{ \frac{d_j}{\alpha_{pj}} \mid \alpha_{pj} \text{ eligible} \right\}$ with relaxed tolerance $\epsilon = 10^{-6}$.
   * Pass 2: Select entering variable $q$ with largest pivot magnitude $|\alpha_{pq}|$ among candidates achieving step $\le \theta$.
4. **FTRAN (Compute Tableau Column)**:
   * Solve $B \alpha = a_q$.
5. **Update**: Update basis $B$, edge weights $\gamma$, and solution vector $\xi$.

#### 3.3.2 Primal-Dual Interior-Point Method (Mehrotra Predictor-Corrector)
Solves continuous LP/QP by iterating the symmetric KKT system:
$$\begin{bmatrix}
-(H + \Theta^{-1}) & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} =
\begin{bmatrix} r_c \\ r_b \end{bmatrix}$$
where $\Theta = X Z^{-1}$.

1. **Affine Predictor Step**: Set $\mu = 0$, solve for $(\Delta x_{\text{aff}}, \Delta y_{\text{aff}}, \Delta z_{\text{aff}})$.
2. **Centering Parameter**:
   $$\sigma = \left( \frac{\mu_{\text{aff}}}{\mu} \right)^3, \quad \mu = \frac{x^T z}{n}$$
3. **Corrector & Gondzio Step**:
   Solve with right-hand side augmented by $\sigma \mu e - \Delta X_{\text{aff}} \Delta Z_{\text{aff}} e + v_{\text{Gondzio}}$.
4. **Step Length & Iterate Update**:
   $$\alpha_P = 0.9995 \cdot \min_{i: \Delta x_i < 0} \left(-\frac{x_i}{\Delta x_i}\right), \quad \alpha_D = 0.9995 \cdot \min_{i: \Delta z_i < 0} \left(-\frac{z_i}{\Delta z_i}\right)$$
   $$x \leftarrow x + \alpha_P \Delta x, \quad y \leftarrow y + \alpha_D \Delta y, \quad z \leftarrow z + \alpha_D \Delta z$$
5. **Vertex Crossover**: Partitions variables into $\{B, N\}$ based on complementarity $\frac{x_j}{z_j}$ and performs clean-up dual simplex pivots to return an exact vertex basis.

---

### 3.4 Mixed-Integer Programming Engine (MILP Core)

```
                       +-------------------------------+
                       | Branch-and-Cut Global Manager |
                       +-------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |  Cutting Plane Engine   |             |  Primal Heuristics      |
      | - Gomory Mixed-Integer  |             | - Feasibility Pump 2.0  |
      | - Knapsack Cover Lift   |             | - RINS / RENS Sub-MIP   |
      | - MIR / Zero-Half       |             | - Conflict Diving       |
      +-------------------------+             +-------------------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                       +-------------------------------+
                       | Lock-Free Work-Stealing Tree  |
                       | (Rayon / Crossbeam Deque)     |
                       +-------------------------------+
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
         +---------------------+               +---------------------+
         | Worker Thread 1     |               | Worker Thread N     |
         | (Dual Simplex Node) |               | (Dual Simplex Node) |
         +---------------------+               +---------------------+
```

#### 3.4.1 Gomory Mixed-Integer (GMI) Cuts
Extracted from optimal tableau row $i$ where basic integer variable $x_{B(i)}$ has fractional value $\bar{b}_i = f_0 + \lfloor \bar{b}_i \rfloor$:
$$\sum_{j \in N, f_j \le f_0} \frac{f_j}{f_0} x_j + \sum_{j \in N, f_j > f_0} \frac{1 - f_j}{1 - f_0} x_j \ge 1$$
* **Cut Filtering**: Cuts with condition number $> 10^4$ or parallelism $|\cos(\theta)| > 0.98$ with existing cuts are discarded.

#### 3.4.2 Hybrid Reliability Branching
* For fractional candidates $j \in I$:
  - If candidate evaluation count $\eta_j < \eta_{\text{rel}} = 8$: Execute **Strong Branching** (solve 50 dual simplex iterations on $x_j \le \lfloor \bar{x}_j \rfloor$ and $x_j \ge \lceil \bar{x}_j \rceil$) and update pseudo-costs:
    $$\Psi_j^- = \frac{\Delta Z^-}{f_0}, \quad \Psi_j^+ = \frac{\Delta Z^+}{1 - f_0}$$
  - If $\eta_j \ge \eta_{\text{rel}}$: Use pseudo-costs directly.
* **Score**: $\text{score}_j = (1-\mu)\min(\Psi_j^- f_0, \Psi_j^+(1-f_0)) + \mu\max(\Psi_j^- f_0, \Psi_j^+(1-f_0))$ with $\mu = 0.16$.

#### 3.4.3 Feasibility Pump 2.0
1. Start with continuous LP relaxation solution $x^{(0)}$.
2. Round fractional variables: $\tilde{x} = \text{round}(x^{(k)})$.
3. If $\tilde{x}$ is LP-feasible $\implies$ integer feasible solution found.
4. Otherwise, solve projection LP:
   $$\min \sum_{j \in I} |x_j - \tilde{x}_j| \quad \text{s.t.} \quad Ax = b, \quad l \le x \le u$$
5. If stall detected: apply random objective perturbation $\Delta c_j \sim \mathcal{U}(-0.1, 0.1)$.

---

### 3.5 Autonomous Synthesis & Code Generation Engine

```
+--------------------------------------------------------------------------------------------------+
|                           AUTONOMOUS CODE GENERATOR ARCHITECTURE                                 |
+--------------------------------------------------------------------------------------------------+
| 1. PROBLEM SPECIFICATION (JSON / Schema / AST):                                                  |
|    - Variables: { name: "flow_crude_1", lower: 0.0, upper: 1000.0, cost: 42.5, integer: false }     |
|    - Constraints: { name: "sulfur_limit", expr: "0.02 * flow_crude_1 + ... <= 15.0" }           |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
| 2. CODE SYNTHESIS ENGINE (Rust / C++ / Python AST Generator):                                    |
|    - Generates self-contained static structs: ModelData, ConstraintMatrix, ObjectiveVector       |
|    - Generates direct FFI call to SutraOpt core runtime                                          |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
| 3. STANDALONE ARTIFACT OUTPUT:                                                                   |
|    - `standalone_solve.rs` / `standalone_solve.cpp` / `solve_problem.py`                         |
|    - Auto-compiles via `rustc -O` / `g++ -O3` into standalone micro-solver binary                |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. API & Interface Specifications

### 4.1 C-ABI Interface Definition (`sutraopt.h`)

```c
#ifndef SUTRAOPT_H
#define SUTRAOPT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SutraModelHandle;

typedef enum {
    SUTRA_STATUS_OPTIMAL = 0,
    SUTRA_STATUS_INFEASIBLE = 1,
    SUTRA_STATUS_UNBOUNDED = 2,
    SUTRA_STATUS_TIME_LIMIT = 3,
    SUTRA_STATUS_NUMERICAL_ERROR = 4
} SutraStatus;

typedef enum {
    SUTRA_VAR_CONTINUOUS = 0,
    SUTRA_VAR_INTEGER = 1,
    SUTRA_VAR_BINARY = 2
} SutraVarType;

/* Model Lifecycle */
SutraModelHandle sutra_create_model(const char* model_name);
void sutra_free_model(SutraModelHandle model);

/* Problem Construction */
int sutra_add_variable(SutraModelHandle model, const char* name, double lb, double ub, double obj, SutraVarType type);
int sutra_add_constraint(SutraModelHandle model, const char* name, size_t nnz, const size_t* indices, const double* values, char sense, double rhs);
int sutra_set_quadratic_objective(SutraModelHandle model, size_t nnz, const size_t* rows, const size_t* cols, const double* values);

/* Autonomous Execution */
SutraStatus sutra_solve_auto(SutraModelHandle model);

/* Solution Extraction */
double sutra_get_objective_value(SutraModelHandle model);
int sutra_get_primal_solution(SutraModelHandle model, double* x_out, size_t max_len);
int sutra_get_dual_solution(SutraModelHandle model, double* y_out, size_t max_len);
int sutra_get_reduced_costs(SutraModelHandle model, double* z_out, size_t max_len);

/* Verification & Certificate Export */
int sutra_generate_certificate_json(SutraModelHandle model, char* buffer_out, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif /* SUTRAOPT_H */
```

### 4.2 Standard Input Schema (`input_problem.json`)

```json
{
  "model_name": "refinery_crude_blending",
  "sense": "minimize",
  "objective": {
    "linear": [
      {"var": "crude_arab_light", "coeff": 45.2},
      {"var": "crude_brent", "coeff": 48.0},
      {"var": "crude_maya", "coeff": 38.5}
    ],
    "quadratic": []
  },
  "variables": [
    {"name": "crude_arab_light", "type": "continuous", "lower": 0.0, "upper": 5000.0},
    {"name": "crude_brent", "type": "continuous", "lower": 0.0, "upper": 4000.0},
    {"name": "crude_maya", "type": "continuous", "lower": 0.0, "upper": 3000.0}
  ],
  "constraints": [
    {
      "name": "total_throughput",
      "sense": ">=",
      "rhs": 8000.0,
      "terms": [
        {"var": "crude_arab_light", "coeff": 1.0},
        {"var": "crude_brent", "coeff": 1.0},
        {"var": "crude_maya", "coeff": 1.0}
      ]
    },
    {
      "name": "max_sulfur_limit",
      "sense": "<=",
      "rhs": 120.0,
      "terms": [
        {"var": "crude_arab_light", "coeff": 0.012},
        {"var": "crude_brent", "coeff": 0.008},
        {"var": "crude_maya", "coeff": 0.035}
      ]
    }
  ],
  "execution_profile": {
    "engine": "auto",
    "presolve": true,
    "scaling": "ruiz",
    "threads": 4,
    "time_limit_sec": 300.0
  }
}
```

---

## 5. Mathematical Verification & Certification Engine

The certifier guarantees complete mathematical auditability:

```
+==================================================================================================+
|                               MATHEMATICAL CERTIFICATE OF OPTIMALITY                             |
+==================================================================================================+
| Model Name: refinery_crude_blending                                                              |
| Solver Engine: SutraOpt Dual Revised Simplex v1.0.0 (SIMD AVX-512)                               |
| Status: OPTIMAL | Iterations: 142 | Solve Time: 1.84 ms                                          |
+--------------------------------------------------------------------------------------------------+
| KKT Parity Verification:                                                                         |
|  * Primal Residual: ||A x* - b||_inf = 4.12e-09 <= 1.0e-06 [PASSED]                              |
|  * Dual Residual:   ||A^T y* + z* - c||_inf = 1.08e-08 <= 1.0e-06 [PASSED]                       |
|  * Duality Gap:     |c^T x* - b^T y*| = 3.94e-09 <= 1.0e-06 [PASSED]                             |
|  * Integrality:     max |x_j - round(x_j)| = 0.00e+00 <= 1.0e-06 [PASSED]                        |
|  * Basis Condition: kappa(B) = 4.82e+03 (Well-Conditioned)                                      |
+--------------------------------------------------------------------------------------------------+
| Cryptographic Integrity Hash:                                                                    |
|  SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855                      |
+==================================================================================================+
```

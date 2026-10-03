# Product Requirements Document (PRD)
## Project: Sovereign Mathematical Optimization & Smart Automation Engine (SutraOpt / BharatOpt)
**Document Version:** 1.0.0  
**Target Release:** Enterprise & Sovereign Industrial Suite  
**Classification:** Strategic Engineering Document  

---

## 1. Executive Summary & Product Vision

### 1.1 Mission
To engineer an indigenous, sovereign, zero-external-dependency, GPU-accelerated mathematical optimization solver and smart automation engine that empowers industrial, defense, energy, and financial infrastructures to automatically formulate, synthesize, solve, verify, and deploy large-scale optimization models with zero reliance on proprietary foreign solver blobs (e.g., Gurobi, CPLEX, Xpress).

### 1.2 Core Value Proposition
SutraOpt combines a **mathematically pure, GPU-accelerated solver core** (Linear, Quadratic, Mixed-Integer Programming) with a **closed-loop Smart Automation Engine**. Users provide high-level declarative domain inputs or industry-standard files, and the engine autonomously generates mathematical formulations, offloads sparse linear algebra to GPU Tensor Cores with mixed-precision iterative refinement, executes continuous and discrete optimization pipelines in milliseconds, verifies optimality using rigorous KKT certificates, and publishes comprehensive solution artifacts.

---

## 2. Target Personas & Stakeholders

| Persona | Role | Primary Pain Point | SutraOpt Solution |
|---|---|---|---|
| **Industrial Process Engineer** | Refinery & Chemical Operations | Complex nonlinear/pooling formulations require manual linearizations and proprietary licenses. | Automated problem synthesis from crude assays, native SOS2/McCormick support, GPU-accelerated execution. |
| **Grid Operations Engineer** | Power Utilities & ISO Dispatchers | Large-scale SCUC ($10^5+$ constraints) stalls on numerical degeneracy in standard solvers. | Hyper-sparse Dual Simplex with Dual Steepest Edge (DSE) and Wolfe-Harris dynamic anti-cycling. |
| **Supply Chain Director** | Logistics & Distribution Networks | High latency and brittle solver integrations across multi-echelon network models. | Automated MILP formulation with Feasibility Pump 2.0 and parallel Branch-and-Cut work-stealing. |
| **Quantitative Portfolio Manager** | Financial Asset Allocation | Inability to embed solvers natively into air-gapped low-latency trading engines due to license constraints. | Zero-dependency C-ABI / Rust native binaries with sub-millisecond Active-Set QP solving. |
| **Sovereign Systems Architect** | Defense & National Infrastructure | Risk of supply chain vulnerabilities and telemetry leaks from proprietary foreign solver binaries. | 100% auditable mathematical first-principles implementation with verifiable SHA-256 certificates. |

---

## 3. Product Goals & Success Criteria

### 3.1 Strategic Goals
1. **100% Sovereign & Independent**: Complete elimination of external solver packages (COIN-OR, SuiteSparse, BLAS/LAPACK proprietary blobs).
2. **Hardware GPU & Tensor-Core Acceleration**: Offload sparse matrix-vector multiplication (SpMV), Ruiz equilibration, and normal equations ($A D^2 A^T$) to CUDA/GPU hardware.
3. **Autonomous End-to-End Operation**: Single-trigger execution transforming raw domain inputs into mathematically certified solutions without manual solver parameter tuning.
4. **Industrial Parity**: Achieve solve time and numerical robustness parity with premier commercial solvers across Netlib, MIPLIB, and QPLIB benchmark suites.
5. **Edge & Cloud Deployment**: Capable of generating standalone, lightweight, self-contained executable solver artifacts for microcontrollers, SCADA systems, and cloud clusters.

### 3.2 Key Performance Indicators (KPIs)
* **Millisecond Decision Latency**: Sub-millisecond to low-millisecond solve time on medium-to-large industrial LP/QP instances ($<1\text{ ms}$ on refinery blending; $<30\text{ ms}$ on mega stress models).
* **Benchmark Solvability & Parity**: Matches or exceeds HiGHS/CPLEX solution quality across 100% of Netlib LP feasible test instances.
* **Mixed-Precision High Accuracy**: Achieves exact double-precision residual bounds $\|Ax - b\|_\infty \le 10^{-12}$ using TensorFloat-32 acceleration with FP64 iterative refinement.
* **Zero-Allocation Simplex**: 0 heap allocations during simplex pivot inner loops.
* **Automation Speed**: Sub-second automated model synthesis and code generation from domain configurations up to $50,000$ variables.

---

## 4. Functional Requirements (FR)

### 4.1 Module 1: Trigger & Multi-Format Ingestion
* **FR-1.1**: The system MUST ingest standard optimization problem files in `.mps` (fixed and free format), `.qps`, `.lp`, and `.nl` formats.
* **FR-1.2**: The system MUST accept structured declarative JSON and YAML schemas defining domain-specific constraints, variables, capacities, and cost functions.
* **FR-1.3**: The system MUST provide an in-memory C-ABI and Python buffer stream API for dynamic programmatic triggering.

### 4.2 Module 2: Autonomous Model Synthesis & Code Generator
* **FR-2.1**: The system MUST automatically transform declarative domain parameters into standard sparse matrices ($A, b, c, l, u, Q$) and integrality vectors.
* **FR-2.2**: The system MUST detect problem topology (sparsity density, matrix condition number $\kappa$, quadratic curvature, integer ratio) and output a structural profile.
* **FR-2.3**: The system MUST automatically generate standalone, reproducible source code files (in Rust, C++, or Python) containing the generated model and dedicated solver execution pipeline.

### 4.3 Module 3: Industrial Presolve & Reductions
* **FR-3.1**: The system MUST perform recursive bound tightening via constraint propagation ($L_i, U_i$).
* **FR-3.2**: The system MUST detect and eliminate singleton rows, singleton columns, and redundant bound implications.
* **FR-3.3**: The system MUST perform coefficient strengthening on mixed-integer disjunctions and big-M formulations.
* **FR-3.4**: The system MUST maintain a dual postsolve mapping to reconstruct exact primal and dual solution vectors, reduced costs, and basis statuses on postsolve.

### 4.4 Module 4: Sovereign Continuous Optimization Engine
* **FR-4.1 (Simplex)**: The system MUST execute a hyper-sparse Dual Revised Simplex algorithm featuring:
  - Dual Steepest Edge (DSE) pricing with recursive weight updates.
  - DEVEX pricing fallback.
  - Harris two-pass ratio test for numerical pivot stability.
  - Graph-reachability hyper-sparse FTRAN and BTRAN.
* **FR-4.2 (Interior-Point)**: The system MUST implement a Primal-Dual Interior-Point Method (IPM) featuring:
  - Mehrotra predictor-corrector step computation.
  - Gondzio multiple centrality corrections.
  - Regularized augmented system factorization.
  - Automatic basis crossover mechanism to recover exact basic vertex solutions.
* **FR-4.3 (Convex QP)**: The system MUST solve convex quadratic objectives via both Active-Set QP and Sparse $LDL^T$ Interior-Point algorithms.

### 4.5 Module 5: Mixed-Integer Programming (MILP Core)
* **FR-5.1**: The system MUST implement a Branch-and-Cut search tree manager supporting lock-free parallel work-stealing concurrency.
* **FR-5.2**: The system MUST generate cutting planes including:
  - Gomory Mixed-Integer (GMI) cuts from simplex tableau rows.
  - Knapsack Minimal Cover cuts.
  - Mixed-Integer Rounding (MIR) and Zero-Half cuts.
* **FR-5.3**: The system MUST implement hybrid reliability branching with directional pseudo-cost updates.
* **FR-5.4**: The system MUST execute primal heuristics including Feasibility Pump 2.0, RINS, and Conflict-Driven Diving.

### 4.6 Module 6: Autonomous Execution Controller & Self-Healing
* **FR-6.1**: The system MUST automatically detect cycling and degeneracy, triggering dynamic Wolfe-Harris bound perturbations ($\delta \in [10^{-9}, 10^{-7}]$).
* **FR-6.2**: The system MUST perform iterative refinement when numerical residual drift exceeds $10^{-6}$.
* **FR-6.3**: The system MUST filter generated cuts based on orthogonality and parallelism to prevent matrix ill-conditioning.

### 4.7 Module 7: Postsolve, Mathematical Certifier & Artifact Generator
* **FR-7.1**: The system MUST verify KKT conditions:
  - Primal Feasibility: $\|Ax - b\|_\infty \le 10^{-6}$
  - Dual Feasibility: $\|A^T y + z - c\|_\infty \le 10^{-6}$
  - Complementary Slackness: $|x^T z| \le 10^{-6}$
  - Integrality Violation: $\max_{j \in I} |x_j - \text{round}(x_j)| \le 10^{-6}$
* **FR-7.2**: The system MUST generate a verifiable **Certificate of Optimality** with SHA-256 cryptographic provenance.
* **FR-7.3**: The system MUST export solution datasets (`solution.json`, `solution.parquet`), execution logs, sensitivity reports, and diagnostic convergence curves.

---

## 5. Non-Functional Requirements (NFR)

### 5.1 Performance & Scalability
* **NFR-1.1**: Simplex inner loop iteration time must not exceed $50\mu s$ on matrices with $m \le 10,000$ and density $\le 0.1\%$.
* **NFR-1.2**: Linear memory scaling: Matrix representations must consume no more than $3\times$ the raw non-zero footprint ($O(\text{nnz})$).
* **NFR-1.3**: Parallel scalability: Branch-and-Cut must demonstrate $\ge 3.2\times$ speedup on 4 CPU cores for tree-dominated benchmark instances.

### 5.2 Reliability & Numerical Safety
* **NFR-2.1**: Zero undefined behavior, memory leaks, or segmentation faults across all valid and invalid inputs.
* **NFR-2.2**: Graceful infeasibility/unboundedness detection with mathematical ray/Farkas certificate extraction.
* **NFR-2.3**: Deterministic execution: Identical random seed and input MUST produce bit-for-bit identical solution vectors.

### 5.3 Portability & Security
* **NFR-3.1**: Zero runtime network access required (100% air-gap capable).
* **NFR-3.2**: Cross-platform support: Windows (x86_64, ARM64), Linux (x86_64, ARM64, RISC-V), macOS (Apple Silicon).
* **NFR-3.3**: Zero external shared library linkage requirements in standalone deployment mode.

---

## 6. Release Phases & Milestones

```
+---------------------------------------------------------------------------------------+
| Milestone 1: Sparse Kernel & Simplex Engine (Months 1-3)                              |
|   -> Pure COO/CSC/CSR, Markowitz Sparse LU, Ruiz Scaling, Dual Simplex Core           |
+---------------------------------------------------------------------------------------+
| Milestone 2: Presolver & High-Speed Pricing (Months 4-6)                              |
|   -> DSE/DEVEX, Hyper-sparse FTRAN/BTRAN, MPS/LP Parsers, Constraint Presolve         |
+---------------------------------------------------------------------------------------+
| Milestone 3: Primal-Dual IPM & Convex QP (Months 7-9)                                 |
|   -> Mehrotra Predictor-Corrector, Gondzio Corrections, Basis Crossover, QP Engine  |
+---------------------------------------------------------------------------------------+
| Milestone 4: Mixed-Integer Branch-and-Cut (Months 10-13)                              |
|   -> GMI Cuts, Knapsack Covers, Reliability Branching, Feasibility Pump 2.0           |
+---------------------------------------------------------------------------------------+
| Milestone 5: Concurrency, C-ABI & Python Ecosystem (Months 14-16)                     |
|   -> Work-Stealing Parallel Tree Search, PyO3/C-ABI, JuMP/Pyomo Connectors            |
+---------------------------------------------------------------------------------------+
| Milestone 6: Smart Automation Engine & Auto-Certifier (Months 17-18)                  |
|   -> Closed-Loop Problem Synthesis, Self-Healing Runtime, Autonomous Artifacts        |
+---------------------------------------------------------------------------------------+
```

# SutraOpt (BharatOpt)

**Indigenous GPU-Accelerated Mathematical Optimization Solver**  
*Sovereign Alternative to CPLEX / Gurobi / Xpress for Strategic Indian Industries*

---

## Overview

SutraOpt is an indigenous high-performance mathematical programming solver engineered for linear programming (LP), mixed-integer linear programming (MILP), and convex quadratic programming (QP). It provides sovereign compute autonomy with native NVIDIA CUDA GPU acceleration, mathematical KKT optimality certification, and specialized domain models for India's energy, refining, and logistics sectors.

---

## Key Capabilities

- **Mathematical Engine**:
  - **LP**: Revised Primal & Dual Simplex with Bland's anti-cycling rule, steepest-edge pricing, and GPU FTRAN/BTRAN triangular solves.
  - **MILP**: Branch-and-Cut with Gomory Mixed-Integer (GMI) cuts, Feasibility Pump 2.0, RINS (Relaxation Induced Neighborhood Search), and Strong Branching.
  - **Convex QP / LP**: Mehrotra Predictor-Corrector Primal-Dual Interior-Point Method (IPM) with mixed-precision Tensor-Core Cholesky factorization and FP64 iterative refinement.
- **Sovereign Trust & Certification**: Cryptographic SHA-256 KKT Optimality Certificates verifying primal/dual residuals and complementary slackness to machine precision.
- **Industrial Readiness**: Specialized domain parsers and schema generators for Petroleum Refinery Crude Blending, National Power Grid SCUC (24h unit commitment), and Pan-India Intermodal Supply Chains.
- **Embedded Deployment**: C-ABI DLL export and standalone Python/C++ code generators for air-gapped DCS/SCADA systems.

---

## Installation

```bash
# Core CPU installation
pip install .

# With GPU Acceleration (NVIDIA CUDA 12.x / 13.x)
pip install .[gpu]
```

---

## CLI Usage

```bash
# Solve an industrial problem with GPU acceleration and cryptographic certification
sutraopt solve datasets/pan_india_petroleum_refinery_stress.json --gpu --certify

# Solve large-scale MILP with time limit and gap tolerance
sutraopt solve datasets/national_discrete_facility_location_milp_stress.json --time-limit 60 --gap 0.01 --certify

# Run MIPLIB 2017 & Netlib standard benchmark suite
sutraopt miplib --time-limit 60 --gap 0.01

# Inspect and profile problem topology
sutraopt info datasets/pan_india_logistics_supply_chain_stress.json

# Export embedded standalone C++ solver code
sutraopt generate datasets/pan_india_petroleum_refinery_stress.json --lang cpp
```

---

## Python API

```python
from sutraopt.engine import SutraOptEngine
from sutraopt.parsers.json_schema import JSONSchemaParser

# Load and parse model
with open("datasets/pan_india_petroleum_refinery_stress.json", "r") as f:
    model = JSONSchemaParser.parse_string(f.read())

engine = SutraOptEngine()

# Solve model (with optional warm-start from prior basis)
result = engine.solve_model(model, use_warm_start=True)

print(f"Status: {result.status}")
print(f"Objective: {result.objective_value:,.2f}")
print(f"KKT Valid: {result.certificate.is_valid}")
print(f"SHA-256 Digest: {result.certificate.sha256_hash}")
```

---

## SIH 2026 Sovereign Architecture

```
                               ┌────────────────────────────────┐
                               │  MPS / QPS / JSON / SCADA C-ABI │
                               └───────────────┬────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │  Sovereign Presolve & Scaling  │
                               │  (Ruiz / Pinf / Bound Implied) │
                               └───────────────┬────────────────┘
                                               ▼
                     ┌─────────────────────────┴─────────────────────────┐
                     ▼                                                   ▼
     ┌───────────────────────────────┐                   ┌───────────────────────────────┐
     │  Branch-and-Cut MILP Engine   │                   │ Mehrotra Interior Point (IPM) │
     │  - Strong Branching           │                   │ - Tensor-Core Cholesky GEMM   │
     │  - GMI Cutting Planes         │                   │ - Mixed-Precision Refinement  │
     │  - Feasibility Pump & RINS    │                   │ - Convex QP / Portfolio Risk  │
     └───────────────┬───────────────┘                   └───────────────┬───────────────┘
                     ▼                                                   ▼
     ┌───────────────────────────────┐                   ┌───────────────────────────────┐
     │  GPU cuSOLVER FTRAN / BTRAN   │                   │ GPU SpMV & Matrix Scaling     │
     └───────────────┬───────────────┘                   └───────────────┬───────────────┘
                     └─────────────────────────┬─────────────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │  SHA-256 KKT Verifier & Cert   │
                               └────────────────────────────────┘
```

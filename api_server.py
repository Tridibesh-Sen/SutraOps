"""
SutraOpt REST API Server
Deploy with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
Docs at:    http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import json
import uuid

from sutraopt.engine import SutraOptEngine
from sutraopt.parsers.json_schema import JSONSchemaParser

app = FastAPI(
    title="SutraOpt (BharatOpt) API",
    description="Indigenous GPU-Accelerated Mathematical Optimization Solver REST API — Sovereign Alternative to CPLEX/Gurobi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single engine instance (shared across all requests — thread-safe for LP/QP)
_engine = SutraOptEngine()


class SolveRequest(BaseModel):
    model: Dict[str, Any]            # Full JSON model payload (same schema as CLI)
    use_gpu: bool = True
    time_limit: Optional[float] = None
    gap_tol: float = 1e-4
    certify: bool = True
    use_warm_start: bool = False


class CertificateData(BaseModel):
    is_valid: bool
    primal_residual: float
    dual_residual: float
    complementary_slackness: float
    sha256: str


class SolveResponse(BaseModel):
    job_id: str
    status: str
    objective_value: Optional[float]
    solve_time_ms: float
    iterations_or_nodes: int
    solution: Dict[str, float]
    certificate: Optional[CertificateData] = None


@app.get("/")
def root():
    """API root — returns metadata."""
    return {
        "name": "SutraOpt (BharatOpt)",
        "tagline": "Indigenous GPU-Accelerated Mathematical Optimization Solver",
        "version": "1.0.0",
        "license": "MIT — ₹0 license cost (100% Sovereign Indian IP)",
        "docs": "/docs",
        "endpoints": ["/health", "/solve", "/info"]
    }


@app.get("/health")
def health():
    """Health check — verify GPU and solver readiness."""
    from sutraopt.linalg.gpu_acceleration import GPULinearAlgebraEngine
    gpu = GPULinearAlgebraEngine()
    info = gpu.get_device_info()
    return {
        "status": "healthy",
        "solver": "SutraOpt 1.0 (BharatOpt)",
        "gpu_device": info["device_name"],
        "gpu_status": info["status"],
        "algorithms": ["LP (Revised Simplex)", "MILP (Branch-and-Cut + GMI)", "QP (Mehrotra IPM)"]
    }


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    """
    Solve an optimization problem.

    **Input:** JSON model with fields:
    - `c`: objective coefficients (list of floats)
    - `A`: constraint matrix (2D list)
    - `row_lower`, `row_upper`: constraint bounds
    - `col_lower`, `col_upper`: variable bounds
    - `is_integer`: list of booleans for integer variables
    - `problem`: one of `generic` / `refinery_blend` / `power_scuc`

    **Output:** Optimal solution + SHA-256 KKT certificate
    """
    job_id = str(uuid.uuid4())[:8]
    try:
        model_json = json.dumps(req.model)
        model = JSONSchemaParser.parse_string(model_json)

        t0 = time.perf_counter()
        result = _engine.solve_model(model, use_warm_start=req.use_warm_start)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        cert_data = None
        if req.certify:
            cert = result.certificate
            cert_data = CertificateData(
                is_valid=bool(cert.is_valid),
                primal_residual=float(cert.primal_residual),
                dual_residual=float(cert.dual_residual),
                complementary_slackness=float(cert.complementary_slackness),
                sha256=cert.sha256_hash
            )

        return SolveResponse(
            job_id=job_id,
            status=result.status,
            objective_value=result.objective_value,
            solve_time_ms=round(elapsed_ms, 3),
            iterations_or_nodes=result.iterations_or_nodes,
            solution=result.solution_vector,
            certificate=cert_data
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Model parse error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {e}")

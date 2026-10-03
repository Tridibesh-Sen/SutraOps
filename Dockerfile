# ==========================================================
# SutraOpt (BharatOpt) Production Container
# ==========================================================
FROM python:3.11-slim

LABEL maintainer="SutraOpt Engineering Team"
LABEL description="Indigenous GPU-Accelerated Mathematical Optimization Solver"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and install sutraopt package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Default entrypoint starts the REST API
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

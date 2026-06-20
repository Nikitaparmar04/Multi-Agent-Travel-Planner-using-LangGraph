# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Trip Planner — Dockerfile
# Base  : python:3.12-slim
# Stages: builder  →  runtime
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only dependency files first (layer caching)
COPY requirements.txt pyproject.toml setup.py ./

# Install dependencies into a prefix directory
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Metadata labels
LABEL maintainer="Nikita Parmar"
LABEL project="Multi-Agent Trip Planner"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Working directory
WORKDIR /app

# Copy application source code
COPY agent/        ./agent/
COPY config/       ./config/
COPY frontend/     ./frontend/
COPY prompt_library/ ./prompt_library/
COPY tools/        ./tools/
COPY utils/        ./utils/
COPY app.py        ./app.py
COPY main.py       ./main.py
COPY setup.py      ./setup.py
COPY pyproject.toml ./pyproject.toml

# Install the local package in editable-equivalent mode
RUN pip install --no-cache-dir -e .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ── Environment Variables (overridden at runtime via --env-file or -e) ─────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ── Expose Ports ──────────────────────────────────────────────────────────────
# FastAPI backend
EXPOSE 8000

# ── Health Check ──────────────────────────────────────────────────────────────
# ✅ Using Python's built-in urllib — no curl/wget needed in runtime image.
# curl was only installed in the builder stage and is NOT copied to runtime.
# Python is always available here, so this approach has zero extra dependencies.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Default Command — FastAPI via Uvicorn ─────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

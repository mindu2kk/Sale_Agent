# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY verification/ ./verification/
COPY agent/ ./agent/
COPY retriever/ ./retriever/
COPY requirements.txt ./

# Create runtime directories and set ownership
RUN mkdir -p logs/errors logs/metrics logs/workflow data && \
    chown -R appuser:appgroup /app

USER appuser

# Expose the API port
EXPOSE 8000

# Health check via the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Async-friendly uvicorn entrypoint
CMD ["uvicorn", "verification.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "asyncio", "--log-level", "info"]

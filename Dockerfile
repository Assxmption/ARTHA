# ─────────────────────────────────────────────────────────────────────────────
# ARTHA — Indian Markets & Commodities Intelligence System
# ─────────────────────────────────────────────────────────────────────────────
# Secrets are NEVER baked into the image: pass them at runtime
# (docker compose `env_file: .env`, or `docker run --env-file .env`).

# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Vite inlines VITE_* vars at build time; these are public (browser) values.
ARG VITE_SUPABASE_URL=""
ARG VITE_SUPABASE_ANON_KEY=""
ENV VITE_SUPABASE_URL=${VITE_SUPABASE_URL} \
    VITE_SUPABASE_ANON_KEY=${VITE_SUPABASE_ANON_KEY}
RUN npm run build

# ── Stage 2: Python API ──────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    AUTH_REQUIRED=true \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Hash-pinned lock (regenerate: uv pip compile requirements.txt --universal
# --python-version 3.12 --generate-hashes -o requirements.lock). --require-hashes
# means a tampered or substituted package fails the build.
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock

COPY app/ app/
COPY docs/backtest_reports/ docs/backtest_reports/
COPY --from=frontend /frontend/dist frontend/dist
# Seed reference data (tracked in git). A named volume mounted on
# /app/data_cache is initialised from these on first start.
COPY data_cache/nse/security_master.json data_cache/nse/security_master.json
COPY data_cache/mcx/contracts.json data_cache/mcx/contracts.json

RUN mkdir -p data/PDFs data/Reports data/Outputs data_cache/models \
    && useradd --create-home --uid 10001 artha \
    && chown -R artha:artha /app
USER artha

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# One worker: analysis-job progress lives in process memory
# (app/api/analysis_routes.py), so with several workers a status/stream
# request can land on a worker that never saw the job.
# Free hosts set $PORT (Render, Railway, Koyeb; Hugging Face Spaces uses 7860).
# --no-server-header: don't advertise uvicorn; client IPs come from
# TRUSTED_PROXY_HOPS (app/security/net.py), not from uvicorn's proxy headers.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-server-header"]

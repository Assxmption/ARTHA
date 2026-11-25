"""
FastAPI Application Entry Point
================================
ARTHA — Indian Markets & Commodities Intelligence System

Security: Defense-in-depth via app.security.apply_security()
  - CSP/HSTS/X-Frame-Options headers
  - Per-IP rate limiting (60 RPM default)
  - Input sanitization (XSS/SQLi rejection)
  - CORS lockdown (allowlist, no wildcards)
  - Request size limits (1 MB max)
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("artha")
# httpx/httpcore log every request URL at INFO; some upstream calls carry
# tokens in the query string, so those URLs must never reach host logs.
for _noisy in ("httpx", "httpcore", "urllib3", "yfinance", "LiteLLM", "litellm"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Refuse to run a production deployment with authentication switched off.
from app.security.auth import check_production_config

_config_problems = check_production_config()
if _config_problems:
    raise RuntimeError("Unsafe production configuration: " + "; ".join(_config_problems))

# ── Ensure data directories exist ─────────────────────────────────────────────
for d in ["data/PDFs", "data/Reports", "data/Outputs", "data_cache"]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ── App Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.llm.router import ModelTier
    db_backend = "Postgres (Supabase)" if os.getenv("DATABASE_URL", "").startswith("postgres") else "SQLite (local)"

    logger.info("=" * 60)
    logger.info("  ARTHA — Indian Markets Intelligence System")
    logger.info("  [DB] %s", db_backend)
    logger.info("  [LLM] Tiered: cheap → mid → deep (see app/llm/router.py)")
    logger.info("  [SEC] Security middleware active (headers, rate-limit, CORS)")
    logger.info("  [API] http://%s:%s", os.getenv("APP_HOST", "0.0.0.0"), os.getenv("APP_PORT", "8000"))
    logger.info("=" * 60)

    # Paper books: keeps the quant engine trading *virtually* (no broker orders).
    paper_enabled = (
        os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
        and "PYTEST_CURRENT_TEST" not in os.environ
    )
    if paper_enabled:
        from app.quant.paper_book import paper_scheduler

        paper_scheduler.start()
        logger.info("  [PAPER] Paper-trading scheduler: %s (simulated orders only)", paper_scheduler.state)
    yield
    if paper_enabled:
        paper_scheduler.stop()
    logger.info("Shutting down ARTHA...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARTHA — Indian Markets Intelligence",
    description=(
        "Production-grade Indian markets (NSE) and commodities (MCX) "
        "intelligence system. Row-level structured fundamentals analysis "
        "with a narrative AI layer, backed by a deterministic quantitative "
        "signal engine. This is not investment advice."
    ),
    version="2.0.0",
    docs_url="/docs" if os.getenv("APP_ENV", "development") == "development" else None,
    redoc_url="/redoc" if os.getenv("APP_ENV", "development") == "development" else None,
    openapi_url="/openapi.json" if os.getenv("APP_ENV", "development") == "development" else None,
    lifespan=lifespan,
)

# ── Security Middleware ────────────────────────────────────────────────────────
# Replaces the old `allow_origins=["*"]` with defense-in-depth.
from app.security import apply_security
apply_security(app)

# ── API Routes ─────────────────────────────────────────────────────────────────
from app.api.quant_routes import quant_router
from app.api.analysis_routes import router as analysis_router
from app.api.simulator_routes import router as simulator_router
from app.api.company_routes import router as company_router
from app.api.screener_routes import router as screener_router
from app.api.kite_routes import router as kite_router
from app.api.system_routes import router as system_router
from app.api.paper_routes import router as paper_router
app.include_router(quant_router)
app.include_router(analysis_router)
app.include_router(simulator_router)
app.include_router(company_router)
app.include_router(screener_router)
app.include_router(kite_router)
app.include_router(system_router)
app.include_router(paper_router)


@app.get("/api/health", tags=["System"])
async def health_check() -> dict:
    """Return a lightweight liveness/readiness response for deployment checks."""
    optional_dependencies = {}
    for package in ("pykalman", "transformers", "torch", "psycopg2"):
        try:
            __import__(package)
            optional_dependencies[package] = True
        except ImportError:
            optional_dependencies[package] = False

    return {
        "status": "ok",
        "service": "artha",
        "factstore_backend": (
            "postgres"
            if os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://"))
            else "sqlite"
        ),
        "frontend_build_present": (Path("frontend/dist/index.html").exists()),
        "optional_dependencies": optional_dependencies,
    }

# ── Static Files (Frontend) ───────────────────────────────────────────────────
frontend_dir = Path("frontend/dist")
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the frontend SPA."""
        return FileResponse(frontend_dir / "index.html")

    _dist_root = frontend_dir.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all for SPA routing: real files inside frontend/dist, else index.html."""
        if full_path.startswith("api/"):
            # Unknown API route: a JSON 404, never the SPA shell.
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        # The path is percent-decoded by Starlette ("..%2F" → "../"), so it must be
        # resolved and contained: anything outside frontend/dist falls back to index.html.
        try:
            requested = (_dist_root / full_path).resolve()
        except (OSError, ValueError):
            requested = None
        if requested is not None and requested.is_relative_to(_dist_root) and requested.is_file():
            return FileResponse(str(requested))
        return FileResponse(str(_dist_root / "index.html"))

@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


# ── Dev Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
        log_level="info",
    )

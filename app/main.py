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
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("artha")

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
    yield
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
from app.api.paper_trade_routes import router as paper_trade_router
app.include_router(quant_router)
app.include_router(analysis_router)
app.include_router(simulator_router)
app.include_router(company_router)
app.include_router(paper_trade_router)

# ── Static Files (Frontend) ───────────────────────────────────────────────────
frontend_dir = Path("frontend/dist")
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the frontend SPA."""
        return FileResponse(frontend_dir / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all for SPA routing."""
        if full_path.startswith("api/"):
            logger.warning(f"API request fell through to SPA catch-all! Path: {full_path}")
            
        index = frontend_dir / "index.html"
        requested = frontend_dir / full_path
        if requested.exists() and requested.is_file():
            return FileResponse(str(requested))
        return FileResponse(str(index))

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

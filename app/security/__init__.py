"""
Security Middleware & Configuration
=====================================
Defense-in-depth security hardening for ARTHA.

Layers:
  1. Security headers (CSP, HSTS, X-Frame-Options, etc.)
  2. Rate limiting (per-IP, configurable)
  3. Input sanitization
  4. CORS lockdown (only allowed origins)
  5. Request size limits
  6. Auth-ready hooks (Supabase JWT validation, Phase 2)

Reference: docs/ARTHA_ARCHITECTURE.md §6 (Security)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ── Security Headers Middleware ─────────────────────────────────────────────────


def _supabase_origin() -> str | None:
    """Supabase project origin from the backend env or the frontend's .env."""
    from urllib.parse import urlparse

    url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    if not url:
        env_file = Path(__file__).resolve().parents[2] / "frontend" / ".env"
        try:
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("VITE_SUPABASE_URL="):
                    url = line.split("=", 1)[1].strip().strip("'\"")
                    break
        except OSError:
            url = None
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return parsed.netloc


def _supabase_connect_sources() -> str:
    host = _supabase_origin()
    if host:
        return f"https://{host} wss://{host}"
    # Unknown project: allow Supabase-hosted projects only, not arbitrary hosts.
    return "https://*.supabase.co wss://*.supabase.co"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add defense-in-depth security headers to every response.

    These headers protect against:
      - XSS (Content-Security-Policy)
      - Clickjacking (X-Frame-Options)
      - MIME sniffing (X-Content-Type-Options)
      - Information leakage (Server, X-Powered-By)
      - Protocol downgrade (Strict-Transport-Security)
      - Referrer leakage (Referrer-Policy)
      - Feature abuse (Permissions-Policy)
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Content Security Policy — strict default.  The SPA authenticates
        # directly against Supabase from the browser, so its origin (HTTPS +
        # realtime WSS) must be allowed in connect-src; with 'self' alone every
        # login request is blocked when the UI is served from this server.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            # 'unsafe-inline' styles: the Kite callback page uses inline style
            # attributes; scripts stay 'self' only, which is what blocks XSS.
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; "
            "img-src 'self' data:; "
            f"connect-src 'self' {_supabase_connect_sources()}; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection (legacy browsers)
        # The legacy XSS auditor is itself exploitable; modern guidance is to disable it (CSP does the job).
        response.headers["X-XSS-Protection"] = "0"

        # HSTS — enforce HTTPS (1 year, include subdomains)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # Referrer policy — don't leak full URL
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # Permissions policy — disable dangerous features
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), interest-cohort=()"
        )

        # Hide server identity
        response.headers["Server"] = "ARTHA"
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        return response


# ── Rate Limiter ────────────────────────────────────────────────────────────────

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Per-IP sliding-window rate limiter.

    Configurable via environment variables:
      RATE_LIMIT_RPM          — requests per minute per IP (default 60 in
                                production, 600 when APP_ENV=development)
      RATE_LIMIT_BURST        — max burst in a 1-second window (10 / 40)

    Returns 429 Too Many Requests when exceeded.
    """

    # A single local user of the live dashboard (chart refetch per toggle,
    # 2-second quote polling, React dev-mode double effects — all from
    # 127.0.0.1 via the Vite proxy) exceeds 60 RPM on its own.  Development
    # therefore gets a higher default; production keeps the strict one.
    DEV_DEFAULTS = (600, 40)
    PROD_DEFAULTS = (60, 10)

    MAX_TRACKED_CLIENTS = 50_000     # hard cap on memory, whatever the traffic
    SWEEP_EVERY_S = 30.0

    def __init__(self, app, rpm: Optional[int] = None, burst: Optional[int] = None):
        super().__init__(app)
        dev = os.getenv("APP_ENV", "development").lower() == "development"
        default_rpm, default_burst = self.DEV_DEFAULTS if dev else self.PROD_DEFAULTS
        self.rpm = int(os.getenv("RATE_LIMIT_RPM", str(rpm or default_rpm)))
        self.burst = int(os.getenv("RATE_LIMIT_BURST", str(burst or default_burst)))
        # Per-client request timestamps (oldest first). Idle clients are swept,
        # so memory is bounded by recent distinct clients, not all-time ones.
        self._windows: dict[str, deque] = {}
        self._last_sweep = 0.0

    def _get_client_ip(self, request: Request) -> str:
        """Client address; X-Forwarded-For only as far as TRUSTED_PROXY_HOPS allows (app/security/net.py)."""
        from app.security.net import client_ip

        return client_ip(request)

    def _sweep(self, now: float) -> None:
        cutoff = now - 60.0
        for ip in [ip for ip, w in self._windows.items() if not w or w[-1] <= cutoff]:
            del self._windows[ip]
        if len(self._windows) > self.MAX_TRACKED_CLIENTS:
            # Under a flood of distinct addresses, keep only the most recently seen half of the cap.
            keep = self.MAX_TRACKED_CLIENTS // 2
            for ip in sorted(self._windows, key=lambda k: self._windows[k][-1])[: len(self._windows) - keep]:
                del self._windows[ip]
        self._last_sweep = now

    def _reject(self, detail: str, retry_after: int) -> Response:
        return Response(
            content=f'{{"detail":"{detail}"}}',
            status_code=429,
            headers={"Content-Type": "application/json", "Retry-After": str(retry_after),
                     "X-RateLimit-Limit": str(self.rpm), "X-RateLimit-Remaining": "0"},
        )

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/api/health"):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.time()
        if now - self._last_sweep > self.SWEEP_EVERY_S or len(self._windows) > self.MAX_TRACKED_CLIENTS:
            self._sweep(now)

        window = self._windows.setdefault(ip, deque())
        while window and window[0] <= now - 60.0:
            window.popleft()
        if len(window) >= self.rpm:
            logger.warning("Rate limit exceeded (%d RPM)", self.rpm)
            return self._reject("Rate limit exceeded. Try again later.", 60)
        recent = 0
        for t in reversed(window):
            if t <= now - 1.0:
                break
            recent += 1
        if recent >= self.burst:
            return self._reject("Burst rate exceeded. Slow down.", 1)

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.rpm - len(window)))
        return response


# ── Request Size Limiter ────────────────────────────────────────────────────────

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject requests with bodies exceeding a configurable size limit.

    Default: 1 MB (more than enough for any ARTHA API request).
    """

    def __init__(self, app, max_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_bytes = int(os.getenv("MAX_REQUEST_BYTES", str(max_bytes)))

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return Response(
                content='{"detail":"Request body too large."}',
                status_code=413,
                headers={"Content-Type": "application/json"},
            )
        return await call_next(request)


# ── Input Sanitization ──────────────────────────────────────────────────────────

# Characters that should never appear in a financial query
_DANGEROUS_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # onclick=, onerror=, etc.
    re.compile(r"data:text/html", re.IGNORECASE),
    re.compile(r"vbscript:", re.IGNORECASE),
    re.compile(r"expression\s*\(", re.IGNORECASE),
    # SQL injection patterns
    re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s", re.IGNORECASE),
    re.compile(r"'\s*(OR|AND)\s+\d+\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"UNION\s+(ALL\s+)?SELECT", re.IGNORECASE),
    re.compile(r"--\s*$", re.MULTILINE),
]

# Max query length (generous but bounded)
_MAX_QUERY_LENGTH = 2000


def sanitize_query(query: str) -> str:
    """
    Sanitize a user query input.

    - Strips leading/trailing whitespace.
    - Rejects queries containing dangerous patterns (XSS, SQLi).
    - Enforces maximum length.
    - Does NOT aggressively strip characters — financial queries may
      contain special chars like &, %, parentheses.

    Raises
    ------
    ValueError
        If the query contains dangerous patterns or is too long.
    """
    query = query.strip()

    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query too long ({len(query)} chars, max {_MAX_QUERY_LENGTH})"
        )

    if len(query) < 3:
        raise ValueError("Query too short (minimum 3 characters)")

    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(query):
            logger.warning("Dangerous pattern detected in query: %s", pattern.pattern)
            raise ValueError("Query contains potentially dangerous content")

    return query


# ── CORS Setup ──────────────────────────────────────────────────────────────────

def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS with a strict allowlist.

    In production, only the specific frontend origin is allowed.
    In development, localhost variants are added.
    """
    env = os.getenv("APP_ENV", "development")

    # Parse allowed origins from environment
    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
    if allowed_origins_str:
        allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]
    elif env == "development":
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    else:
        # Production: no origins allowed unless explicitly configured
        # This forces the deployer to set ALLOWED_ORIGINS
        allowed_origins = []
        logger.warning(
            "ALLOWED_ORIGINS not set in production — CORS will block all cross-origin requests"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Auth is a bearer header, never a cookie, so credentialed CORS is off.
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Groq-Api-Key", "X-Gemini-Api-Key"],
        max_age=600,  # Cache preflight for 10 minutes
    )


# ── Full Security Setup ────────────────────────────────────────────────────────

def apply_security(app: FastAPI) -> None:
    """
    Apply all security layers to a FastAPI app.

    Call this once during app initialization.

    Order matters — middleware is applied in reverse order, so the
    first added is the outermost layer:
      1. Request size limit (reject oversized requests early)
      2. Rate limiter (protect against abuse)
      3. Supabase JWT auth (identity + route protection)
      4. Security headers (defense-in-depth)
      5. CORS (access control)
    """
    from app.auth import SupabaseAuthMiddleware

    # 1. Request size limiter (outermost)
    app.add_middleware(RequestSizeLimitMiddleware)

    # 2. Rate limiter
    app.add_middleware(RateLimiterMiddleware)

    # 3. Supabase JWT authentication
    app.add_middleware(SupabaseAuthMiddleware)

    # 4. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 5. CORS (innermost of our middleware, but FastAPI processes it first)
    configure_cors(app)

    logger.info("Security middleware applied: auth, headers, rate limiting, CORS, size limits")

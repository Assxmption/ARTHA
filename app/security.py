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
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ── Security Headers Middleware ─────────────────────────────────────────────────

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

        # Content Security Policy — strict default
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS — enforce HTTPS (1 year, include subdomains)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Referrer policy — don't leak full URL
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy — disable dangerous features
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=()"
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
      RATE_LIMIT_RPM=60       — requests per minute per IP
      RATE_LIMIT_BURST=10     — max burst in a 1-second window

    Returns 429 Too Many Requests when exceeded.
    """

    def __init__(self, app, rpm: int = 60, burst: int = 10):
        super().__init__(app)
        self.rpm = int(os.getenv("RATE_LIMIT_RPM", str(rpm)))
        self.burst = int(os.getenv("RATE_LIMIT_BURST", str(burst)))
        # Per-IP request timestamps
        self._minute_windows: dict[str, list[float]] = defaultdict(list)
        self._second_windows: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, respecting X-Forwarded-For for reverse proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first (client) IP, not the proxy chain
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/api/health"):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.time()

        # ── Per-minute window ───────────────────────────────────────────
        window = self._minute_windows[ip]
        cutoff = now - 60.0
        # Prune old entries
        self._minute_windows[ip] = [t for t in window if t > cutoff]
        if len(self._minute_windows[ip]) >= self.rpm:
            logger.warning("Rate limit exceeded for IP %s (%d RPM)", ip, self.rpm)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # ── Per-second burst ────────────────────────────────────────────
        burst_window = self._second_windows[ip]
        burst_cutoff = now - 1.0
        self._second_windows[ip] = [t for t in burst_window if t > burst_cutoff]
        if len(self._second_windows[ip]) >= self.burst:
            return Response(
                content='{"detail":"Burst rate exceeded. Slow down."}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": "1",
                },
            )

        # Record this request
        self._minute_windows[ip].append(now)
        self._second_windows[ip].append(now)

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.rpm - len(self._minute_windows[ip]))
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

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
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
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

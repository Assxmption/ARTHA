"""
Supabase JWT Authentication Middleware
=======================================
Validates Supabase-issued JWTs on protected routes.

Behavior:
  - If AUTH_REQUIRED=true (production): rejects unauthenticated requests
    on protected routes with 401.
  - If AUTH_REQUIRED=false (default, dev): allows all requests through
    but still extracts user_id if a valid token is present.

Protected routes (require auth when enabled):
  POST /api/analyze
  POST /api/sim/*
  GET  /api/sim/*
  GET  /api/analyze/*
  GET  /api/analysis-reports

Public routes (never require auth):
  GET  /api/company/*
  GET  /api/news/*
  GET  /api/watchlist
  GET  /api/health
  GET  /api/quant/*
  GET  /docs, /redoc

Reference: docs/ARTHA_ARCHITECTURE.md §4.10 (Supabase), security.py §6
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import jwt
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("artha.auth")

# Routes that NEVER require authentication
PUBLIC_PREFIXES = (
    "/api/company",
    "/api/news",
    "/api/watchlist",
    "/api/health",
    "/api/quant",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/assets",
)

# Routes that require auth when AUTH_REQUIRED=true
PROTECTED_PREFIXES = (
    "/api/analyze",
    "/api/sim",
    "/api/analysis-reports",
)


def _is_protected(path: str) -> bool:
    """Check if a route path requires authentication."""
    for prefix in PROTECTED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    """
    Supabase JWT validation middleware.

    Reads the JWT from the Authorization header (Bearer token),
    validates it using the Supabase JWT secret, and attaches the
    user_id to request.state.

    Configuration (env vars):
      SUPABASE_JWT_SECRET  — The JWT secret from Supabase project settings
      AUTH_REQUIRED         — "true" to enforce auth on protected routes
    """

    def __init__(self, app):
        super().__init__(app)
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "")
        self.auth_required = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
        self.algorithm = "HS256"

        if self.auth_required and not self.jwt_secret:
            logger.warning(
                "AUTH_REQUIRED=true but SUPABASE_JWT_SECRET is not set — "
                "all authenticated requests will fail"
            )

        if self.jwt_secret:
            logger.info("Supabase auth active (required=%s)", self.auth_required)
        else:
            logger.info("Supabase auth passive (no JWT secret configured)")

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract Bearer token from Authorization header."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip()
        return None

    def _validate_token(self, token: str) -> Optional[dict]:
        """Validate a Supabase JWT and return claims."""
        if not self.jwt_secret:
            return None

        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.algorithm],
                audience="authenticated",
                options={
                    "verify_exp": True,
                    "verify_aud": True,
                },
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.debug("JWT expired")
            return None
        except jwt.InvalidAudienceError:
            logger.debug("JWT audience mismatch")
            return None
        except jwt.DecodeError as e:
            logger.debug("JWT decode error: %s", e)
            return None
        except Exception as e:
            logger.warning("JWT validation error: %s", e)
            return None

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for OPTIONS (preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Default: no user
        request.state.user_id = None
        request.state.user_email = None
        request.state.user_role = None

        # Try to extract and validate token regardless of route
        token = self._extract_token(request)
        if token:
            claims = self._validate_token(token)
            if claims:
                request.state.user_id = claims.get("sub")
                request.state.user_email = claims.get("email")
                request.state.user_role = claims.get("role", "authenticated")

        # If auth is required and this is a protected route, enforce it
        if self.auth_required and _is_protected(path):
            if request.state.user_id is None:
                return Response(
                    content='{"detail":"Authentication required. Provide a valid Supabase JWT in the Authorization header."}',
                    status_code=401,
                    headers={
                        "Content-Type": "application/json",
                        "WWW-Authenticate": "Bearer",
                    },
                )

        return await call_next(request)

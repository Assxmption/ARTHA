"""
Request authentication (Supabase JWT) and the operator (admin) check.

* ``get_current_user`` — the verified token claims.  Without a token it
  returns an anonymous identity only when ``AUTH_REQUIRED`` is not "true"
  (local development); otherwise 401.
* ``require_user`` — a real signed-in user, always (even in development a
  missing token yields the dev identity, never a crash).
* ``require_admin`` — the deployment operator: the token's email or user id
  must be listed in ``ADMIN_EMAILS`` / ``ADMIN_USER_IDS``.  Used for anything
  that changes shared state or spends the operator's resources (broker
  session, paper engine, cache rebuilds).  In local development without
  auth, the anonymous developer is the operator.

Tokens are verified with PyJWT (HS256, Supabase's legacy JWT secret): the
signature, expiry and ``aud == "authenticated"`` are all required.  Error
responses never echo the decoder's message.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

security = HTTPBearer(auto_error=False)

_DEV_USER: Dict[str, Any] = {"sub": "anonymous", "role": "development"}


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").lower() == "true"


def is_production() -> bool:
    return os.getenv("APP_ENV", "development").lower() == "production"


def verify_token(token: str) -> Dict[str, Any]:
    secret = os.getenv("SUPABASE_JWT_SECRET", "")
    if not secret:
        raise HTTPException(503, "Authentication is not configured on this server.")
    try:
        claims = jwt.decode(
            token, secret, algorithms=["HS256"], audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
        if claims.get("role") not in (None, "authenticated"):
            raise jwt.InvalidTokenError("not a user session")
        return claims
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired session. Please sign in again.",
                            headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> Dict[str, Any]:
    if credentials and credentials.credentials:
        if not os.getenv("SUPABASE_JWT_SECRET") and not auth_required():
            return dict(_DEV_USER)   # development without a Supabase project
        return verify_token(credentials.credentials)
    if not auth_required():
        return dict(_DEV_USER)
    raise HTTPException(401, "Authentication required.", headers={"WWW-Authenticate": "Bearer"})


_PUBLIC_USER: Dict[str, Any] = {"sub": "public", "role": "anon"}


def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> Dict[str, Any]:
    """Claims if a valid token is sent; otherwise an unprivileged public identity (never 401)."""
    if not auth_required() and not os.getenv("SUPABASE_JWT_SECRET"):
        return dict(_DEV_USER)
    if credentials and credentials.credentials:
        try:
            return verify_token(credentials.credentials)
        except HTTPException:
            pass
    return dict(_DEV_USER) if not auth_required() else dict(_PUBLIC_USER)


def is_dev_identity(user: Dict[str, Any]) -> bool:
    return user.get("sub") == "anonymous" and user.get("role") == "development"


def _csv_env(name: str) -> set[str]:
    return {v.strip().lower() for v in os.getenv(name, "").split(",") if v.strip()}


def is_admin(user: Dict[str, Any]) -> bool:
    if is_dev_identity(user):
        return not auth_required() and not is_production()
    email = str(user.get("email") or "").lower()
    sub = str(user.get("sub") or "").lower()
    return bool((email and email in _csv_env("ADMIN_EMAILS")) or (sub and sub in _csv_env("ADMIN_USER_IDS")))


def require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if is_dev_identity(user) and (auth_required() or is_production()):
        raise HTTPException(401, "Sign in to use this feature.", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not is_admin(user):
        raise HTTPException(403, "Only the operator of this deployment can do this.")
    return user


def check_production_config() -> list[str]:
    """Problems that make a production start unsafe (empty list = OK)."""
    if not is_production():
        return []
    problems = []
    if not auth_required():
        problems.append("AUTH_REQUIRED must be 'true' in production")
    if not os.getenv("SUPABASE_JWT_SECRET"):
        problems.append("SUPABASE_JWT_SECRET must be set in production")
    return problems

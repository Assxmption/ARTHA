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

Tokens are verified with PyJWT.  Supabase projects on the newer signing keys
issue ES256/RS256 tokens: those are checked against the project's public JWKS
(``SUPABASE_URL`` + ``/auth/v1/.well-known/jwks.json``, pinned by config, never
taken from the token).  Legacy HS256 tokens are checked with
``SUPABASE_JWT_SECRET``.  Signature, expiry, ``sub`` and
``aud == "authenticated"`` are always required.  Error responses never echo
the decoder's message.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

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


_ASYMMETRIC = ("ES256", "RS256")
_jwks_lock = threading.Lock()
_jwks_clients: dict[str, "jwt.PyJWKClient"] = {}


def _jwks_url() -> Optional[str]:
    base = (os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL") or "").strip().rstrip("/")
    if not base.startswith("https://"):
        return None
    return f"{base}/auth/v1/.well-known/jwks.json"


def _jwks_client(url: str) -> "jwt.PyJWKClient":
    with _jwks_lock:
        client = _jwks_clients.get(url)
        if client is None:
            # Keys are cached (and refetched on an unknown kid), so most requests
            # never touch the network.
            client = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600, timeout=5)
            _jwks_clients[url] = client
        return client


def auth_configured() -> bool:
    return bool(os.getenv("SUPABASE_JWT_SECRET") or _jwks_url())


def decode_supabase_jwt(token: str) -> Dict[str, Any]:
    """Verified claims of a Supabase user session; raises ``jwt.PyJWTError`` otherwise."""
    alg = jwt.get_unverified_header(token).get("alg")
    options = {"require": ["exp", "sub"]}
    if alg in _ASYMMETRIC:
        url = _jwks_url()
        if not url:
            raise jwt.InvalidTokenError("asymmetric token but SUPABASE_URL is not configured")
        key = _jwks_client(url).get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=list(_ASYMMETRIC), audience="authenticated", options=options)
    elif alg == "HS256":
        secret = os.getenv("SUPABASE_JWT_SECRET", "")
        if not secret:
            raise jwt.InvalidTokenError("HS256 token but SUPABASE_JWT_SECRET is not configured")
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated", options=options)
    else:
        raise jwt.InvalidAlgorithmError(f"unsupported alg {alg!r}")
    if claims.get("role") not in (None, "authenticated"):
        raise jwt.InvalidTokenError("not a user session")
    return claims


def verify_token(token: str) -> Dict[str, Any]:
    if not auth_configured():
        raise HTTPException(503, "Authentication is not configured on this server.")
    try:
        return decode_supabase_jwt(token)
    except (jwt.PyJWTError, ValueError):
        raise HTTPException(401, "Invalid or expired session. Please sign in again.",
                            headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> Dict[str, Any]:
    if credentials and credentials.credentials:
        if not auth_configured() and not auth_required():
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
    if not auth_required() and not auth_configured():
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
    if not auth_configured():
        problems.append("SUPABASE_URL (for JWKS) or SUPABASE_JWT_SECRET must be set in production")
    return problems

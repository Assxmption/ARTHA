"""
System status for the Settings page: which data feeds and model providers
are usable right now, how fresh the live engine is, and the fixed validation
gate. No secrets: only booleans, timestamps and cooldowns.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.llm.keys import KeyFormatError, keys_from_headers, policy_status
from app.security.auth import is_admin, optional_user
from app.security.net import client_ip

router = APIRouter(prefix="/api/system", tags=["System"])


def _status() -> dict:
    from app.data import kite
    from app.llm.router import get_key_status
    from app.quant import backtest
    from app.quant.live_dashboard import ENGINE_VERSION, live_dashboard_cache

    session = kite.load_session()
    kite_state = {
        "configured": kite.is_configured(),
        "has_current_session": bool(session and session.is_current()),
    }

    cached = live_dashboard_cache._load()
    engine = {
        "version": ENGINE_VERSION,
        "as_of": cached.get("as_of") if cached else None,
        "computed_at": cached.get("computed_at") if cached else None,
        "last_error": live_dashboard_cache._last_error,
    }

    return {
        "environment": os.getenv("APP_ENV", "development"),
        "auth_required": os.getenv("AUTH_REQUIRED", "false").lower() == "true",
        "factstore_backend": (
            "postgres"
            if os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://"))
            else "sqlite"
        ),
        "kite": kite_state,
        "llm": get_key_status()["providers"],
        "live_engine": engine,
        # Read-only by design (AGENTS.md rule 6): shown, never editable.
        "validation_gate": {
            "min_oos_sharpe": backtest.MIN_OOS_SHARPE,
            "max_oos_drawdown": backtest.MAX_OOS_DRAWDOWN,
            "min_oos_sessions": backtest.MIN_OOS_TRADING_DAYS,
        },
        "order_placement": False,
    }


@router.get("/status")
async def system_status(request: Request, user=Depends(optional_user)) -> dict:
    out = await run_in_threadpool(_status)
    out["llm_policy"] = policy_status(_visitor(request, user))
    if not is_admin(user):   # internal error text is for the operator only
        out["live_engine"]["last_error"] = "refresh failed" if out["live_engine"]["last_error"] else None
    return out


def _visitor(request: Request, user: dict | None) -> str:
    sub = (user or {}).get("sub")
    return sub if sub and sub not in ("anonymous", "public") else f"ip:{client_ip(request)}"


@router.get("/llm-policy")
async def llm_policy(request: Request, user=Depends(optional_user)) -> dict:
    """Who pays for AI reports here: visitor keys, the owner's demo quota, or both."""
    return policy_status(_visitor(request, user))


# Free, token-less endpoints that only list models — enough to prove a key works.
_KEY_PROBES = {
    "groq": ("https://api.groq.com/openai/v1/models", lambda k: {"Authorization": f"Bearer {k}"}),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", lambda k: {"x-goog-api-key": k}),
}


@router.post("/llm-key-check")
async def llm_key_check(request: Request, user=Depends(optional_user)) -> dict:
    """
    Check the visitor's own keys (sent in the X-Groq-Api-Key / X-Gemini-Api-Key
    headers) against each provider's model list.  Keys are not stored or logged.
    """
    try:
        keys = keys_from_headers(request.headers)
    except KeyFormatError as e:
        raise HTTPException(400, {"code": "LLM_KEY_INVALID", "message": str(e)})
    if not keys.any:
        raise HTTPException(400, {"code": "LLM_KEY_MISSING", "message": "No key supplied"})
    results = {}
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
        for provider in keys.providers:
            url, headers = _KEY_PROBES[provider]
            try:
                r = await client.get(url, headers=headers(keys.get(provider)))
                results[provider] = {
                    "ok": r.status_code == 200,
                    "status": r.status_code,
                    "message": {200: "Key works", 401: "Key rejected", 403: "Key rejected or API not enabled",
                                429: "Key works but is rate limited right now"}.get(r.status_code, "Unexpected response"),
                }
            except httpx.HTTPError:
                results[provider] = {"ok": False, "status": None, "message": "Provider unreachable"}
    return {"results": results}

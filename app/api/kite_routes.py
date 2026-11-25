"""
Kite Connect login/session routes (read-only integration).

  GET  /api/kite/status     → configured / connected / user (no secrets)
  GET  /api/kite/login      → 302 to Kite's login page (one-time CSRF state)
  POST /api/kite/login-url  → same URL as JSON (for authenticated XHR flows)
  GET  /api/kite/callback   → Kite redirect target; exchanges request_token
  POST /api/kite/logout     → invalidates the token at Kite and locally

The callback URL registered on the Kite app must equal KITE_REDIRECT_URL.
"""

from __future__ import annotations

import html
import logging
import time
from datetime import datetime, time as dtime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.data import kite
from app.security.auth import is_admin, optional_user, require_admin

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kite", tags=["Kite Connect"])

_status_cache: dict = {"at": 0.0, "body": None}
_STATUS_TTL_S = 60


def _session_window() -> dict:
    """When the stored token was issued and when Kite rolls it over (06:00 IST)."""
    session = kite.load_session()
    if session is None:
        return {}
    created = datetime.fromisoformat(session.created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=kite.IST)
    expires = datetime.combine(kite._session_trading_day(created) + timedelta(days=1), dtime(6, 0), tzinfo=kite.IST)
    return {"session_created_at": created.isoformat(timespec="seconds"),
            "session_expires_at": expires.isoformat(timespec="seconds")}


_ADMIN_ONLY_FIELDS = ("user_id", "user_name", "exchanges", "session_created_at", "session_expires_at", "reason")


@router.get("/status")
async def kite_status(user=Depends(optional_user)):
    body = await _kite_status_body()
    # The broker account behind this deployment is the operator's business:
    # visitors only learn whether live data is on.
    if is_admin(user):
        return body
    return {k: v for k, v in body.items() if k not in _ADMIN_ONLY_FIELDS}


async def _kite_status_body() -> dict:
    if not kite.is_configured():
        return {"configured": False, "connected": False}
    now = time.monotonic()
    if _status_cache["body"] and now - _status_cache["at"] < _STATUS_TTL_S:
        return _status_cache["body"]
    try:
        client = await run_in_threadpool(kite.get_client)
        profile = await run_in_threadpool(client.profile)
        body = {"configured": True, "connected": True, "user_id": profile.get("user_id"),
                "user_name": profile.get("user_name"), "exchanges": profile.get("exchanges"),
                **_session_window()}
    except kite.KiteNotConnected as e:
        body = {"configured": True, "connected": False, "reason": str(e)}
    except kite.KiteError as e:
        body = {"configured": True, "connected": False, "reason": str(e)}
    _status_cache.update(at=now, body=body)
    return body


@router.get("/login")
async def kite_login(user=Depends(require_admin)):
    try:
        url, _ = kite.login_url()
    except kite.KiteError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return RedirectResponse(url, status_code=302)


@router.post("/login-url")
async def kite_login_url(user=Depends(require_admin)):
    try:
        url, _ = kite.login_url()
    except kite.KiteError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"login_url": url}


def _page(title: str, message: str, ok: bool) -> HTMLResponse:
    colour = "#2e7d5b" if ok else "#b3261e"
    body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>ARTHA · Kite</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'></head>"
        "<body style='font-family:system-ui,sans-serif;background:#14120f;color:#ebe1d3;"
        "display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0'>"
        f"<div style='max-width:420px;padding:24px;border:1px solid {colour}'>"
        f"<h2 style='margin-top:0;color:{colour}'>{html.escape(title)}</h2>"
        f"<p>{html.escape(message)}</p><p>You can close this tab and return to ARTHA.</p>"
        "</div></body></html>"
    )
    return HTMLResponse(body, status_code=200 if ok else 400)


@router.get("/callback", response_class=HTMLResponse)
async def kite_callback(
    request_token: str | None = Query(default=None, max_length=128),
    status: str | None = Query(default=None, max_length=32),
    state: str | None = Query(default=None, max_length=128),
):
    if status != "success" or not request_token:
        return _page("Kite login not completed", f"Kite returned status: {status or 'missing'}", False)
    if not kite.consume_state(state):
        # Not started from this server, replayed, or older than 10 minutes.
        return _page("Login link expired", "Start the Kite login again from ARTHA.", False)
    try:
        session = await run_in_threadpool(kite.exchange_request_token, request_token)
    except kite.KiteError as e:
        logger.warning("Kite token exchange failed: %s", e)
        return _page("Kite login failed", str(e), False)
    _status_cache.update(at=0.0, body=None)
    logger.info("Kite session established for user %s", session.user_id)
    return _page("Connected to Kite", f"Signed in as {session.user_id}. Live market data is now enabled.", True)


@router.post("/logout")
async def kite_logout(user=Depends(require_admin)):
    session = kite.load_session()
    if session is not None:
        try:
            import httpx

            await run_in_threadpool(
                # The token goes in the request, never logged: httpx's INFO request
                # log is silenced in app/main.py, and errors below never echo it.
                lambda: httpx.delete(
                    f"{kite.API_ROOT}/session/token",
                    headers={"X-Kite-Version": "3"},
                    params={"api_key": kite._api_key(), "access_token": session.access_token},
                    timeout=10,
                )
            )
        except Exception as e:  # local logout still proceeds
            logger.warning("Kite logout call failed: %s", type(e).__name__)
    kite.clear_session()
    _status_cache.update(at=0.0, body=None)
    return {"connected": False}

"""
Zerodha Kite Connect — read-only market data
=============================================
Login (request_token → access_token), quotes, instruments and historical
candles over the Kite Connect v3 REST API.

Scope is deliberately READ-ONLY.  This module contains no order, GTT,
margin-pledge or fund-transfer endpoints, so nothing in ARTHA can place an
order through it (AGENTS.md rule 9).  Adding order placement requires a human
confirmation step and the SEBI retail-algo requirements (static IP, Algo-ID)
— see docs/ARTHA_ARCHITECTURE.md §6.6.

Secrets:
  * KITE_API_KEY / KITE_API_SECRET come from the environment (.env).
  * The daily access token is stored in data_cache/kite/session.json
    (gitignored, mode 0600), never logged or returned by the API.
  * Kite invalidates access tokens daily (around 06:00 IST), so a stored
    session is only trusted for the IST trading day it was created on.

Rate limits respected (Kite docs): quote ≤ 1 req/s, historical ≤ 3 req/s.
Historical requests are chunked to Kite's per-interval range limits.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_ROOT = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"
IST = ZoneInfo("Asia/Kolkata")
SESSION_FILE = Path("data_cache/kite/session.json")
INSTRUMENTS_DIR = Path("data_cache/kite/instruments")
HISTORICAL_DIR = Path("data_cache/kite/historical")

# Max calendar days per historical request, per Kite docs.
_MAX_DAYS_PER_REQUEST = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}
INTERVALS = tuple(_MAX_DAYS_PER_REQUEST)

# Index tradingsymbols on the NSE segment.
INDEX_SYMBOLS = {"NIFTY 50", "NIFTY BANK", "INDIA VIX", "NIFTY FIN SERVICE", "NIFTY IT"}


class KiteError(RuntimeError):
    """Kite returned an error or is unreachable."""


class KiteNotConnected(KiteError):
    """No valid access token: the user must log in via the Kite redirect."""


# ── Rate limiting ───────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_quote_limiter = _RateLimiter(1.0)
_historical_limiter = _RateLimiter(3.0)
_default_limiter = _RateLimiter(8.0)


# ── Session storage ─────────────────────────────────────────────────────────────

def _api_key() -> str:
    return os.getenv("KITE_API_KEY", "").strip()


def _api_secret() -> str:
    return os.getenv("KITE_API_SECRET", "").strip()


def is_configured() -> bool:
    return bool(_api_key() and _api_secret())


def _session_trading_day(now: Optional[datetime] = None) -> date:
    """Kite tokens roll over around 06:00 IST; map a timestamp to its token day."""
    now = now or datetime.now(IST)
    return (now - timedelta(hours=6)).date()


@dataclass
class KiteSession:
    access_token: str
    user_id: str
    created_at: str  # ISO, IST

    def is_current(self, now: Optional[datetime] = None) -> bool:
        created = datetime.fromisoformat(self.created_at)
        return _session_trading_day(created) == _session_trading_day(now)


def load_session(path: Optional[Path] = None) -> Optional[KiteSession]:
    path = path or SESSION_FILE
    env_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            session = KiteSession(**raw)
            if session.is_current():
                return session
            logger.info("Stored Kite session is from a previous trading day; login required")
        except Exception as e:
            logger.warning("Unreadable Kite session file: %s", e)
    if env_token:
        # Manually supplied token (e.g. copied from another tool): trust it for
        # today; Kite will reject it with a TokenException if it has expired.
        return KiteSession(env_token, os.getenv("KITE_USER_ID", ""), datetime.now(IST).isoformat())
    return None


def save_session(session: KiteSession, path: Optional[Path] = None) -> None:
    path = path or SESSION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(session.__dict__))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def clear_session(path: Optional[Path] = None) -> None:
    path = path or SESSION_FILE
    if path.exists():
        path.unlink()


# ── Login flow ──────────────────────────────────────────────────────────────────

_STATE_TTL_S = 900
_state_lock = threading.Lock()


def _states_file() -> Path:
    # Persisted (not in-memory) so a server reload between "open login page"
    # and "Kite redirects back" doesn't invalidate a legitimate login.
    return SESSION_FILE.parent / "pending_states.json"


def _load_states() -> dict[str, float]:
    try:
        return json.loads(_states_file().read_text())
    except Exception:
        return {}


def _save_states(states: dict[str, float]) -> None:
    path = _states_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(states))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def login_url() -> tuple[str, str]:
    """Kite login URL plus the one-time `state` that the callback must echo."""
    if not is_configured():
        raise KiteError("KITE_API_KEY / KITE_API_SECRET are not set")
    state = secrets.token_urlsafe(24)
    now = time.time()
    with _state_lock:
        states = {s: t for s, t in _load_states().items() if now - t <= _STATE_TTL_S}
        states[state] = now
        _save_states(states)
    params = {"v": "3", "api_key": _api_key(), "redirect_params": urlencode({"state": state})}
    return f"{LOGIN_URL}?{urlencode(params)}", state


def consume_state(state: Optional[str]) -> bool:
    """Single-use CSRF check for the redirect callback."""
    if not state:
        return False
    with _state_lock:
        states = _load_states()
        issued = states.pop(state, None)
        _save_states(states)
    return issued is not None and time.time() - issued <= _STATE_TTL_S


def checksum(api_key: str, request_token: str, api_secret: str) -> str:
    return hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()


def exchange_request_token(request_token: str, http: Optional[httpx.Client] = None) -> KiteSession:
    """Swap the one-time request_token for the day's access token and store it."""
    if not is_configured():
        raise KiteError("KITE_API_KEY / KITE_API_SECRET are not set")
    client = http or httpx.Client(timeout=15)
    resp = client.post(
        f"{API_ROOT}/session/token",
        headers={"X-Kite-Version": "3"},
        data={
            "api_key": _api_key(),
            "request_token": request_token,
            "checksum": checksum(_api_key(), request_token, _api_secret()),
        },
    )
    body = _json_or_error(resp)
    data = body.get("data") or {}
    if not data.get("access_token"):
        raise KiteError("Kite session response had no access_token")
    session = KiteSession(
        access_token=data["access_token"],
        user_id=str(data.get("user_id", "")),
        created_at=datetime.now(IST).isoformat(),
    )
    save_session(session)
    return session


def _json_or_error(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        raise KiteError(f"Kite HTTP {resp.status_code}: non-JSON response")
    if resp.status_code >= 400 or body.get("status") == "error":
        etype = body.get("error_type", "")
        message = body.get("message", f"HTTP {resp.status_code}")
        if etype == "TokenException" or resp.status_code == 403:
            clear_session()
            raise KiteNotConnected(f"Kite session invalid or expired: {message}")
        raise KiteError(f"Kite {etype or 'error'}: {message}")
    return body


# ── Read-only client ───────────────────────────────────────────────────────────

class KiteClient:
    """Thin read-only client. Construct via `get_client()`."""

    def __init__(self, session: KiteSession, http: Optional[httpx.Client] = None):
        self.session = session
        self.http = http or httpx.Client(timeout=20)

    @property
    def _headers(self) -> dict:
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {_api_key()}:{self.session.access_token}",
        }

    def _get(self, path: str, params=None, limiter: _RateLimiter = _default_limiter) -> dict:
        limiter.wait()
        try:
            resp = self.http.get(f"{API_ROOT}{path}", headers=self._headers, params=params)
        except httpx.HTTPError as e:
            raise KiteError(f"Kite unreachable: {e}") from e
        return _json_or_error(resp)

    def profile(self) -> dict:
        data = self._get("/user/profile")["data"]
        # Only non-sensitive identity fields leave this module.
        return {k: data.get(k) for k in ("user_id", "user_name", "broker", "exchanges", "products")}

    # Quotes ---------------------------------------------------------------
    def quote(self, instruments: Iterable[str]) -> dict:
        """Full quotes (OHLC, depth, OI, circuit limits); ≤ 500 per request."""
        keys = list(instruments)
        out: dict = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            out.update(self._get("/quote", params=[("i", k) for k in chunk], limiter=_quote_limiter)["data"])
        return out

    def ltp(self, instruments: Iterable[str]) -> dict[str, float]:
        keys = list(instruments)
        data = self._get("/quote/ltp", params=[("i", k) for k in keys], limiter=_quote_limiter)["data"]
        return {k: float(v["last_price"]) for k, v in data.items()}

    # Instruments ------------------------------------------------------------
    def instruments(self, exchange: str) -> pd.DataFrame:
        """Instrument master for an exchange, cached once per trading day."""
        INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cache = INSTRUMENTS_DIR / f"{exchange}_{_session_trading_day()}.csv"
        if cache.exists():
            return pd.read_csv(cache, dtype={"instrument_token": int})
        _default_limiter.wait()
        resp = self.http.get(f"{API_ROOT}/instruments/{exchange}", headers=self._headers)
        if resp.status_code != 200:
            _json_or_error(resp)
        frame = pd.read_csv(io.StringIO(resp.text), dtype={"instrument_token": int})
        frame.to_csv(cache, index=False)
        for old in INSTRUMENTS_DIR.glob(f"{exchange}_*.csv"):
            if old != cache:
                old.unlink(missing_ok=True)
        return frame

    def instrument_token(self, tradingsymbol: str, exchange: str = "NSE") -> int:
        frame = self.instruments(exchange)
        match = frame[frame["tradingsymbol"] == tradingsymbol]
        if match.empty:
            raise KiteError(f"Unknown Kite instrument {exchange}:{tradingsymbol}")
        return int(match["instrument_token"].iloc[0])

    # Historical -------------------------------------------------------------
    def historical(
        self,
        instrument_token: int,
        interval: str,
        start: datetime,
        end: datetime,
        oi: bool = False,
    ) -> pd.DataFrame:
        """OHLCV candles; chunked to Kite's per-interval range limits."""
        if interval not in _MAX_DAYS_PER_REQUEST:
            raise ValueError(f"Unsupported Kite interval {interval!r}")
        step = timedelta(days=_MAX_DAYS_PER_REQUEST[interval])
        rows: list[list] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + step - timedelta(seconds=1), end)
            body = self._get(
                f"/instruments/historical/{instrument_token}/{interval}",
                params={
                    "from": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                    "to": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "oi": int(oi),
                },
                limiter=_historical_limiter,
            )
            rows.extend(body["data"]["candles"])
            cursor = chunk_end + timedelta(seconds=1)
        cols = ["date", "open", "high", "low", "close", "volume"] + (["oi"] if oi else [])
        if not rows:
            return pd.DataFrame(columns=cols)
        frame = pd.DataFrame([r[: len(cols)] for r in rows], columns=cols)
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_convert(IST).dt.tz_localize(None)
        return frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


_client_lock = threading.Lock()
_client: Optional[KiteClient] = None


def get_client() -> KiteClient:
    """Client bound to today's session; raises KiteNotConnected if none."""
    global _client
    if not is_configured():
        raise KiteNotConnected("Kite API key/secret not configured")
    with _client_lock:
        session = load_session()
        if session is None:
            _client = None
            raise KiteNotConnected("Not logged in to Kite today")
        if _client is None or _client.session.access_token != session.access_token:
            _client = KiteClient(session)
        return _client


def is_connected() -> bool:
    try:
        get_client()
        return True
    except KiteNotConnected:
        return False


def kite_symbol(symbol: str) -> tuple[str, str]:
    """Map an ARTHA symbol to (exchange, tradingsymbol)."""
    from app.data.symbols import resolve_symbol

    s = resolve_symbol(symbol)  # former tickers (ZOMATO → ETERNAL) map to the current listing
    aliases = {"NIFTY": "NIFTY 50", "NIFTY50": "NIFTY 50", "^NSEI": "NIFTY 50",
               "BANKNIFTY": "NIFTY BANK", "^NSEBANK": "NIFTY BANK",
               "INDIAVIX": "INDIA VIX", "^INDIAVIX": "INDIA VIX"}
    return "NSE", aliases.get(s, s)

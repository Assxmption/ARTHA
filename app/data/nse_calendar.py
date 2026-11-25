"""
NSE equity-segment trading calendar.

Holidays for 2026 come from NSE's 2026 trading-holiday circular (FAOP71777).
They were transcribed from Upstox's copy of that list, which says it was
verified against the circular; the circular itself could not be downloaded
when this file was written.  Re-check against nseindia.com when it is updated.

For years not listed in ``KNOWN_YEARS`` the calendar falls back to
"every weekday is a session" — callers that fill orders must still require a
price observed today (app/quant/paper_book.is_fresh), so an unknown holiday
can never produce a fill against a stale price.

Muhurat trading (2026-11-08, a Sunday) is not included: NSE announces its
timings by a separate circular.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
REGULAR_OPEN, REGULAR_CLOSE = time(9, 15), time(15, 30)

KNOWN_YEARS = frozenset({2026})

# Weekday closures (equity, equity derivatives, SLB).
NSE_HOLIDAYS: dict[date, str] = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 3): "Holi",
    date(2026, 3, 26): "Shri Ram Navami",
    date(2026, 3, 31): "Shri Mahavir Jayanti",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 28): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali - Balipratipada",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
    date(2026, 12, 25): "Christmas",
}

# Weekend sessions NSE has announced: date -> (open, close, reason).
SPECIAL_SESSIONS: dict[date, tuple[time, time, str]] = {
    date(2026, 2, 1): (time(9, 15), time(15, 0), "Union Budget (Sunday live session)"),
}


def holiday_name(d: date) -> Optional[str]:
    """Why ``d`` has no session, or None if it is a trading day."""
    if d in SPECIAL_SESSIONS:
        return None
    if d in NSE_HOLIDAYS:
        return NSE_HOLIDAYS[d]
    if d.weekday() >= 5:
        return "Weekend"
    return None


def is_trading_day(d: date) -> bool:
    return holiday_name(d) is None


def session_hours(d: date) -> Optional[tuple[time, time]]:
    if d in SPECIAL_SESSIONS:
        o, c, _ = SPECIAL_SESSIONS[d]
        return o, c
    return (REGULAR_OPEN, REGULAR_CLOSE) if is_trading_day(d) else None


def _ist(now: Optional[datetime]) -> datetime:
    now = now or datetime.now(IST)
    return now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)


def is_market_open(now: Optional[datetime] = None) -> bool:
    now = _ist(now)
    hours = session_hours(now.date())
    return bool(hours) and hours[0] <= now.time() <= hours[1]


def market_status(now: Optional[datetime] = None) -> dict:
    """{"open": bool, "reason": str} for UI and API payloads."""
    now = _ist(now)
    d = now.date()
    name = holiday_name(d)
    if name:
        return {"open": False, "reason": f"NSE closed: {name}"}
    o, c = session_hours(d)
    if now.time() < o:
        return {"open": False, "reason": f"Pre-open (session starts {o:%H:%M} IST)"}
    if now.time() > c:
        return {"open": False, "reason": "Session closed for the day"}
    return {"open": True, "reason": "Session open"}


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_close_boundary(now: Optional[datetime] = None) -> datetime:
    """Most recent session close (IST) that has already passed."""
    now = _ist(now)
    d = now.date()
    hours = session_hours(d)
    if hours and now.time() >= hours[1]:
        return datetime.combine(d, hours[1], IST)
    d = previous_trading_day(d)
    return datetime.combine(d, session_hours(d)[1], IST)

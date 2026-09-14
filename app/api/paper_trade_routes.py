"""
Paper Trading API Routes
========================
Exposes the PaperTradingDaemon to the frontend via REST endpoints.

All operations are virtual — no broker API calls.
AGENTS.md Rule 9: No autonomous order placement.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.quant.paper_trade import PaperTradingDaemon, DaemonState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper-trade", tags=["paper-trading"])

# ── Module-level daemon instance ────────────────────────────────────────────────
# Initialized lazily on first start.  Only one daemon can run at a time.
_daemon: Optional[PaperTradingDaemon] = None

# Default symbols — top NIFTY 50 constituents by weight
DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "BHARTIARTL", "SBIN", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO",
]


# ── Request/Response Models ────────────────────────────────────────────────────

class StartRequest(BaseModel):
    symbols: list[str] | None = None
    initial_capital: float = 5_00_00_000.0  # ₹5 Cr
    rebalance_interval_min: int = 15


class ResetRequest(BaseModel):
    new_capital: float | None = None
    confirm: bool = False  # Must be True to reset


# ── Routes ──────────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_daemon(request: StartRequest):
    """Start the paper trading daemon."""
    global _daemon

    if _daemon and _daemon._state == DaemonState.RUNNING:
        return {
            "status": "already_running",
            "message": "Daemon is already running",
        }

    symbols = request.symbols or DEFAULT_SYMBOLS

    _daemon = PaperTradingDaemon(
        symbols=symbols,
        initial_capital=request.initial_capital,
        rebalance_interval_min=request.rebalance_interval_min,
    )
    _daemon.start()

    return {
        "status": "started",
        "symbols": symbols,
        "initial_capital": request.initial_capital,
        "rebalance_interval_min": request.rebalance_interval_min,
    }


@router.post("/stop")
async def stop_daemon():
    """Stop the paper trading daemon."""
    global _daemon

    if not _daemon or _daemon._state == DaemonState.STOPPED:
        return {"status": "not_running", "message": "Daemon is not running"}

    _daemon.stop()
    return {"status": "stopped"}


@router.get("/status")
async def get_status():
    """Get current daemon status, NAV, PnL, regime."""
    if not _daemon:
        return {
            "state": "STOPPED",
            "market_status": PaperTradingDaemon.get_market_status(),
            "virtual_nav": 0,
            "total_pnl": 0,
            "message": "Daemon has not been started yet",
        }

    status = _daemon.get_status()
    return {
        "state": status.state.value,
        "started_at": status.started_at,
        "last_signal_time": status.last_signal_time,
        "last_error": status.last_error,
        "current_regime": status.current_regime,
        "hedge_ratio": status.hedge_ratio,
        "virtual_nav": status.virtual_nav,
        "today_pnl": status.today_pnl,
        "total_pnl": status.total_pnl,
        "total_pnl_pct": status.total_pnl_pct,
        "n_positions": status.n_positions,
        "n_trades_today": status.n_trades_today,
        "market_status": status.market_status,
    }


@router.get("/portfolio")
async def get_portfolio():
    """Get current positions and cash balance."""
    if not _daemon:
        raise HTTPException(404, "Daemon not initialized")

    return _daemon.get_portfolio()


@router.get("/trades")
async def get_trades(limit: int = 50, offset: int = 0):
    """Get trade history (most recent first)."""
    if not _daemon:
        raise HTTPException(404, "Daemon not initialized")

    trades = _daemon.get_trade_log(limit=limit, offset=offset)
    return {"trades": trades, "limit": limit, "offset": offset}


@router.get("/nav-history")
async def get_nav_history(limit: int = 365):
    """Get NAV time series for charting."""
    if not _daemon:
        raise HTTPException(404, "Daemon not initialized")

    history = _daemon.get_nav_history(limit=limit)
    return {"history": history}


@router.post("/reset")
async def reset_daemon(request: ResetRequest):
    """
    Reset virtual capital and clear all history.

    Requires confirm=True to prevent accidental resets.
    """
    if not request.confirm:
        raise HTTPException(
            400,
            "Set confirm=true to reset. This will clear all positions, "
            "trades, and NAV history.",
        )

    if not _daemon:
        raise HTTPException(404, "Daemon not initialized")

    _daemon.reset(new_capital=request.new_capital)
    return {
        "status": "reset",
        "new_capital": request.new_capital or _daemon.broker.initial_capital,
    }

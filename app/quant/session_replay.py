"""
Session Replay — how the live book did in one real trading session
===================================================================
Paper simulation of a single NSE session with no look-ahead:

  1. Run the live engine with data through the PREVIOUS session's close only
     (``compute_live_dashboard(as_of=prev_session)``).  That fixes the book
     going into the target session: per-symbol sleeve weights, the
     vol-target leverage, and any open validated pair trades.
  2. Mark that fixed book to market through the target session's 1-minute
     bars (Kite when connected, otherwise Yahoo — labelled), relative to the
     previous official close.
  3. Report the end-of-day result on official closing prices (NSE closes on a
     closing-session VWAP, not the last trade) alongside NIFTY 50.

No orders are placed anywhere — this is arithmetic on historical prices.
Positions are carried from the previous close, so no rebalancing trades or
costs occur intraday; a rebalance due at today's close happens after the
measured window.  One session is statistical noise: the output says so.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.data.nse import _session_closed, ist_today, normalize_session_dates
from app.quant.live_dashboard import BENCHMARK, compute_live_dashboard, fetch_adjusted_close_panel

logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/Reports")
SESSION_OPEN = dtime(9, 15)
SESSION_LAST_MINUTE = dtime(15, 29)

MinuteFetcher = Callable[[list[str], date], tuple[pd.DataFrame, str]]
DailyFetcher = Callable[[list[str], date, date], tuple[pd.DataFrame, str]]


# ── Data sources ───────────────────────────────────────────────────────────────

def _kite_client():
    from app.data import kite

    try:
        return kite.get_client()
    except kite.KiteNotConnected:
        return None


def fetch_minute_closes(symbols: list[str], session: date) -> tuple[pd.DataFrame, str]:
    """1-minute closes for one session, one column per symbol (BENCHMARK = NIFTY 50)."""
    client = _kite_client()
    if client is not None:
        from app.data import kite

        frames = {}
        start = datetime.combine(session, SESSION_OPEN)
        end = datetime.combine(session, dtime(15, 30))
        for sym in symbols:
            exchange, ts = kite.kite_symbol("NIFTY 50" if sym == BENCHMARK else sym)
            try:
                bars = client.historical(client.instrument_token(ts, exchange), "minute", start, end)
            except kite.KiteError as e:
                logger.warning("Kite minute bars unavailable for %s: %s", sym, e)
                continue
            if not bars.empty:
                frames[sym] = bars.set_index("date")["close"]
        return pd.DataFrame(frames), "kite (1-minute, exchange data)"

    import yfinance as yf

    tickers = {(BENCHMARK if s == BENCHMARK else f"{s}.NS"): s for s in symbols}
    raw = yf.download(list(tickers), start=session.isoformat(),
                      end=(session + timedelta(days=1)).isoformat(),
                      interval="1m", auto_adjust=False, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_convert("Asia/Kolkata").tz_localize(None)
    return close, "yfinance (1-minute, delayed)"


def fetch_daily_closes(symbols: list[str], start: date, end: date) -> tuple[pd.DataFrame, str]:
    """Official as-traded daily closes (Kite day candles, else Yahoo unadjusted)."""
    client = _kite_client()
    if client is not None:
        from app.data import kite

        frames = {}
        for sym in symbols:
            exchange, ts = kite.kite_symbol("NIFTY 50" if sym == BENCHMARK else sym)
            try:
                bars = client.historical(client.instrument_token(ts, exchange), "day",
                                         datetime.combine(start, dtime()), datetime.combine(end, dtime(23, 59)))
            except kite.KiteError as e:
                logger.warning("Kite daily bars unavailable for %s: %s", sym, e)
                continue
            if not bars.empty:
                frames[sym] = bars.assign(date=normalize_session_dates(bars["date"])).set_index("date")["close"]
        return pd.DataFrame(frames), "kite (daily official close)"

    import yfinance as yf

    tickers = {(BENCHMARK if s == BENCHMARK else f"{s}.NS"): s for s in symbols}
    raw = yf.download(list(tickers), start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                      auto_adjust=False, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns=tickers)
    close.index = pd.DatetimeIndex(normalize_session_dates(pd.Series(close.index)))
    return close, "yfinance (daily close, unadjusted)"


# ── Core arithmetic (pure, tested) ─────────────────────────────────────────────

def minute_grid(session: date) -> pd.DatetimeIndex:
    return pd.date_range(datetime.combine(session, SESSION_OPEN),
                         datetime.combine(session, SESSION_LAST_MINUTE), freq="1min")


def returns_vs_prev_close(minute_closes: pd.DataFrame, prev_close: pd.Series, session: date) -> pd.DataFrame:
    """
    Per-minute return of each symbol relative to the previous official close.
    Before a symbol's first trade the return is 0 (its price is still the
    previous close); gaps are forward-filled, never interpolated.
    """
    grid = minute_grid(session)
    px = minute_closes.reindex(grid).ffill()
    cols = [c for c in px.columns if c in prev_close.index and np.isfinite(prev_close[c]) and prev_close[c] > 0]
    rets = px[cols].div(prev_close[cols]) - 1.0
    return rets.fillna(0.0)


def pair_return(rets: pd.DataFrame, a: str, b: str, position: int, hedge: float) -> pd.Series:
    """Same spread-return convention as statarb_backtest.backtest_pair."""
    return position * (rets[a] - hedge * rets[b]) / (1 + abs(hedge))


def _drawdown(curve: pd.Series) -> float:
    wealth = 1 + curve
    return float(abs((wealth / wealth.cummax() - 1).min()))


# ── Replay ─────────────────────────────────────────────────────────────────────

def replay_session(
    session: Optional[date] = None,
    *,
    fetch_prices=fetch_adjusted_close_panel,
    fetch_minutes: MinuteFetcher = fetch_minute_closes,
    fetch_daily: DailyFetcher = fetch_daily_closes,
    save: bool = True,
) -> dict:
    # Session calendar from NIFTY 50 daily closes.
    end = session or ist_today()
    calendar, daily_source = fetch_daily([BENCHMARK], end - timedelta(days=15), end)
    sessions = [d.date() for d in calendar[BENCHMARK].dropna().index]
    if not sessions:
        raise RuntimeError("No NIFTY 50 sessions available to define the calendar")
    if session is not None and session not in sessions:
        raise RuntimeError(f"{session} is not an NSE session")
    target = session or sessions[-1]
    if sessions.index(target) == 0:
        raise RuntimeError("Need the previous session's close to replay")
    prev = sessions[sessions.index(target) - 1]

    # 1. Book as of the previous close — nothing from the target session.
    state = compute_live_dashboard(fetch_prices=fetch_prices, as_of=prev)
    if state["as_of"] != str(prev):
        raise RuntimeError(f"Engine state is as of {state['as_of']}, expected {prev}")
    port = state["portfolio"]
    weights: dict[str, float] = port["book_weights"]
    scale = port["vt_scale_next_session"] or 1.0
    open_pairs = [p for p in state["pairs"]["pairs"] if p["validated"] and p["position_after_close"] != 0]
    validated_pairs = [p for p in state["pairs"]["pairs"] if p["validated"]]

    symbols = sorted(set(weights) | {s for p in open_pairs for s in (p["x"], p["y"])} | {BENCHMARK})

    # 2. Previous and target official closes; target-session minute bars.
    daily, daily_source = fetch_daily(symbols, prev - timedelta(days=7), target)
    daily = daily.loc[: pd.Timestamp(target)]
    if pd.Timestamp(prev) not in daily.index or pd.Timestamp(target) not in daily.index:
        raise RuntimeError("Official closes for both sessions are required")
    prev_close, target_close = daily.loc[pd.Timestamp(prev)], daily.loc[pd.Timestamp(target)]
    minutes, minute_source = fetch_minutes(symbols, target)
    rets = returns_vs_prev_close(minutes, prev_close, target)

    missing = sorted(s for s in symbols if s not in rets.columns)
    held = {s: w for s, w in weights.items() if s in rets.columns}
    raw_curve = sum((rets[s] * w for s, w in held.items()), pd.Series(0.0, index=rets.index))
    book_curve = raw_curve * scale
    pair_curves = {
        p["pair"]: pair_return(rets, p["x"], p["y"], p["position_after_close"], p["entry_hedge_ratio"])
        for p in open_pairs if p["x"] in rets and p["y"] in rets
    }
    nifty_curve = rets[BENCHMARK] if BENCHMARK in rets else None

    # 3. End of day on official closes.
    eod = (target_close / prev_close - 1.0)
    contributions = {s: float(eod[s] * w * scale) for s, w in held.items() if np.isfinite(eod.get(s, np.nan))}
    book_eod = float(sum(contributions.values()))
    nifty_eod = float(eod[BENCHMARK]) if BENCHMARK in eod and np.isfinite(eod[BENCHMARK]) else None
    pair_eod = {
        p["pair"]: float(p["position_after_close"] * (eod[p["x"]] - p["entry_hedge_ratio"] * eod[p["y"]])
                         / (1 + abs(p["entry_hedge_ratio"])))
        for p in open_pairs
    }

    def curve_points(curve: pd.Series, every: int = 5) -> list[dict]:
        pts = curve.iloc[::every]
        if pts.index[-1] != curve.index[-1]:
            pts = pd.concat([pts, curve.iloc[-1:]])
        return [{"time": t.strftime("%H:%M"), "value": round(float(v), 6)} for t, v in pts.items()]

    ranked = sorted(contributions.items(), key=lambda kv: kv[1])
    gross = float(sum(abs(w) for w in held.values()) * scale)
    result = {
        "session": str(target),
        "previous_session": str(prev),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_complete": target < ist_today() or _session_closed(),
        "sources": {"minute": minute_source, "daily": daily_source,
                    "engine": state["data_source"]},
        "book": {
            "as_of_close": str(prev),
            "symbols": len(held),
            "net_exposure": round(float(sum(held.values()) * scale), 4),
            "gross_exposure": round(gross, 4),
            "vol_target_scale": round(scale, 4),
            "weights": {s: round(w * scale, 6) for s, w in held.items()},
            "sleeve_rebalance": {
                name: {"decision_date": c["decision_date"], "next_decision_date": c["next_decision_date"]}
                for name, c in port["current_composition"].items()
            },
            "engine_oos_validated": port["oos_vol_targeted"]["validated"],
        },
        "result": {
            "book_return_eod": round(book_eod, 6),
            "nifty_return_eod": None if nifty_eod is None else round(nifty_eod, 6),
            "excess_vs_nifty": None if nifty_eod is None else round(book_eod - nifty_eod, 6),
            "book_intraday_high": round(float(book_curve.max()), 6),
            "book_intraday_low": round(float(book_curve.min()), 6),
            "book_intraday_max_drawdown": round(_drawdown(book_curve), 6),
            "last_minute_vs_official_close_gap": round(float(book_curve.iloc[-1]) - book_eod, 6),
            "pairs_validated": len(validated_pairs),
            "pairs_open": len(open_pairs),
            "pair_returns_eod": {k: round(v, 6) for k, v in pair_eod.items()},
            "top_contributors": [{"symbol": s, "contribution": round(c, 6)} for s, c in ranked[::-1][:5]],
            "bottom_contributors": [{"symbol": s, "contribution": round(c, 6)} for s, c in ranked[:5]],
        },
        "curves": {
            "book": curve_points(book_curve),
            "nifty": curve_points(nifty_curve) if nifty_curve is not None else [],
            **{f"pair:{k}": curve_points(v) for k, v in pair_curves.items()},
        },
        "data_gaps": missing,
        "caveats": [
            "One session is statistical noise; it neither validates nor refutes the strategy.",
            "Book fixed at the previous close; no intraday trading or costs are simulated.",
            "Intraday curve uses last-trade minute closes; EOD uses NSE official closes.",
            "Engine portfolio has not passed the out-of-sample gate"
            if not port["oos_vol_targeted"]["validated"] else "Engine portfolio passed the OOS gate.",
            "Not investment advice.",
        ],
    }
    if save and result["session_complete"]:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"session_replay_{target:%Y%m%d}.json").write_text(json.dumps(result, indent=2))
    return result

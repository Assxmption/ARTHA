"""
ARTHA Quant API Routes
======================
REST endpoints for the quant engine dashboard.

Endpoints:
  GET  /api/quant/reports              → List all backtest reports
  GET  /api/quant/reports/latest       → Get the latest report
  GET  /api/quant/reports/{filename}   → Get specific report
  GET  /api/quant/portfolio            → Current portfolio holdings & signals
  GET  /api/quant/performance          → Historical performance data for charting
  GET  /api/quant/regime               → Current and historical regime state
  GET  /api/quant/pairs                → Current pair trading signals
"""

import json
import logging

import pandas as pd
from pathlib import Path
from datetime import datetime

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.security.auth import optional_user, require_admin
from app.security.net import client_ip

_SYMBOL_RE = re.compile(r"^[A-Z0-9&_\-^.]{1,25}$")

logger = logging.getLogger("artha.api.quant")

quant_router = APIRouter(prefix="/api/quant", tags=["Quant Engine"])

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "backtest_reports"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


@quant_router.get("/mcx/contracts")
async def list_mcx_contracts(commodity: str | None = None):
    """List configured MCX contract metadata used for sizing."""
    from app.data.mcx_contracts import MCXContractMaster

    master = MCXContractMaster()
    return {"contracts": master.to_dict(commodity)}


@quant_router.get("/mcx/ohlcv/{commodity}")
async def get_mcx_ohlcv(
    commodity: str,
    start_date: str,
    end_date: str | None = None,
    instrument: str = "FUTCOM",
    include_contract_metadata: bool = False,
):
    """Read-only MCX EOD history from the cached/provider-backed data layer."""
    from datetime import date

    from app.data.mcx import fetch_mcx_history
    from app.data.mcx_contracts import MCXContractMaster, contract_metadata_for_frame

    try:
        start = date.fromisoformat(start_date)
        finish = date.fromisoformat(end_date) if end_date else None
        frame = fetch_mcx_history(
            start,
            finish,
            commodity=commodity,
            instrument=instrument,
        )
        if include_contract_metadata and not frame.empty:
            frame = contract_metadata_for_frame(frame, MCXContractMaster())
        records = frame.where(frame.notna(), None).to_dict(orient="records")
        for record in records:
            for key in ("trade_date", "expiry_date"):
                if record.get(key) is not None:
                    record[key] = record[key].isoformat()
        return {
            "commodity": commodity.upper(),
            "instrument": instrument.upper(),
            "data": records,
            "count": len(records),
            "sizing_metadata_included": include_contract_metadata,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("MCX data request failed for %s", commodity)
        raise HTTPException(status_code=502, detail=f"MCX data unavailable: {exc}") from exc


def _load_report(filename: str) -> dict:
    """Load a JSON report from the backtest reports directory."""
    path = REPORT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report {filename} not found")
    if not path.suffix == ".json":
        raise HTTPException(status_code=400, detail="Invalid report format")
    with open(path) as f:
        return json.load(f)


def _get_latest_report() -> tuple[str, dict]:
    """Get the most recent v4 report, falling back to any report."""
    if not REPORT_DIR.exists():
        raise HTTPException(status_code=404, detail="No reports directory")
    
    reports = sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports available")
    
    # Prefer v4 reports
    v4_reports = [r for r in reports if "v4" in r.name]
    target = v4_reports[0] if v4_reports else reports[0]
    
    with open(target) as f:
        return target.name, json.load(f)


@quant_router.get("/reports")
async def list_reports():
    """List all available backtest reports."""
    if not REPORT_DIR.exists():
        return {"reports": []}
    
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path) as f:
            data = json.load(f)
        reports.append({
            "filename": path.name,
            "version": data.get("version", "unknown"),
            "generated_at": data.get("generated_at"),
            "universe_size": data.get("universe_size") or data.get("combined_strategy", {}).get("n_stocks"),
            "benchmark_sharpe": data.get("benchmark_sharpe"),
            "strategy_sharpe": (
                data.get("combined_strategy", {}).get("vt_sharpe")
                or data.get("factor_alpha_tilt", {}).get("vt_sharpe")
            ),
        })
    return {"reports": reports}


@quant_router.get("/reports/latest")
async def get_latest_report():
    """Get the most recent backtest report with full data."""
    filename, data = _get_latest_report()
    return {"filename": filename, **data}


@quant_router.get("/reports/{filename}")
async def get_report(filename: str):
    """Get a specific report by filename."""
    data = _load_report(filename)
    return {"filename": filename, **data}


async def _live_dashboard() -> dict:
    """Live engine result (cached; recomputed when a new session closes)."""
    from starlette.concurrency import run_in_threadpool

    from app.quant.live_dashboard import LiveDataError, live_dashboard_cache

    try:
        return await run_in_threadpool(live_dashboard_cache.get)
    except LiveDataError as e:
        raise HTTPException(
            status_code=503,
            detail={"code": "LIVE_DATA_UNAVAILABLE", "message": str(e)},
        )


def _freshness(result: dict) -> dict:
    return {
        "as_of": result["as_of"],
        "computed_at": result["computed_at"],
        "provisional": result["provisional"],
        "stale": result.get("stale", False),
        "refreshing": result.get("refreshing", False),
        "engine_version": result["engine_version"],
        "data_source": result["data_source"],
        "windows": result["windows"],
        "disclaimer": result["disclaimer"],
    }


@quant_router.get("/portfolio")
async def get_portfolio():
    """
    Live portfolio statistics, recomputed from current NSE prices.

    Out-of-sample metrics of the predeclared sleeve portfolio over the last
    504 sessions (train-only weights), plus the rule-6 gate outcome.  Values
    are reported as they come out, including when the gate fails.
    """
    result = await _live_dashboard()
    port, pairs = result["portfolio"], result["pairs"]
    vt, raw, bench = port["oos_vol_targeted"], port["oos"], port["benchmark_oos"]
    return {
        **_freshness(result),
        "universe_size": result["universe"]["with_data"],
        "universe": result["universe"],
        "gate": result["gate"],
        # Field names kept for the existing dashboard UI.
        "vt_sharpe": vt["sharpe_ratio"],
        "vt_max_dd": vt["max_drawdown"],
        "vt_return": vt["annualized_return"],
        "vt_validated": vt["validated"],
        "total_sharpe": raw["sharpe_ratio"],
        "total_sortino": raw["sortino_ratio"],
        "total_return": raw["annualized_return"],
        "total_max_dd": raw["max_drawdown"],
        "win_rate": raw["win_rate"],
        "win_rate_basis": "fraction of positive OOS sessions",
        "profit_factor": raw["profit_factor"],
        "calmar_ratio": raw["calmar_ratio"],
        "n_days": raw["num_trading_days"],
        "benchmark_sharpe": bench["sharpe_ratio"],
        "benchmark_return": bench["annualized_return"],
        "benchmark_max_dd": bench["max_drawdown"],
        "sleeves": port["sleeves"],
        "weights": port["weights"],
        "current_composition": port["current_composition"],
        "equity_curve": port["equity_curve"],
        "pairs_validated": pairs["validated"],
        "pairs_total": pairs["backtested"],
        "top_pairs": pairs["pairs"][:5],
    }


@quant_router.get("/pairs")
async def get_pairs():
    """Live stat-arb scan: train-window discovery, OOS-gated validation."""
    result = await _live_dashboard()
    return {**_freshness(result), **result["pairs"], "gate": result["gate"]}


@quant_router.post("/live/refresh")
async def refresh_live_dashboard(user=Depends(require_admin)):
    """Force a recomputation (e.g. right after the session closes)."""
    from starlette.concurrency import run_in_threadpool

    from app.quant.live_dashboard import live_dashboard_cache

    current = await run_in_threadpool(live_dashboard_cache.get)
    computed = datetime.fromisoformat(current["computed_at"])
    # A full recompute downloads ~5 years for the whole universe; don't let
    # repeated clicks hammer Yahoo.
    if (datetime.now(computed.tzinfo) - computed).total_seconds() < 120:
        return {**_freshness(current), "refreshed": False, "reason": "computed under 2 minutes ago"}
    result = await run_in_threadpool(live_dashboard_cache.refresh)
    return {**_freshness(result), "refreshed": True}


@quant_router.get("/performance")
async def get_performance():
    """
    Get performance comparison data for charting.
    Returns strategy vs benchmark metrics across versions.
    """
    if not REPORT_DIR.exists():
        return {"versions": []}
    
    versions = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        with open(path) as f:
            data = json.load(f)
        
        version = data.get("version", "unknown")
        strategy = data.get("combined_strategy") or data.get("factor_alpha_tilt", {})
        
        versions.append({
            "version": version,
            "filename": path.name,
            "generated_at": data.get("generated_at"),
            "strategy_sharpe": strategy.get("vt_sharpe") or strategy.get("total_sharpe"),
            "strategy_return": strategy.get("vt_return") or strategy.get("total_return"),
            "strategy_max_dd": strategy.get("vt_max_dd") or strategy.get("total_max_dd"),
            "strategy_vol": strategy.get("total_vol"),
            "benchmark_sharpe": data.get("benchmark_sharpe"),
            "win_rate": strategy.get("total_win_rate") or strategy.get("vt_win_rate"),
            "profit_factor": strategy.get("total_profit_factor"),
        })
    
    return {"versions": versions}


def _nifty_daily_kite():
    """NIFTY 50 daily closes (5y) from Kite, as a DataFrame with 'Close'; None if unavailable."""
    from datetime import datetime as _dt, timedelta as _td

    from app.data import kite
    from app.data.nse import ist_today, normalize_session_dates

    try:
        client = kite.get_client()
        token = client.instrument_token("NIFTY 50", "NSE")
        end = ist_today()
        bars = client.historical(token, "day", _dt.combine(end - _td(days=5 * 365), _dt.min.time()),
                                 _dt.combine(end, _dt.max.time().replace(microsecond=0)))
    except Exception:
        return None
    if bars.empty:
        return None
    bars.index = pd.DatetimeIndex(normalize_session_dates(bars["date"]))
    return bars.rename(columns={"close": "Close"})


@quant_router.get("/regime")
async def get_regime():
    """
    Get current market regime state from live HMM detection on NIFTY 50.
    
    Runs a 3-state Gaussian HMM on 5 years of NIFTY 50 data to classify
    the current market as BULL, BEAR, or SIDEWAYS. Results are cached
    for 4 hours to avoid repeated computation.
    """
    import time as _time
    
    cache_key = "_regime_cache"
    cache_ttl = 4 * 3600  # 4 hours
    
    # Check cache
    cached = getattr(get_regime, cache_key, None)
    if cached and (_time.time() - cached.get("_ts", 0)) < cache_ttl:
        return cached["data"]
    
    try:
        import yfinance as yf
        from app.quant.regime import RegimeDetector
        
        # 5 years of NIFTY 50 closes: Kite (exchange data) when connected.
        df = _nifty_daily_kite()
        regime_source = "kite:NSE:NIFTY 50"
        if df is None:
            df = yf.Ticker("^NSEI").history(period="5y")
            regime_source = "yfinance:^NSEI"

        if df.empty:
            return {
                "current_regime": "UNKNOWN",
                "regime_statistics": {},
                "regime_timeline": [],
                "error": "Could not fetch NIFTY 50 data",
            }
        
        prices = df['Close']
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # Run HMM
        detector = RegimeDetector(n_states=3)
        success = detector.fit(prices)
        
        if not success:
            return {
                "current_regime": "UNKNOWN",
                "regime_statistics": {},
                "regime_timeline": [],
                "error": "HMM fitting failed — insufficient data",
            }
        
        current = detector.current_regime(prices)
        stats = detector.get_state_statistics()
        regimes = detector.predict(prices)
        
        # Build regime timeline (transitions)
        transitions = []
        prev = None
        for dt, r in regimes.items():
            label = r.value if hasattr(r, 'value') else str(r)
            if label != "UNKNOWN" and label != prev:
                transitions.append({
                    "date": str(dt.date()),
                    "regime": label,
                })
                prev = label
        
        # Regime distribution over last year
        last_year = regimes[regimes.index >= str((datetime.now().date() - __import__('datetime').timedelta(days=365)))]
        distribution = {}
        for r in last_year:
            label = r.value if hasattr(r, 'value') else str(r)
            distribution[label] = distribution.get(label, 0) + 1
        
        # Time in the current state and the model's own probability of leaving
        # it next session (1 - A[s, s] from the fitted transition matrix).
        labelled = [r.value if hasattr(r, 'value') else str(r) for r in regimes.values]
        current_label = current.value if hasattr(current, 'value') else str(current)
        sessions_in_state = 0
        for label in reversed(labelled):
            if label != current_label:
                break
            sessions_in_state += 1
        state_since = (
            str(regimes.index[len(labelled) - sessions_in_state].date())
            if sessions_in_state else None
        )
        prob_leave = None
        model = getattr(detector, "_model", None)
        state_labels = getattr(detector, "_state_labels", {})
        idx = next((i for i, lab in state_labels.items()
                    if (lab.value if hasattr(lab, "value") else str(lab)) == current_label), None)
        if model is not None and idx is not None and hasattr(model, "transmat_"):
            prob_leave = round(float(1.0 - model.transmat_[idx, idx]), 4)

        result = {
            "current_regime": current.value if hasattr(current, 'value') else str(current),
            "as_of": str(prices.index[-1].date()),
            "source": regime_source,
            "sessions_in_state": sessions_in_state,
            "state_since": state_since,
            "prob_leave_next_session": prob_leave,
            "data_points": len(prices),
            "date_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
            "regime_statistics": stats,
            "regime_transitions": transitions[-20:],  # Last 20 transitions
            "last_year_distribution": distribution,
        }
        
        # Cache
        setattr(get_regime, cache_key, {"data": result, "_ts": _time.time()})
        return result
    
    except Exception as e:
        logger.error("Regime detection failed: %s", e, exc_info=True)
        return {
            "current_regime": "UNKNOWN",
            "regime_statistics": {},
            "error": str(e),
        }


@quant_router.get("/fundamentals/{symbol}")
def get_fundamentals(symbol: str):
    """Get multi-year row-level fundamentals for a specific symbol."""
    from app.data.fundamentals import get_full_fundamentals
    
    try:
        rows = get_full_fundamentals(symbol.upper())
        if not rows:
            raise HTTPException(status_code=404, detail=f"No fundamental data found for {symbol}")
            
        # Group by fiscal year for easy frontend rendering
        data_by_year = {}
        for row in rows:
            if row.fiscal_year not in data_by_year:
                data_by_year[row.fiscal_year] = {}
            data_by_year[row.fiscal_year][row.metric] = {
                "value": row.value,
                "unit": row.unit
            }
            
        return {
            "symbol": symbol.upper(),
            "data": data_by_year
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal error; see server logs")

# Shared narrative cache: a narrative is generated once per symbol per IST day
# and served to every visitor, so page views never spend LLM quota twice.
_NARRATIVE_TTL_S = 12 * 3600
_narrative_cache: dict[str, tuple[float, str]] = {}
_narrative_locks: dict[str, "threading.Lock"] = {}
_narrative_guard = __import__("threading").Lock()


def _narrative_key(symbol: str) -> str:
    from app.data.nse import ist_today

    return f"{symbol}:{ist_today().isoformat()}"


@quant_router.get("/fundamentals/{symbol}/narrative")
@quant_router.get("/fundamentals/{symbol}/narrative_stream")
def stream_fundamentals_narrative_route(symbol: str, request: Request, user=Depends(optional_user)):
    """Stream the LLM narrative of a symbol's fundamentals (cached per symbol per day)."""
    import threading
    import time

    from fastapi.responses import StreamingResponse

    from app.agents.fundamentals_agent import stream_fundamentals_narrative
    from app.data.fundamentals import get_full_fundamentals
    from app.factstore.store import FactStore
    from app.llm.keys import KeyFormatError, LLMKeyRequired, keys_from_headers, resolve_access, use_keys

    symbol = symbol.upper().strip()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(400, "Invalid symbol")
    key = _narrative_key(symbol)
    cached = _narrative_cache.get(key)
    if cached and time.time() - cached[0] < _NARRATIVE_TTL_S:
        return StreamingResponse(iter([cached[1]]), media_type="text/plain",
                                 headers={"X-Narrative-Cache": "hit"})

    # A fresh narrative spends LLM quota: the visitor's own key, or the
    # owner's under the demo policy (app/llm/keys.py).
    try:
        user_keys = keys_from_headers(request.headers)
    except KeyFormatError as e:
        raise HTTPException(400, {"code": "LLM_KEY_INVALID", "message": str(e)})
    sub = (user or {}).get("sub")
    visitor = sub if sub and sub not in ("anonymous", "public") else f"ip:{client_ip(request)}"
    try:
        access = resolve_access(user_keys, visitor)
    except LLMKeyRequired as e:
        raise HTTPException(402, {"code": "LLM_KEY_REQUIRED", "message": e.reason})

    with _narrative_guard:
        lock = _narrative_locks.setdefault(key, threading.Lock())

    def generator():
        # Yield immediately so the frontend replaces its spinner with the stream.
        yield f"*Compiling financial statements and retrieving row-level metrics for {symbol}...*\n\n"
        with lock, use_keys(access.keys):
            hit = _narrative_cache.get(key)      # another request may have just generated it
            if hit and time.time() - hit[0] < _NARRATIVE_TTL_S:
                yield hit[1]
                return
            text = ""
            try:
                rows = get_full_fundamentals(symbol)
                if not rows:
                    yield f"No fundamental data found for {symbol}."
                    return
                store = FactStore()
                job_id = f"ui_fund_{symbol}_{int(time.time())}"
                for row in rows:
                    row.job_id = job_id
                store.put_facts(rows)
                for chunk in stream_fundamentals_narrative(symbol, job_id, store, rows):
                    text += chunk
                    yield chunk
            except Exception as e:
                logger.error("Error streaming fundamentals narrative for %s: %s", symbol, type(e).__name__)
                yield "\n\n*Error generating narrative.*"
                return
            if text and "*Error generating narrative.*" not in text:
                _narrative_cache[key] = (time.time(), text)
                if len(_narrative_cache) > 500:     # bounded: drop the oldest entries
                    for k, _ in sorted(_narrative_cache.items(), key=lambda kv: kv[1][0])[:100]:
                        _narrative_cache.pop(k, None)

    # A synchronous generator runs in a threadpool, so the event loop never blocks.
    return StreamingResponse(generator(), media_type="text/plain", headers={"X-Narrative-Cache": "miss"})

def get_fundamentals_narrative(symbol: str):
    """Generate LLM narrative for a specific symbol's fundamentals."""
    from app.data.fundamentals import get_full_fundamentals
    from app.factstore.store import FactStore
    from app.agents.fundamentals_agent import narrate_fundamentals
    import time
    
    try:
        rows = get_full_fundamentals(symbol.upper())
        if not rows:
            raise HTTPException(status_code=404, detail=f"No fundamental data found for {symbol}")
            
        # Run the LLM narrator model
        try:
            store = FactStore()
            job_id = f"ui_fund_{symbol.upper()}_{int(time.time())}"
            # Assign job_id to rows so citations can be validated
            for row in rows:
                row.job_id = job_id
            store.put_facts(rows)
            
            narrative_fact = narrate_fundamentals(symbol.upper(), job_id, store, rows)
            narrative = narrative_fact.content if narrative_fact else "## Fundamental Analysis\n\n*LLM narration unavailable.*"
        except Exception as e:
            logger.error(f"Failed to generate narrative for {symbol}: {e}")
            narrative = "## Fundamental Analysis\n\n*Error generating narrative.*"
            
        return {
            "symbol": symbol.upper(),
            "narrative": narrative
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fundamentals narrative for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal error; see server logs")

_DAILY_RULES = {"1d": None, "1wk": "W-FRI", "1mo": "ME"}


def _resample_daily(df, rule):
    import pandas as pd

    g = df.set_index("date").resample(rule)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).dropna(subset=["open"])
    # Label each bar with its first session, as charting platforms do.
    firsts = df.set_index("date")["close"].resample(rule).apply(lambda x: x.index.min())
    out.index = pd.DatetimeIndex(firsts.loc[out.index].values)
    return out.reset_index(names="date")


@quant_router.get("/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    days: int = Query(180, ge=1, le=3650),
    interval: str = "1d",
    indicators: str = "",
):
    """
    Chart data: OHLCV bars plus requested indicators, with provenance.

    interval: 1m, 3m, 5m, 15m, 30m, 60m/1h (intraday) or 1d, 1wk, 1mo.
    indicators: comma list from app.quant.indicators.AVAILABLE
    (sma20, sma50, sma200, ema9, ema21, bb, vwap, rsi, macd, atr).
    Source is Kite Connect when a session is active, otherwise NSE archive /
    Yahoo — the `source` and `live` fields say which.
    """
    from datetime import date, timedelta

    import pandas as pd
    from starlette.concurrency import run_in_threadpool

    from app.data import kite
    from app.data.intraday import INTRADAY_INTERVALS, fetch_intraday
    from app.data.nse import fetch_nse_ohlcv, ist_today
    from app.quant import indicators as ind

    requested = [n for n in (x.strip().lower() for x in indicators.split(",")) if n]
    unknown = [n for n in requested if n not in ind.AVAILABLE]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown indicators: {unknown}")
    intraday = interval in INTRADAY_INTERVALS
    if not intraday and interval not in _DAILY_RULES:
        raise HTTPException(status_code=400, detail=f"Unsupported interval {interval!r}")

    try:
        if intraday:
            df, source = await run_in_threadpool(fetch_intraday, symbol.upper(), interval, days)
            df = df.copy()
            df["provisional"] = False
        else:
            # Extra history so long-window indicators (SMA200) are warmed up.
            warmup = {"1d": 300, "1wk": 1500, "1mo": 6000}[interval]
            end_date = ist_today()
            start_date = end_date - timedelta(days=days + warmup)
            df = await run_in_threadpool(fetch_nse_ohlcv, symbol.upper(), start_date, end_date)
            source = str(df["source"].iloc[-1]) if len(df) and "source" in df else "none"
            if _DAILY_RULES[interval] and len(df):
                provisional_last = bool(df["provisional"].iloc[-1]) if "provisional" in df else False
                df = _resample_daily(df, _DAILY_RULES[interval])
                df["provisional"] = False
                df.loc[df.index[-1], "provisional"] = provisional_last
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No OHLCV data found for {symbol}")

        df = df.reset_index(drop=True)
        series = ind.compute(df, requested, intraday=intraday)
        if not intraday:
            keep = df["date"] >= pd.Timestamp(date.today() - timedelta(days=days))
            idx = keep.to_numpy().nonzero()[0]
            df = df.loc[keep].reset_index(drop=True)
            series = {k: [v[i] for i in idx] for k, v in series.items()}

        times = [int(t.timestamp()) for t in pd.to_datetime(df["date"])]
        bars = [
            {"time": t, "open": float(o), "high": float(h), "low": float(l),
             "close": float(c), "volume": float(v)}
            for t, o, h, l, c, v in zip(times, df["open"], df["high"], df["low"], df["close"], df["volume"])
        ]
        provisional = bool(df["provisional"].iloc[-1]) if "provisional" in df else False
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "source": source,
            "live": source.startswith("kite"),
            "kite_connected": kite.is_connected(),
            "last_bar": {"time": times[-1], "source": source, "provisional": provisional},
            "data": bars,
            "indicators": {"times": times, **series},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal error; see server logs")


@quant_router.get("/ticker")
async def get_ticker():
    """Top-bar index levels (NIFTY 50, BANKNIFTY, SENSEX, INDIA VIX)."""
    from starlette.concurrency import run_in_threadpool

    from app.data.intraday import get_index_ticker

    return await run_in_threadpool(get_index_ticker)


@quant_router.get("/movers")
async def get_movers(top: int = Query(10, ge=1, le=50)):
    """Gainers, losers, most active and breadth across the ARTHA universe."""
    from starlette.concurrency import run_in_threadpool

    from app.data.market_movers import movers_cache

    try:
        return await run_in_threadpool(movers_cache.get, top)
    except Exception as e:
        logger.warning("Movers failed: %s", e)
        raise HTTPException(status_code=503, detail="Market data unavailable right now")


@quant_router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Live quote with market depth (Kite) or a delayed snapshot (Yahoo)."""
    from starlette.concurrency import run_in_threadpool

    from app.data.intraday import get_quote as _quote

    try:
        return await run_in_threadpool(_quote, symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=502, detail="Quote unavailable right now")


@quant_router.get("/session-replay")
async def session_replay(date: str | None = Query(None, max_length=10), refresh: bool = False,
                         user=Depends(optional_user)):
    """
    Paper replay of one NSE session: the live engine's book as of the
    previous close, marked to market through the session's 1-minute bars.
    Completed sessions are cached in data/Reports/session_replay_YYYYMMDD.json.
    """
    import json as _json
    from datetime import date as _date

    from starlette.concurrency import run_in_threadpool

    from app.quant.session_replay import REPORT_DIR, replay_session

    from app.security.auth import is_admin

    if refresh and not is_admin(user):
        refresh = False          # forcing a ~1-minute recompute is operator-only
    target = None
    if date:
        try:
            target = _date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
        cached = REPORT_DIR / f"session_replay_{target:%Y%m%d}.json"
        if cached.exists() and not refresh:
            return _json.loads(cached.read_text())
    import time as _time

    if target is None and not refresh:
        # A completed session's replay never changes: reuse the saved report
        # for the most recent closed session instead of recomputing (~1 min).
        from app.data.nse import _now_ist
        from app.quant.live_dashboard import _last_close_boundary

        last_closed = _last_close_boundary(_now_ist()).date()
        saved = REPORT_DIR / f"session_replay_{last_closed:%Y%m%d}.json"
        if saved.exists():
            return _json.loads(saved.read_text())

    memo = _replay_memo.get("latest")
    if target is None and memo and not refresh and _time.monotonic() - memo[0] < 600:
        return memo[1]
    if _replay_slot.locked():
        raise HTTPException(status_code=429, detail="A session replay is already being computed; try again shortly")
    async with _replay_slot:     # at most one replay computation at a time
        try:
            result = await run_in_threadpool(lambda: replay_session(target))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
    if target is None:
        _replay_memo["latest"] = (_time.monotonic(), result)
    return result


_replay_memo: dict = {}
_replay_slot = __import__("asyncio").Lock()


@quant_router.get("/options/{symbol}")
async def get_options_chain(symbol: str):
    """
    Get real NSE options chain (Calls and Puts) with OI, volume, IV.
    
    Sources (in priority order):
      1. nselib — purpose-built NSE scraper
      2. Direct NSE India API with proper headers
      3. Explicit unavailable response if both fail

    Synthetic option chains are never returned by the production API. A
    missing feed must not influence options signals, sizing, or paper trades.
    """
    from app.data.nse_fno import get_option_chain
    
    try:
        result = get_option_chain(symbol.upper())
        
        # Missing real data is an unavailable market-data condition, not a
        # reason to fabricate prices or open interest.
        if result.get("source") == "none" or not result.get("calls"):
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "OPTIONS_DATA_UNAVAILABLE",
                    "symbol": symbol.upper(),
                    "message": "Real NSE option-chain data is currently unavailable",
                },
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "OPTIONS_DATA_UNAVAILABLE",
                "symbol": symbol.upper(),
                "message": "Real NSE option-chain data is currently unavailable",
            },
        ) from e


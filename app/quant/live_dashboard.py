"""
Live Dashboard Engine
=====================
Recomputes the dashboard's portfolio and stat-arb panels from current NSE
data instead of serving a frozen backtest report.

Every number is produced by deterministic code from real prices fetched at
run time (AGENTS.md rule 1); nothing is synthetic or hard-coded.

Protocol (chronological, no look-ahead):
  * Universe — the point-in-time security master, active as of today.
  * Prices — Yahoo adjusted closes (splits/dividends), because returns over
    multi-year windows are meaningless on as-traded prices across corporate
    actions.  The NIFTY 50 index defines the session calendar.
  * Windows — the last ``MIN_OOS_TRADING_DAYS`` sessions are out-of-sample;
    the ``TRAIN_YEARS`` before that are training.
  * Portfolio — the repo's predeclared causal rank sleeves
    (app/quant/sleeve_validation.py) combined with inverse-volatility weights
    fitted on the training window only, then volatility-targeted with a
    one-day-lagged estimate.  OOS metrics are reported as they come out —
    good or bad — and the rule-6 gate (Sharpe ≥ 1, max DD ≤ 25 %, ≥ 504 OOS
    sessions) decides the ``validated`` flag.
  * Pairs — Engle-Granger discovery on the training window only, OOS
    backtest of the most significant candidates, same rule-6 gate.  The
    number of hypotheses tested is reported because a 5 % test over ~2,000
    pairs is expected to produce ~100 false positives; only the OOS gate
    separates signal from that noise.

Output describes historical, validated-or-not statistics.  It is not
investment advice and makes no recommendation (rule 11).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.data.nse import _now_ist, _session_closed, ist_today, normalize_session_dates
from app.data.security_master import PointInTimeSecurityMaster
from app.quant.backtest import MIN_OOS_TRADING_DAYS, compute_metrics
from app.quant.multi_strategy import vol_target_returns
from app.quant.sleeve_validation import RankSleeveConfig, _portfolio_weights, rank_sleeve_returns

logger = logging.getLogger(__name__)

ENGINE_VERSION = "live-2"  # bump invalidates cached results
TRAIN_YEARS = 3
OOS_SESSIONS = MIN_OOS_TRADING_DAYS
TARGET_VOL = 0.10
MAX_PAIRS_BACKTESTED = 20
SECURITY_MASTER_PATH = Path("data_cache/nse/security_master.json")
CACHE_PATH = Path("data_cache/live/dashboard.json")
BENCHMARK = "^NSEI"

# Predeclared before seeing any OOS data; identical to the research harness.
SLEEVES: tuple[RankSleeveConfig, ...] = (
    RankSleeveConfig("momentum_12m_skip_1m", 252, 21, 21),
    RankSleeveConfig("momentum_12m_skip_3m", 252, 63, 21),
    RankSleeveConfig("momentum_6m", 126, 21, 21),
    RankSleeveConfig("reversal_1m", 21, 1, 5, reverse_rank=True),
    RankSleeveConfig("reversal_3m", 63, 5, 21, reverse_rank=True),
    RankSleeveConfig("momentum_long_short", 252, 21, 21, long_only=False),
)

PriceFetcher = Callable[[list[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


class LiveDataError(RuntimeError):
    """Raised when live data is insufficient to compute the dashboard honestly."""


# ── Data ────────────────────────────────────────────────────────────────────────

def fetch_adjusted_close_panel(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Adjusted closes for NSE symbols plus NIFTY 50, one column per symbol."""
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols] + [BENCHMARK]
    raw = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise LiveDataError("Yahoo returned no price data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.rename(columns=lambda t: t[:-3] if t.endswith(".NS") else t)
    close.index = pd.DatetimeIndex(normalize_session_dates(pd.Series(close.index)))
    return close.sort_index()


def _metrics_dict(returns: pd.Series) -> dict:
    m = compute_metrics(returns.dropna())
    gate = {
        k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else float(v))
        for k, v in asdict(m).items()
        if k != "is_valid"
    }
    gate["validated"] = bool(m.validate())
    return gate


# ── Portfolio panel ────────────────────────────────────────────────────────────

def sleeve_book_after(
    panel: pd.DataFrame,
    config: RankSleeveConfig,
    test_start: pd.Timestamp,
    position: int,
    master: PointInTimeSecurityMaster,
) -> dict:
    """
    The book a sleeve holds going into session ``position + 1``.

    Replicates rank_sleeve_returns' schedule exactly (decisions every
    ``rebalance_days`` from the first eligible session of the OOS window), so
    the replay and the reported OOS backtest are the same strategy.
    """
    dates = panel.index
    first_pos = int(np.searchsorted(dates.values, np.datetime64(test_start), side="left"))
    first = max(config.lookback_days + config.skip_days, first_pos)
    if position < first:
        return {"long": [], "short": [], "decision_date": None, "next_decision_date": None}
    decision = first + ((position - first) // config.rebalance_days) * config.rebalance_days
    score_end = decision - config.skip_days
    score_start = score_end - config.lookback_days
    score = (panel.iloc[score_end] / panel.iloc[score_start] - 1.0)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    active = set(master.symbols_at(dates[decision].date()))
    score = score[score.index.isin(active)]
    nxt = decision + config.rebalance_days
    out = {
        "decision_date": str(dates[decision].date()),
        "next_decision_date": str(dates[nxt].date()) if nxt < len(dates) else f"+{nxt - len(dates) + 1} sessions",
    }
    if len(score) < 5:
        return {"long": [], "short": [], **out}
    ranked = score.sort_values(ascending=config.reverse_rank)
    n = max(1, int(len(ranked) * config.top_fraction))
    return {
        "long": list(ranked.index[:n]),
        "short": [] if config.long_only else list(ranked.index[-n:]),
        **out,
    }


def book_weights(composition: dict, weights: dict[str, float]) -> dict[str, float]:
    """Net per-symbol exposure (fraction of capital) of the combined sleeve book."""
    net: dict[str, float] = {}
    by_name = {cfg.name: cfg for cfg in SLEEVES}
    for name, book in composition.items():
        cfg, w = by_name[name], weights.get(name, 0.0)
        longs, shorts = book["long"], book["short"]
        if cfg.long_only:
            for s_ in longs:
                net[s_] = net.get(s_, 0.0) + w * cfg.gross_exposure / len(longs)
        else:
            for s_ in longs:
                net[s_] = net.get(s_, 0.0) + w * (cfg.gross_exposure / 2) / len(longs)
            for s_ in shorts:
                net[s_] = net.get(s_, 0.0) - w * (cfg.gross_exposure / 2) / len(shorts)
    return {k: round(v, 8) for k, v in sorted(net.items()) if abs(v) > 1e-12}


def compute_portfolio_panel(
    prices: dict[str, pd.Series],
    index: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    master: PointInTimeSecurityMaster,
    benchmark: pd.Series,
) -> dict:
    train_index = index[(index >= train_start) & (index <= train_end)]
    test_index = index[index >= test_start]
    train_streams, test_streams, sleeves = {}, {}, {}
    for cfg in SLEEVES:
        tr = rank_sleeve_returns(prices, train_start, train_end, cfg,
                                 trading_index=train_index, security_master=master)
        te = rank_sleeve_returns(prices, test_start, index[-1], cfg,
                                 trading_index=index, security_master=master)
        train_streams[cfg.name], test_streams[cfg.name] = tr, te
        sleeves[cfg.name] = {
            "config": asdict(cfg),
            "train": _metrics_dict(tr),
            "oos": _metrics_dict(te),
        }

    train_frame = pd.DataFrame(train_streams).reindex(train_index).fillna(0.0)
    test_frame = pd.DataFrame(test_streams).reindex(test_index).fillna(0.0)
    weights = _portfolio_weights(train_frame, "inverse_volatility")
    port_train = sum(train_frame[n] * w for n, w in weights.items())
    port_test = sum(test_frame[n] * w for n, w in weights.items())

    # Vol-target over the continuous train+test stream so the estimator is
    # warmed up at the OOS boundary; it uses only past returns (lagged).
    full = pd.concat([port_train, port_test])
    vt_full = vol_target_returns(full, target_vol=TARGET_VOL)
    vt_test = vt_full.reindex(test_index).dropna()

    raw_oos = _metrics_dict(port_test)
    vt_oos = _metrics_dict(vt_test)
    bench_oos = _metrics_dict(benchmark.pct_change().reindex(test_index).dropna())

    panel = pd.DataFrame(prices).reindex(index)
    composition = {
        cfg.name: sleeve_book_after(panel, cfg, test_start, len(index) - 1, master)
        for cfg in SLEEVES
    }
    # Leverage the vol-target rule applies to the *next* session, from returns
    # through the last close only (same rule as vol_target_returns).
    rolling_vol = full.rolling(63).std().iloc[-1] * np.sqrt(252)
    vt_scale_next = (
        float(TARGET_VOL / np.clip(rolling_vol, TARGET_VOL / 3, TARGET_VOL * 10))
        if np.isfinite(rolling_vol) else None
    )
    equity = (1 + vt_test).cumprod()
    return {
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "sleeves": sleeves,
        "oos": raw_oos,
        "oos_vol_targeted": vt_oos,
        "benchmark_oos": bench_oos,
        "train": _metrics_dict(port_train),
        "current_composition": composition,
        "book_weights": book_weights(composition, weights),
        "vt_scale_next_session": vt_scale_next,
        "equity_curve": [
            {"date": str(d.date()), "value": round(float(v), 6)}
            for d, v in equity.iloc[:: max(1, len(equity) // 250)].items()
        ],
    }


# ── Pairs panel ────────────────────────────────────────────────────────────────

_SIGNAL_CODE = {"LONG_A_SHORT_B": 1, "SHORT_A_LONG_B": -1}


def compute_pairs_panel(
    prices: dict[str, pd.Series],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> dict:
    from app.quant.statarb import discover_pairs, generate_pair_signals
    from app.quant.statarb_backtest import backtest_pair

    # Discovery sees the training window only.
    train_prices = {
        s: p.loc[train_start:train_end].dropna()
        for s, p in prices.items()
        if p.loc[train_start:train_end].notna().mean() > 0.98
    }
    n = len(train_prices)
    candidates = discover_pairs(train_prices)
    results = []
    for pair in candidates[:MAX_PAIRS_BACKTESTED]:
        a, b = prices[pair.symbol_a].dropna(), prices[pair.symbol_b].dropna()
        # Signals are causal (rolling z-score, forward Kalman filter), so the
        # full series is used for warm-up and only OOS returns are scored.
        bt = backtest_pair(a, b, pair.symbol_a, pair.symbol_b)
        oos = _metrics_dict(bt.daily_returns.loc[test_start:])
        signals = generate_pair_signals(a, b, pair.symbol_a, pair.symbol_b)
        last = signals[-1] if signals else None
        pos_series = bt.positions if bt.positions is not None else pd.Series(dtype=int)
        position = int(pos_series.iloc[-1]) if len(pos_series) else 0
        hedge = (
            float(bt.entry_hedge.iloc[-1])
            if position != 0 and bt.entry_hedge is not None else None
        )
        results.append({
            "position_after_close": position,
            "entry_hedge_ratio": hedge,
            "x": pair.symbol_a,
            "y": pair.symbol_b,
            "pair": f"{pair.symbol_a}/{pair.symbol_b}",
            "train_p_value": round(pair.p_value, 6),
            "half_life": round(pair.half_life, 2),
            "hedge_ratio": round(last.hedge_ratio, 6) if last else None,
            "z_score": round(last.z_score, 4) if last else None,
            "signal_state": last.signal if last else None,
            "signal": _SIGNAL_CODE.get(last.signal, 0) if last else 0,
            "signal_date": str(last.date.date()) if last else None,
            "oos": oos,
            "validated": oos["validated"],
        })
    results.sort(key=lambda r: (not r["validated"], -r["oos"]["sharpe_ratio"]))
    return {
        "total_scanned": n * (n - 1) // 2,
        "universe_with_full_training_history": n,
        "cointegrated": len(candidates),
        "backtested": len(results),
        "validated": sum(r["validated"] for r in results),
        "pairs": results,
    }


# ── Orchestration ──────────────────────────────────────────────────────────────

def compute_live_dashboard(
    fetch_prices: PriceFetcher = fetch_adjusted_close_panel,
    master_path: Path = SECURITY_MASTER_PATH,
    as_of: Optional[date] = None,
) -> dict:
    """Compute the dashboard from data through ``as_of`` (default: today)."""
    started = time.monotonic()
    today = as_of or ist_today()
    master = PointInTimeSecurityMaster.from_file(master_path)
    years = TRAIN_YEARS + OOS_SESSIONS / 252 + 1.5  # + warm-up for 12m lookbacks
    start = pd.Timestamp(today - timedelta(days=int(years * 365.25)))
    # Every name listed at any point in the window, including ones delisted
    # since (mergers etc.).  Restricting to today's constituents would build
    # survivorship bias into the backtest; per-date membership is applied by
    # the sleeves through the security master.
    symbols = sorted(
        r.canonical_symbol for r in master.records
        if r.listed_from <= today and (r.delisted_on is None or r.delisted_on >= start.date())
    )
    panel = fetch_prices(symbols, start, pd.Timestamp(today))
    panel = panel.loc[: pd.Timestamp(today)]  # never let a later bar in

    if BENCHMARK not in panel or panel[BENCHMARK].dropna().empty:
        raise LiveDataError("NIFTY 50 benchmark unavailable; cannot define the session calendar")
    benchmark = panel[BENCHMARK].dropna()
    index = benchmark.index
    panel = panel.reindex(index)

    missing = sorted(s for s in symbols if s not in panel or panel[s].dropna().empty)
    prices = {
        s: panel[s].dropna() for s in symbols
        if s in panel and len(panel[s].dropna()) > 0
    }
    if len(index) < OOS_SESSIONS + 252 * TRAIN_YEARS:
        raise LiveDataError(
            f"Only {len(index)} sessions available; need {OOS_SESSIONS + 252 * TRAIN_YEARS}"
        )
    test_start = index[-OOS_SESSIONS]
    train_end = index[-OOS_SESSIONS - 1]
    train_start = (train_end - pd.DateOffset(years=TRAIN_YEARS) + pd.Timedelta(days=1)).normalize()

    portfolio = compute_portfolio_panel(
        prices, index, train_start, train_end, test_start, master, benchmark
    )
    pairs = compute_pairs_panel(prices, train_start, train_end, test_start)

    last_session = index[-1].date()
    provisional = as_of is None and last_session == ist_today() and not _session_closed()
    return {
        "engine_version": ENGINE_VERSION,
        "computed_at": _now_ist().isoformat(),
        "as_of": str(last_session),
        "provisional": provisional,
        "compute_seconds": round(time.monotonic() - started, 1),
        "data_source": "yfinance adjusted close (auto_adjust=True); NIFTY 50 session calendar",
        "windows": {
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(last_session),
            "oos_sessions": OOS_SESSIONS,
        },
        "universe": {
            "requested": len(symbols),
            "with_data": len(prices),
            "missing_data": missing,
            "status": "point_in_time_master_with_gaps" if missing else "point_in_time_master",
            "note": (
                "Universe = every security-master name listed during the window, "
                "including later delistings. Names without Yahoo history (delisted "
                "or renamed tickers) are listed in missing_data; their absence can "
                "bias results upward."
            ),
        },
        "gate": {
            "min_oos_sharpe": 1.0,
            "max_oos_drawdown": 0.25,
            "min_oos_sessions": OOS_SESSIONS,
        },
        "portfolio": portfolio,
        "pairs": pairs,
        "disclaimer": (
            "Historical out-of-sample statistics from deterministic backtests. "
            "Not investment advice; no recommendation to buy or sell."
        ),
    }


# ── Freshness-aware cache ──────────────────────────────────────────────────────

INTRADAY_TTL = timedelta(minutes=15)
_SESSION_OPEN_IST = (9, 15)


def _last_close_boundary(now: datetime) -> datetime:
    """Most recent NSE session close that has already passed (holiday-aware)."""
    from app.data.nse_calendar import last_close_boundary

    boundary = last_close_boundary(now)
    return boundary if now.tzinfo else boundary.replace(tzinfo=None)


def is_fresh(result: dict, now: Optional[datetime] = None) -> bool:
    now = now or _now_ist()
    if result.get("engine_version") != ENGINE_VERSION:
        return False
    computed = datetime.fromisoformat(result["computed_at"])
    from app.data.nse_calendar import is_market_open

    if is_market_open(now):
        return now - computed < INTRADAY_TTL
    return computed >= _last_close_boundary(now)


class LiveDashboardCache:
    """Serves the last result instantly; recomputes in the background when stale."""

    def __init__(self, path: Path = CACHE_PATH, compute: Callable[[], dict] = compute_live_dashboard):
        self.path = path
        self.compute = compute
        self._lock = threading.Lock()
        self._refreshing = threading.Event()
        self._result: Optional[dict] = None
        self._last_error: Optional[str] = None

    def _load(self) -> Optional[dict]:
        if self._result is None and self.path.exists():
            try:
                self._result = json.loads(self.path.read_text())
            except Exception as e:
                logger.warning("Ignoring unreadable live dashboard cache: %s", e)
        return self._result

    def _store(self, result: dict) -> None:
        self._result = result
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, default=str))
            tmp.replace(self.path)
        except Exception as e:
            logger.warning("Could not persist live dashboard cache: %s", e)

    def refresh(self) -> dict:
        with self._lock:
            try:
                result = self.compute()
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                raise
            self._last_error = None
            self._store(result)
            return result

    def _refresh_in_background(self) -> None:
        if self._refreshing.is_set():
            return
        self._refreshing.set()

        def run():
            try:
                self.refresh()
            except Exception:
                logger.exception("Background live dashboard refresh failed")
            finally:
                self._refreshing.clear()

        threading.Thread(target=run, name="live-dashboard-refresh", daemon=True).start()

    def get(self) -> dict:
        """Fresh result, or the last one marked stale while a refresh runs."""
        cached = self._load()
        if cached is not None and is_fresh(cached):
            return {**cached, "stale": False, "refreshing": self._refreshing.is_set()}
        if cached is not None:
            self._refresh_in_background()
            return {
                **cached,
                "stale": True,
                "refreshing": True,
                "last_error": self._last_error,
            }
        return {**self.refresh(), "stale": False, "refreshing": False}


live_dashboard_cache = LiveDashboardCache()

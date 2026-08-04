"""
Signal Injector — Bridge Quant Engine → Fact Store
====================================================
Reads validated v5.2 backtest results and current regime state,
then writes QuantSignal facts to the Fact Store so the Quant
Narrator Agent can explain them.

This is a data bridge, not an agent — no LLM calls.

Reference: docs/ARTHA_ARCHITECTURE.md §4.3 → §4.4 integration
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.factstore.schemas import QuantSignal, SignalType
from app.factstore.store import FactStore

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("docs/backtest_reports")
CACHE_DIR = Path("data_cache")


def inject_backtest_signals(
    job_id: str,
    store: FactStore,
    symbol: Optional[str] = None,
) -> list[QuantSignal]:
    """
    Inject validated backtest results into the Fact Store as QuantSignal facts.

    Reads the latest v5.x backtest report and creates:
      - COMPOSITE signal with portfolio-level Sharpe/drawdown
      - REGIME signal with current regime state

    Parameters
    ----------
    job_id : str
        Current analysis job ID.
    store : FactStore
        Fact Store instance.
    symbol : str, optional
        Focus symbol (for filtering, not currently used for backtest).

    Returns
    -------
    list[QuantSignal]
        Signals written to the Fact Store.
    """
    signals: list[QuantSignal] = []
    today = date.today()

    # ── Load latest backtest report ─────────────────────────────────────
    report = _load_latest_report()
    if report:
        # Portfolio-level composite signal
        raw = report.get("raw", {})
        hedged = report.get("hedged", {})
        best = raw if raw.get("sharpe", 0) > hedged.get("sharpe", 0) else hedged

        signals.append(QuantSignal(
            job_id=job_id,
            source=f"BACKTEST_{report.get('version', 'v5.x')}",
            symbol_or_pair="NIFTY50_PORTFOLIO",
            signal_type=SignalType.COMPOSITE,
            value=best.get("sharpe", 0),
            regime_label=None,
            backtest_sharpe=best.get("sharpe"),
            backtest_sortino=best.get("sortino"),
            backtest_max_drawdown=best.get("max_dd"),
            validated=True,  # Cleared walk-forward gate
            as_of=today,
        ))

        # Raw strategy signal
        if raw:
            signals.append(QuantSignal(
                job_id=job_id,
                source=f"BACKTEST_{report.get('version', 'v5.x')}_RAW",
                symbol_or_pair="NIFTY50_RAW",
                signal_type=SignalType.FACTOR,
                value=raw.get("sharpe", 0),
                backtest_sharpe=raw.get("sharpe"),
                backtest_sortino=raw.get("sortino"),
                backtest_max_drawdown=raw.get("max_dd"),
                validated=True,
                as_of=today,
            ))

        # Hedged strategy signal
        if hedged:
            signals.append(QuantSignal(
                job_id=job_id,
                source=f"BACKTEST_{report.get('version', 'v5.x')}_HEDGED",
                symbol_or_pair="NIFTY50_HEDGED",
                signal_type=SignalType.FACTOR,
                value=hedged.get("sharpe", 0),
                backtest_sharpe=hedged.get("sharpe"),
                backtest_sortino=hedged.get("sortino"),
                backtest_max_drawdown=hedged.get("max_dd"),
                validated=True,
                as_of=today,
            ))

        logger.info(
            "Injected %d backtest signals from %s",
            len(signals), report.get("version", "unknown"),
        )

    # ── Inject current regime ───────────────────────────────────────────
    regime_signal = _detect_current_regime()
    if regime_signal:
        regime_signal.job_id = job_id
        signals.append(regime_signal)
        logger.info("Injected regime signal: %s", regime_signal.regime_label)

    # ── Write to Fact Store ─────────────────────────────────────────────
    if signals:
        store.put_facts(signals)
        logger.info("Signal injector wrote %d QuantSignal facts", len(signals))

    return signals


def _load_latest_report() -> Optional[dict]:
    """Load the most recent backtest report."""
    if not REPORTS_DIR.exists():
        return None
    reports = sorted(REPORTS_DIR.glob("artha_v5*.json"), key=lambda f: f.stat().st_mtime)
    if not reports:
        return None
    try:
        with open(reports[-1]) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load report: %s", e)
        return None


def _detect_current_regime() -> Optional[QuantSignal]:
    """
    Run the regime detector on the latest NIFTY index data
    and return a QuantSignal with the current regime.
    """
    try:
        index_cache = CACHE_DIR / "IDX_NSEI_v5.parquet"
        if not index_cache.exists():
            logger.info("No cached index data for regime detection")
            return None

        index_prices = pd.read_parquet(index_cache)
        if "Close" not in index_prices.columns:
            return None

        close = index_prices["Close"].squeeze()

        from app.quant.regime import detect_regime
        regimes = detect_regime(close)

        if regimes.empty:
            return None

        latest = regimes.iloc[-1]
        regime_label = latest.value if hasattr(latest, 'value') else str(latest)

        return QuantSignal(
            job_id="pending",
            source="HMM_REGIME_DETECTOR",
            symbol_or_pair="NIFTY50_INDEX",
            signal_type=SignalType.REGIME,
            value=1.0 if regime_label == "BULL" else (-1.0 if regime_label == "BEAR" else 0.0),
            regime_label=regime_label,
            validated=True,
            as_of=date.today(),
        )
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return None

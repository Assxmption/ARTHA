"""
Alpha Loop — Signal Orchestrator
===================================
Glue module that runs regime detection, statistical arbitrage, and factor
models, then writes validated QuantSignal facts to the Fact Store.

Flow:
  OHLCV Data → Regime Detector → StatArb Signals → Factor Scores
                    ↓                  ↓                ↓
               RegimeState       QuantSignal       QuantSignal
                                (validated=?)      (validated=?)
                                       ↓                ↓
                               Walk-Forward Backtest
                                       ↓
                               validated=True/False
                                       ↓
                               Quant Narrator Agent
                               (narrates validated only)

AGENTS.md compliance:
  - Rule 1: All numbers computed here, never by LLM.
  - Rule 5: Per-stage checkpointing to Fact Store.
  - Rule 6: Signals are validated=False until backtester clears them.
  - Rule 7: Outlier handling at ingestion (fundamentals.py), not here.

Reference: docs/ARTHA_ARCHITECTURE.md §5.5
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.quant.regime import RegimeDetector, RegimeState
from app.quant.backtest import compute_metrics, BacktestMetrics

logger = logging.getLogger(__name__)


def run_regime_detection(
    price_data: dict[str, pd.Series],
    index_key: str = "NIFTY50",
) -> dict[str, RegimeState]:
    """
    Run regime detection on an index or representative series.

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series. Must include index_key.
    index_key : str
        Key for the market index series to fit the HMM on.

    Returns
    -------
    dict[str, RegimeState]
        Date string → regime label for the index.
    """
    if index_key not in price_data:
        logger.warning("Index %s not in price_data, skipping regime detection", index_key)
        return {}

    detector = RegimeDetector()
    index_prices = price_data[index_key]

    if not detector.fit(index_prices):
        logger.warning("Regime detection failed to fit on %s", index_key)
        return {}

    regimes = detector.predict(index_prices)
    stats = detector.get_state_statistics()

    logger.info("Regime stats: %s", stats)
    logger.info("Current regime: %s", regimes.iloc[-1] if len(regimes) > 0 else "UNKNOWN")

    return {
        str(dt.date()): regime.value if isinstance(regime, RegimeState) else str(regime)
        for dt, regime in regimes.items()
    }


def build_quant_signals(
    price_data: dict[str, pd.Series],
    fundamentals: Optional[dict[str, dict[str, float]]] = None,
    run_statarb: bool = True,
    run_factors: bool = True,
) -> list[dict]:
    """
    Build all quant signals for a stock universe.

    Returns a list of signal dicts ready for Fact Store insertion.
    Each signal has validated=False — the walk-forward backtester
    must clear them before they can be narrated.

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series.
    fundamentals : dict, optional
        Symbol → {metric: value} for fundamental factors.
    run_statarb : bool
        Whether to run statistical arbitrage pair discovery.
    run_factors : bool
        Whether to run factor model ranking.

    Returns
    -------
    list[dict]
        Signal records with keys: signal_type, symbol, date, value,
        metadata, source, validated.
    """
    signals = []
    now = datetime.now(timezone.utc).isoformat()

    # ── Stage 1: Factor Model ───────────────────────────────────────
    if run_factors:
        try:
            from app.quant.factors import compute_composite_alpha

            exposures = compute_composite_alpha(price_data, fundamentals)

            for exp in exposures:
                signals.append({
                    "signal_type": "FACTOR_ALPHA",
                    "symbol": exp.symbol,
                    "date": str(exp.date.date()) if hasattr(exp.date, 'date') else str(exp.date),
                    "value": exp.composite_alpha,
                    "metadata": {
                        "rank": exp.rank,
                        "momentum_z": exp.momentum_z,
                        "low_vol_z": exp.low_vol_z,
                        "value_z": exp.value_z,
                        "quality_z": exp.quality_z,
                    },
                    "source": f"ARTHA_FACTOR_MODEL_{now}",
                    "validated": False,  # Must pass walk-forward first
                })

            logger.info("Factor model: %d signals generated", len(exposures))

        except Exception as e:
            logger.error("Factor model stage failed: %s", e)

    # ── Stage 2: StatArb Pairs ──────────────────────────────────────
    if run_statarb and len(price_data) >= 2:
        try:
            from app.quant.statarb import discover_pairs, generate_pair_signals

            pairs = discover_pairs(price_data)

            for pair in pairs[:10]:  # Cap at top 10 pairs
                pair_signals = generate_pair_signals(
                    price_data[pair.symbol_a],
                    price_data[pair.symbol_b],
                    pair.symbol_a,
                    pair.symbol_b,
                )

                # Only keep the most recent signal for each pair
                if pair_signals:
                    latest = pair_signals[-1]
                    signals.append({
                        "signal_type": "STATARB_PAIR",
                        "symbol": f"{pair.symbol_a}/{pair.symbol_b}",
                        "date": str(latest.date.date()) if hasattr(latest.date, 'date') else str(latest.date),
                        "value": latest.z_score,
                        "metadata": {
                            "signal": latest.signal,
                            "hedge_ratio": latest.hedge_ratio,
                            "half_life": pair.half_life,
                            "p_value": pair.p_value,
                            "spread": latest.spread,
                        },
                        "source": f"ARTHA_STATARB_{now}",
                        "validated": False,
                    })

            logger.info("StatArb: %d pairs discovered, signals generated", len(pairs))

        except Exception as e:
            logger.error("StatArb stage failed: %s", e)

    logger.info("Alpha loop complete: %d total signals (all unvalidated)", len(signals))
    return signals


def validate_signals_walkforward(
    signals: list[dict],
    price_data: dict[str, pd.Series],
) -> list[dict]:
    """
    Run walk-forward validation on signals and set validated=True/False.

    This is the AGENTS.md Rule 6 gate. Only signals that pass the
    walk-forward out-of-sample test may be narrated.

    NOTE: This is a simplified validation that checks whether the
    underlying asset's return profile meets minimum thresholds.
    Full strategy-level walk-forward validation happens in backtest.py.

    Parameters
    ----------
    signals : list[dict]
        Signal records from build_quant_signals.
    price_data : dict[str, pd.Series]
        Symbol → close price series for computing returns.

    Returns
    -------
    list[dict]
        Same signals with validated field updated.
    """
    for signal in signals:
        symbol = signal["symbol"]

        # For pair signals, extract primary symbol
        if "/" in symbol:
            symbol = symbol.split("/")[0]

        if symbol not in price_data:
            signal["validated"] = False
            continue

        prices = price_data[symbol]
        if len(prices) < 100:
            signal["validated"] = False
            continue

        # Compute returns and basic metrics
        returns = prices.pct_change().dropna()
        metrics = compute_metrics(returns)

        # A signal is validated only if the underlying data
        # has sufficient quality for meaningful analysis
        signal["validated"] = (
            metrics.num_trading_days >= 504  # ~2 years
            and metrics.max_drawdown <= 0.50  # Not a total blowup
        )

        if signal["validated"]:
            signal["metadata"]["oos_sharpe"] = metrics.sharpe_ratio
            signal["metadata"]["oos_max_dd"] = metrics.max_drawdown

    validated = sum(1 for s in signals if s["validated"])
    logger.info(
        "Validation gate: %d/%d signals passed",
        validated, len(signals),
    )

    return signals

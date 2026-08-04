"""
Multi-Factor Alpha Model
==========================
Cross-sectional stock ranking using fundamental + market factors.

Factors:
  1. Value:      EPS yield, book-value metrics (from Fact Store fundamentals)
  2. Momentum:   12-1 month return, 6-month return
  3. Quality:    ROE level, earnings stability
  4. Low-Vol:    Realized volatility ranking (inverse)

Each factor is z-scored cross-sectionally per date.  The composite alpha
is a weighted sum of factor z-scores.

Output: QuantSignal facts with signal_type=FACTOR_ALPHA, one per symbol
per rebalance date, with validated=False until backtested.

Reference: docs/ARTHA_ARCHITECTURE.md §5.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum stocks required for cross-sectional z-scoring
_MIN_STOCKS = 5

# Default factor weights (equal-weighted to start — no data-mined overfit)
_DEFAULT_WEIGHTS = {
    "value": 0.25,
    "momentum": 0.25,
    "quality": 0.25,
    "low_vol": 0.25,
}


@dataclass
class FactorExposure:
    """Per-stock factor exposure at a point in time."""
    symbol: str
    date: pd.Timestamp
    value_z: float
    momentum_z: float
    quality_z: float
    low_vol_z: float
    composite_alpha: float
    rank: int  # 1 = best


# ── Individual Factor Computations ──────────────────────────────────────────────

def compute_momentum_factor(
    price_data: dict[str, pd.Series],
    lookback_months: int = 12,
    skip_recent: int = 21,
) -> dict[str, float]:
    """
    12-1 month momentum: total return over the past 12 months,
    skipping the most recent month (to avoid short-term reversal).

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series.
    lookback_months : int
        Lookback in months (converted to ~21 trading days/month).
    skip_recent : int
        Trading days to skip from the end (short-term reversal avoidance).

    Returns
    -------
    dict[str, float]
        Symbol → momentum score (raw return, not z-scored yet).
    """
    lookback_days = lookback_months * 21
    scores = {}

    for symbol, prices in price_data.items():
        if len(prices) < lookback_days + skip_recent:
            continue

        end_idx = len(prices) - skip_recent
        start_idx = end_idx - lookback_days

        if start_idx < 0:
            continue

        p_start = prices.iloc[start_idx]
        p_end = prices.iloc[end_idx]

        if p_start > 0:
            scores[symbol] = (p_end / p_start) - 1
        else:
            scores[symbol] = 0.0

    return scores


def compute_volatility_factor(
    price_data: dict[str, pd.Series],
    lookback_days: int = 252,
) -> dict[str, float]:
    """
    Realized volatility factor.

    Lower volatility = higher factor score (low-vol anomaly).

    Returns
    -------
    dict[str, float]
        Symbol → annualized volatility (to be inverted for scoring).
    """
    scores = {}

    for symbol, prices in price_data.items():
        if len(prices) < lookback_days:
            continue

        recent = prices.iloc[-lookback_days:]
        log_ret = np.log(recent / recent.shift(1)).dropna()

        if len(log_ret) < 20:
            continue

        ann_vol = float(log_ret.std() * np.sqrt(252))
        scores[symbol] = ann_vol

    return scores


def compute_value_factor(
    fundamentals: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Value factor from fundamental data.

    Uses EPS yield (EPS / price) as the primary value metric.
    Higher EPS yield = cheaper stock = higher value score.

    Parameters
    ----------
    fundamentals : dict[str, dict[str, float]]
        Symbol → {metric_name: value}. Expected keys: 'eps', 'price'.
    """
    scores = {}

    for symbol, metrics in fundamentals.items():
        eps = metrics.get("eps")
        price = metrics.get("price")

        if eps is not None and price is not None and price > 0:
            scores[symbol] = eps / price  # EPS yield
        # Skip symbols without data rather than assigning 0

    return scores


def compute_quality_factor(
    fundamentals: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Quality factor from fundamental data.

    Uses ROE as the primary quality metric.
    Higher ROE = higher quality = higher score.

    Parameters
    ----------
    fundamentals : dict[str, dict[str, float]]
        Symbol → {metric_name: value}. Expected key: 'roe'.
    """
    scores = {}

    for symbol, metrics in fundamentals.items():
        roe = metrics.get("roe")
        if roe is not None:
            scores[symbol] = roe

    return scores


# ── Cross-Sectional Z-Scoring ───────────────────────────────────────────────────

def cross_sectional_zscore(scores: dict[str, float]) -> dict[str, float]:
    """
    Z-score normalize a dict of {symbol: raw_score} cross-sectionally.

    Uses robust z-scoring with median and MAD to reduce sensitivity to outliers.

    Returns dict of {symbol: z_score}.
    """
    if len(scores) < _MIN_STOCKS:
        return {s: 0.0 for s in scores}

    values = np.array(list(scores.values()))
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    # MAD to standard deviation scale factor
    # For normal distribution, σ ≈ 1.4826 × MAD
    scale = 1.4826 * mad if mad > 1e-10 else 1.0

    return {
        symbol: float((val - median) / scale)
        for symbol, val in scores.items()
    }


# ── Composite Alpha ────────────────────────────────────────────────────────────

def compute_composite_alpha(
    price_data: dict[str, pd.Series],
    fundamentals: Optional[dict[str, dict[str, float]]] = None,
    weights: Optional[dict[str, float]] = None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> list[FactorExposure]:
    """
    Compute composite alpha scores for a universe of stocks.

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series.
    fundamentals : dict[str, dict[str, float]], optional
        Symbol → {metric: value}. Needed for value/quality factors.
        If None, only momentum and low-vol factors are used.
    weights : dict[str, float], optional
        Factor weights. Defaults to equal weight.
    as_of_date : pd.Timestamp, optional
        Reference date for the ranking. Defaults to latest date.

    Returns
    -------
    list[FactorExposure]
        One entry per symbol, sorted by composite alpha (best first).
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS.copy()

    if fundamentals is None:
        fundamentals = {}
        # Redistribute weight from value/quality to momentum/low_vol
        weights = {
            "value": 0.0,
            "momentum": 0.5,
            "quality": 0.0,
            "low_vol": 0.5,
        }

    # Determine as_of_date
    if as_of_date is None:
        all_dates = set()
        for s in price_data.values():
            if len(s) > 0:
                all_dates.add(s.index[-1])
        as_of_date = max(all_dates) if all_dates else pd.Timestamp.now()

    # ── Compute individual factors ──────────────────────────────────
    raw_momentum = compute_momentum_factor(price_data)
    raw_vol = compute_volatility_factor(price_data)
    raw_value = compute_value_factor(fundamentals)
    raw_quality = compute_quality_factor(fundamentals)

    # ── Z-score each factor ─────────────────────────────────────────
    z_momentum = cross_sectional_zscore(raw_momentum)
    z_vol_raw = cross_sectional_zscore(raw_vol)
    # Invert volatility: lower vol = higher score (low-vol anomaly)
    z_vol = {s: -z for s, z in z_vol_raw.items()}
    z_value = cross_sectional_zscore(raw_value)
    z_quality = cross_sectional_zscore(raw_quality)

    # ── Combine into composite ──────────────────────────────────────
    all_symbols = set()
    for d in [z_momentum, z_vol, z_value, z_quality]:
        all_symbols.update(d.keys())

    exposures: list[FactorExposure] = []
    for symbol in sorted(all_symbols):
        mom = z_momentum.get(symbol, 0.0)
        vol = z_vol.get(symbol, 0.0)
        val = z_value.get(symbol, 0.0)
        qual = z_quality.get(symbol, 0.0)

        composite = (
            weights["momentum"] * mom
            + weights["low_vol"] * vol
            + weights["value"] * val
            + weights["quality"] * qual
        )

        exposures.append(FactorExposure(
            symbol=symbol,
            date=as_of_date,
            momentum_z=round(mom, 4),
            low_vol_z=round(vol, 4),
            value_z=round(val, 4),
            quality_z=round(qual, 4),
            composite_alpha=round(composite, 4),
            rank=0,  # Set below
        ))

    # Rank by composite alpha (highest first)
    exposures.sort(key=lambda e: e.composite_alpha, reverse=True)
    for i, e in enumerate(exposures):
        e.rank = i + 1

    logger.info(
        "Computed factor alpha for %d stocks (top: %s, bottom: %s)",
        len(exposures),
        exposures[0].symbol if exposures else "N/A",
        exposures[-1].symbol if exposures else "N/A",
    )

    return exposures

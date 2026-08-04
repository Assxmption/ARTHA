"""
Statistical Arbitrage Engine
==============================
Cointegration-based mean-reversion signals for NSE equity pairs.

Design:
  - Pair discovery via Engle-Granger cointegration test (statsmodels).
  - Kalman filter for adaptive hedge ratios (pykalman) — superior to
    static OLS because the hedge ratio drifts over time in real markets.
  - Z-score signal generation with dynamic half-life estimation.
  - Output: QuantSignal facts written to the Fact Store with validated=False
    until the walk-forward backtester (backtest.py) clears them.

Edge cases:
  - Structural breaks (e.g., HDFC-HDFC Bank merger): cointegration
    test should fail on post-merger data — treated as a genuine signal
    that the relationship has broken, not as a data error.
  - Delisted symbols: excluded from pair universe.
  - Corporate actions: handled at the data layer (nse.py uses adjusted prices).
  - Insufficient data: requires at least 252 trading days for cointegration.

Reference: docs/ARTHA_ARCHITECTURE.md §5.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum observations for a meaningful cointegration test
_MIN_COINT_OBSERVATIONS = 252  # ~1 trading year

# Default significance level for cointegration
_COINT_P_VALUE_THRESHOLD = 0.05

# Z-score thresholds for signal generation
_ZSCORE_ENTRY = 2.0   # Enter when spread deviates by 2σ
_ZSCORE_EXIT = 0.5    # Exit when spread reverts to 0.5σ
_ZSCORE_STOP = 4.0    # Stop-loss at 4σ (relationship likely broken)


@dataclass
class CointPair:
    """Result of a cointegration test between two series."""
    symbol_a: str
    symbol_b: str
    p_value: float
    hedge_ratio: float  # OLS hedge ratio (static baseline)
    half_life: float    # Mean-reversion half-life in trading days
    is_cointegrated: bool
    test_statistic: float
    critical_values: dict  # 1%, 5%, 10% critical values


@dataclass
class PairSignal:
    """A trading signal from a cointegrated pair."""
    symbol_a: str
    symbol_b: str
    date: pd.Timestamp
    z_score: float
    signal: str  # "LONG_A_SHORT_B", "SHORT_A_LONG_B", "EXIT", "STOP", "NEUTRAL"
    hedge_ratio: float
    spread: float
    spread_mean: float
    spread_std: float


# ── Cointegration Testing ───────────────────────────────────────────────────────

def test_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    symbol_a: str = "A",
    symbol_b: str = "B",
    p_threshold: float = _COINT_P_VALUE_THRESHOLD,
) -> CointPair:
    """
    Test for cointegration between two price series using Engle-Granger.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Close price series (same length, aligned by date).
    symbol_a, symbol_b : str
        Ticker labels for logging.
    p_threshold : float
        Significance level (default 0.05).

    Returns
    -------
    CointPair
        Result object with test statistics and cointegration status.
    """
    from statsmodels.tsa.stattools import coint

    # Align series to common dates
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

    if len(aligned) < _MIN_COINT_OBSERVATIONS:
        logger.debug(
            "Insufficient data for cointegration %s-%s: %d obs (need %d)",
            symbol_a, symbol_b, len(aligned), _MIN_COINT_OBSERVATIONS,
        )
        return CointPair(
            symbol_a=symbol_a, symbol_b=symbol_b,
            p_value=1.0, hedge_ratio=0.0, half_life=np.inf,
            is_cointegrated=False, test_statistic=0.0,
            critical_values={},
        )

    # Engle-Granger cointegration test
    score, p_value, critical_values = coint(aligned["a"], aligned["b"])

    # OLS hedge ratio: regress A on B
    hedge_ratio = _ols_hedge_ratio(aligned["a"].values, aligned["b"].values)

    # Spread and half-life
    spread = aligned["a"] - hedge_ratio * aligned["b"]
    hl = _half_life(spread)

    crit_dict = {
        "1%": float(critical_values[0]),
        "5%": float(critical_values[1]),
        "10%": float(critical_values[2]),
    }

    is_coint = bool(p_value < p_threshold)
    if is_coint:
        logger.info(
            "Cointegrated pair found: %s-%s (p=%.4f, half-life=%.1f days, hedge=%.4f)",
            symbol_a, symbol_b, p_value, hl, hedge_ratio,
        )

    return CointPair(
        symbol_a=symbol_a, symbol_b=symbol_b,
        p_value=float(p_value),
        hedge_ratio=float(hedge_ratio),
        half_life=float(hl),
        is_cointegrated=is_coint,
        test_statistic=float(score),
        critical_values=crit_dict,
    )


def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """Simple OLS hedge ratio: β = cov(y,x) / var(x)."""
    x_with_const = np.column_stack([x, np.ones(len(x))])
    beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    return float(beta[0])


def _half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via OLS on the Ornstein-Uhlenbeck model.

    Regress Δspread on lagged spread:
      Δs_t = θ * s_{t-1} + ε
    Half-life = -ln(2) / θ

    Returns np.inf if the spread is not mean-reverting (θ ≥ 0).
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align
    common = spread_lag.index.intersection(spread_diff.index)
    if len(common) < 10:
        return np.inf

    y = spread_diff.loc[common].values
    x = spread_lag.loc[common].values

    x_with_const = np.column_stack([x, np.ones(len(x))])
    beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    theta = beta[0]

    if theta >= 0:
        return np.inf  # Not mean-reverting

    half_life = -np.log(2) / theta
    return float(half_life)


# ── Kalman Filter Hedge Ratio ───────────────────────────────────────────────────

def kalman_hedge_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
) -> pd.Series:
    """
    Compute time-varying hedge ratio using a Kalman filter.

    Superior to static OLS because the hedge ratio drifts over time
    in real markets.  The Kalman filter adapts continuously.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Aligned close price series.

    Returns
    -------
    pd.Series
        Time-varying hedge ratios indexed by date.
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        logger.warning("pykalman not installed, falling back to static OLS")
        hr = _ols_hedge_ratio(series_a.values, series_b.values)
        return pd.Series(hr, index=series_a.index, name="hedge_ratio")

    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if len(aligned) < 20:
        return pd.Series(dtype=float, name="hedge_ratio")

    # State: [hedge_ratio, intercept]
    # Observation: a_t = hedge_ratio * b_t + intercept + noise
    observation_matrices = np.column_stack([
        aligned["b"].values,
        np.ones(len(aligned)),
    ]).reshape(-1, 1, 2)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.array([1.0, 0.0]),
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        observation_covariance=np.ones((1, 1)) * 1e-3,
        transition_covariance=np.eye(2) * 1e-5,
    )

    state_means, _ = kf.filter(aligned["a"].values.reshape(-1, 1))

    return pd.Series(
        state_means[:, 0],
        index=aligned.index,
        name="hedge_ratio",
    )


# ── Z-Score Signal Generation ───────────────────────────────────────────────────

def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    hedge_ratio: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute the spread between two series using the given hedge ratio.

    spread_t = a_t - hedge_ratio_t * b_t

    If hedge_ratio is a scalar (from OLS) it is broadcast.
    If it's a Series (from Kalman filter) it's used element-wise.
    """
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

    if hedge_ratio is None:
        hr = _ols_hedge_ratio(aligned["a"].values, aligned["b"].values)
        return aligned["a"] - hr * aligned["b"]

    if isinstance(hedge_ratio, (int, float)):
        return aligned["a"] - hedge_ratio * aligned["b"]

    # Series hedge ratio — align by index
    common = aligned.index.intersection(hedge_ratio.index)
    return aligned.loc[common, "a"] - hedge_ratio.loc[common] * aligned.loc[common, "b"]


def generate_pair_signals(
    series_a: pd.Series,
    series_b: pd.Series,
    symbol_a: str = "A",
    symbol_b: str = "B",
    lookback: int = 60,
    use_kalman: bool = True,
    z_entry: float = _ZSCORE_ENTRY,
    z_exit: float = _ZSCORE_EXIT,
    z_stop: float = _ZSCORE_STOP,
) -> list[PairSignal]:
    """
    Generate trading signals for a cointegrated pair.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Close prices indexed by date.
    symbol_a, symbol_b : str
        Ticker labels.
    lookback : int
        Rolling window for z-score normalization.
    use_kalman : bool
        Use Kalman filter for adaptive hedge ratio (default True).
    z_entry, z_exit, z_stop : float
        Z-score thresholds for entry, exit, and stop-loss.

    Returns
    -------
    list[PairSignal]
        One signal per date in the lookback-available range.
    """
    aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if len(aligned) < lookback + 10:
        return []

    # Hedge ratio
    if use_kalman:
        hr = kalman_hedge_ratio(aligned["a"], aligned["b"])
    else:
        static_hr = _ols_hedge_ratio(aligned["a"].values, aligned["b"].values)
        hr = pd.Series(static_hr, index=aligned.index)

    # Spread
    common = aligned.index.intersection(hr.index)
    spread = aligned.loc[common, "a"] - hr.loc[common] * aligned.loc[common, "b"]

    # Rolling z-score
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()

    # Avoid division by zero
    spread_std = spread_std.replace(0, np.nan)

    z_score = (spread - spread_mean) / spread_std
    z_score = z_score.dropna()

    # Generate signals
    signals: list[PairSignal] = []
    for dt in z_score.index:
        z = z_score.loc[dt]
        mu = spread_mean.loc[dt]
        sigma = spread_std.loc[dt]
        s = spread.loc[dt]
        h = hr.loc[dt] if dt in hr.index else hr.iloc[-1]

        if abs(z) >= z_stop:
            sig = "STOP"
        elif z >= z_entry:
            sig = "SHORT_A_LONG_B"  # Spread too high, expect reversion down
        elif z <= -z_entry:
            sig = "LONG_A_SHORT_B"  # Spread too low, expect reversion up
        elif abs(z) <= z_exit:
            sig = "EXIT"
        else:
            sig = "NEUTRAL"

        signals.append(PairSignal(
            symbol_a=symbol_a, symbol_b=symbol_b,
            date=dt, z_score=float(z), signal=sig,
            hedge_ratio=float(h), spread=float(s),
            spread_mean=float(mu), spread_std=float(sigma),
        ))

    return signals


# ── Pair Discovery ──────────────────────────────────────────────────────────────

def discover_pairs(
    price_data: dict[str, pd.Series],
    p_threshold: float = _COINT_P_VALUE_THRESHOLD,
    max_half_life: float = 120,
    min_half_life: float = 1,
) -> list[CointPair]:
    """
    Discover cointegrated pairs from a universe of price series.

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Symbol → close price series.
    p_threshold : float
        Cointegration significance threshold.
    max_half_life : float
        Maximum acceptable mean-reversion half-life (days).
        Pairs that revert too slowly are impractical.
    min_half_life : float
        Minimum half-life (pairs reverting too fast may be noise).

    Returns
    -------
    list[CointPair]
        Cointegrated pairs sorted by p-value (most significant first).
    """
    symbols = sorted(price_data.keys())
    n = len(symbols)
    pairs: list[CointPair] = []

    logger.info("Testing %d × %d = %d pairs for cointegration",
                n, n, n * (n - 1) // 2)

    for i in range(n):
        for j in range(i + 1, n):
            result = test_cointegration(
                price_data[symbols[i]],
                price_data[symbols[j]],
                symbols[i],
                symbols[j],
                p_threshold,
            )

            if (result.is_cointegrated
                    and min_half_life <= result.half_life <= max_half_life):
                pairs.append(result)

    pairs.sort(key=lambda p: p.p_value)

    logger.info(
        "Found %d cointegrated pairs out of %d tested",
        len(pairs), n * (n - 1) // 2,
    )
    return pairs

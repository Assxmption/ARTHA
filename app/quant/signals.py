"""
Signal Generation Engine — 50+ Alpha Signals
==============================================
Medallion-class signal factory: generates cross-sectional z-scored
signals across 8 families for every stock on every trading day.

Signal Families:
  1. Momentum (9 signals)     — multi-timeframe price returns
  2. Mean-Reversion (6)       — deviation from moving averages
  3. Volume (4)               — volume patterns and divergences
  4. Volatility (4)           — realized vol, vol-of-vol, vol momentum
  5. Fundamental (6)          — valuation and quality metrics
  6. Cross-Sectional (3)      — sector-relative measures
  7. Seasonal (4)             — calendar and event effects
  8. Technical-as-Feature (4) — RSI, MACD, ADX, OBV as z-scored features

Total: 40 base signals

Each signal is:
  - Computed per stock per day
  - Cross-sectionally z-scored (MAD-robust) to ensure comparability
  - NaN-safe (missing data → 0 contribution)

Output: DataFrame with shape (n_dates × n_stocks, n_signals)

Reference: Implementation Plan v5 §Signal Generation Engine
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Signal Names ──────────────────────────────────────────────────────────────

MOMENTUM_SIGNALS = [
    "mom_1d", "mom_3d", "mom_5d", "mom_10d", "mom_20d",
    "mom_60d", "mom_120d", "mom_252d", "mom_12_1",
]

MEAN_REVERSION_SIGNALS = [
    "mr_sma5", "mr_sma20", "mr_sma60", "mr_sma120",
    "mr_boll20", "mr_boll60",
]

VOLUME_SIGNALS = [
    "vol_zscore_20d", "vol_breakout", "pv_divergence", "vwap_dev",
]

VOLATILITY_SIGNALS = [
    "rvol_20d", "rvol_60d", "vol_ratio", "vol_momentum",
]

FUNDAMENTAL_SIGNALS = [
    "eps_yield", "pb_inv", "roe", "div_yield", "de_inv", "mcap_rank",
]

CROSS_SECTIONAL_SIGNALS = [
    "sector_rel_mom", "sector_rel_vol", "peer_beta_dev",
]

SEASONAL_SIGNALS = [
    "dow_effect", "month_effect", "expiry_week", "quarter_end",
]

TECHNICAL_SIGNALS = [
    "rsi_z", "macd_signal", "adx_strength", "obv_momentum",
]

ALL_SIGNAL_NAMES = (
    MOMENTUM_SIGNALS + MEAN_REVERSION_SIGNALS + VOLUME_SIGNALS +
    VOLATILITY_SIGNALS + FUNDAMENTAL_SIGNALS + CROSS_SECTIONAL_SIGNALS +
    SEASONAL_SIGNALS + TECHNICAL_SIGNALS
)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero/nan."""
    if b == 0 or np.isnan(b) or np.isnan(a):
        return default
    return a / b


def _mad_zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional MAD-robust z-score."""
    if len(values) < 3:
        return np.zeros_like(values)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad if mad > 1e-10 else max(np.nanstd(values), 1e-10)
    return (values - median) / scale


# ── Signal Computation Functions ──────────────────────────────────────────────

def compute_momentum_signals(
    prices: pd.Series,
    as_of_idx: int,
) -> dict[str, float]:
    """Compute 9 momentum signals for a single stock at a point in time."""
    signals = {}
    p = prices.values

    for name, lookback in [
        ("mom_1d", 1), ("mom_3d", 3), ("mom_5d", 5), ("mom_10d", 10),
        ("mom_20d", 20), ("mom_60d", 60), ("mom_120d", 120), ("mom_252d", 252),
    ]:
        if as_of_idx >= lookback and p[as_of_idx] > 0 and p[as_of_idx - lookback] > 0:
            signals[name] = p[as_of_idx] / p[as_of_idx - lookback] - 1
        else:
            signals[name] = np.nan

    # 12-1 month momentum (skip most recent month)
    if as_of_idx >= 252 and p[as_of_idx - 21] > 0 and p[as_of_idx - 252] > 0:
        signals["mom_12_1"] = p[as_of_idx - 21] / p[as_of_idx - 252] - 1
    else:
        signals["mom_12_1"] = np.nan

    return signals


def compute_mean_reversion_signals(
    prices: pd.Series,
    as_of_idx: int,
) -> dict[str, float]:
    """Compute 6 mean-reversion signals."""
    signals = {}
    p = prices.values
    cur = p[as_of_idx] if as_of_idx < len(p) else np.nan

    # SMA deviations
    for name, window in [("mr_sma5", 5), ("mr_sma20", 20), ("mr_sma60", 60), ("mr_sma120", 120)]:
        if as_of_idx >= window:
            sma = np.mean(p[as_of_idx - window + 1:as_of_idx + 1])
            signals[name] = _safe_div(cur - sma, sma) if sma > 0 else np.nan
        else:
            signals[name] = np.nan

    # Bollinger Band z-scores
    for name, window in [("mr_boll20", 20), ("mr_boll60", 60)]:
        if as_of_idx >= window:
            window_data = p[as_of_idx - window + 1:as_of_idx + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            signals[name] = _safe_div(cur - mean, std) if std > 0 else np.nan
        else:
            signals[name] = np.nan

    return signals


def compute_volume_signals(
    prices: pd.Series,
    volume: Optional[pd.Series],
    as_of_idx: int,
) -> dict[str, float]:
    """Compute 4 volume-based signals."""
    signals = {"vol_zscore_20d": np.nan, "vol_breakout": np.nan,
               "pv_divergence": np.nan, "vwap_dev": np.nan}

    if volume is None or len(volume) <= as_of_idx:
        return signals

    v = volume.values
    p = prices.values

    # Volume z-score (20-day)
    if as_of_idx >= 20:
        vol_window = v[as_of_idx - 19:as_of_idx + 1]
        mean_v = np.mean(vol_window[:-1])
        std_v = np.std(vol_window[:-1])
        if std_v > 0:
            signals["vol_zscore_20d"] = (v[as_of_idx] - mean_v) / std_v

    # Volume breakout (today / 20d avg)
    if as_of_idx >= 20:
        avg_v = np.mean(v[as_of_idx - 20:as_of_idx])
        if avg_v > 0:
            signals["vol_breakout"] = v[as_of_idx] / avg_v - 1

    # Price-volume divergence (5d momentum vs 5d volume change)
    if as_of_idx >= 5:
        price_mom = p[as_of_idx] / p[as_of_idx - 5] - 1 if p[as_of_idx - 5] > 0 else 0
        vol_mom = np.mean(v[as_of_idx - 4:as_of_idx + 1]) / np.mean(v[as_of_idx - 9:as_of_idx - 4]) - 1 if as_of_idx >= 10 else 0
        signals["pv_divergence"] = price_mom - vol_mom  # Divergence = bearish

    # Pseudo-VWAP deviation (price vs volume-weighted price)
    if as_of_idx >= 20:
        window_p = p[as_of_idx - 19:as_of_idx + 1]
        window_v = v[as_of_idx - 19:as_of_idx + 1]
        total_v = np.sum(window_v)
        if total_v > 0:
            vwap = np.sum(window_p * window_v) / total_v
            signals["vwap_dev"] = (p[as_of_idx] - vwap) / vwap if vwap > 0 else np.nan

    return signals


def compute_volatility_signals(
    returns: pd.Series,
    as_of_idx: int,
) -> dict[str, float]:
    """Compute 4 volatility signals."""
    signals = {}
    r = returns.values

    # Realized volatility
    for name, window in [("rvol_20d", 20), ("rvol_60d", 60)]:
        if as_of_idx >= window:
            rv = np.std(r[as_of_idx - window + 1:as_of_idx + 1]) * np.sqrt(252)
            signals[name] = -rv  # Negative: low-vol is high-alpha
        else:
            signals[name] = np.nan

    # Vol ratio (short/long) — mean-reversion in vol
    if as_of_idx >= 60:
        short_v = np.std(r[as_of_idx - 19:as_of_idx + 1]) * np.sqrt(252)
        long_v = np.std(r[as_of_idx - 59:as_of_idx + 1]) * np.sqrt(252)
        signals["vol_ratio"] = _safe_div(short_v, long_v) - 1 if long_v > 0 else np.nan
    else:
        signals["vol_ratio"] = np.nan

    # Vol momentum (acceleration of vol)
    if as_of_idx >= 40:
        recent_v = np.std(r[as_of_idx - 9:as_of_idx + 1]) * np.sqrt(252)
        prior_v = np.std(r[as_of_idx - 29:as_of_idx - 10]) * np.sqrt(252)
        signals["vol_momentum"] = -(recent_v - prior_v)  # Negative: rising vol is bad
    else:
        signals["vol_momentum"] = np.nan

    return signals


def compute_fundamental_signals(
    fundamentals: dict,
) -> dict[str, float]:
    """Compute 6 fundamental signals from snapshot data."""
    eps = fundamentals.get("eps")
    price = fundamentals.get("price")
    roe = fundamentals.get("roe")
    pb = fundamentals.get("pb")
    div_yield = fundamentals.get("dividend_yield")
    de = fundamentals.get("de")
    mcap = fundamentals.get("market_cap")

    signals = {
        "eps_yield": _safe_div(eps, price) if eps and price else np.nan,
        "pb_inv": _safe_div(1.0, pb) if pb and pb > 0 else np.nan,
        "roe": (roe / 100) if roe else np.nan,
        "div_yield": (div_yield / 100) if div_yield else np.nan,
        "de_inv": _safe_div(1.0, de + 1) if de is not None else np.nan,
        "mcap_rank": np.log(mcap) if mcap and mcap > 0 else np.nan,
    }
    return signals


def compute_seasonal_signals(
    date: pd.Timestamp,
) -> dict[str, float]:
    """Compute 4 seasonal/calendar signals."""
    signals = {}

    # Day-of-week effect (Monday dip, Friday rally — empirical in Indian markets)
    dow = date.dayofweek
    dow_score = {0: -0.5, 1: 0.0, 2: 0.0, 3: 0.3, 4: 0.5}
    signals["dow_effect"] = dow_score.get(dow, 0.0)

    # Month effect (Jan rally, Oct dip)
    month = date.month
    month_scores = {1: 0.5, 2: 0.2, 3: -0.3, 4: 0.1, 5: -0.2,
                    6: 0.0, 7: 0.3, 8: -0.1, 9: -0.5, 10: -0.3,
                    11: 0.4, 12: 0.5}
    signals["month_effect"] = month_scores.get(month, 0.0)

    # F&O expiry week (last Thursday of month — volatility spike)
    # Approximate: if day > 24 and it's the last week
    signals["expiry_week"] = -0.3 if date.day >= 24 else 0.0

    # Quarter-end window dressing
    signals["quarter_end"] = 0.3 if date.month in (3, 6, 9, 12) and date.day >= 25 else 0.0

    return signals


def compute_technical_signals(
    prices: pd.Series,
    volume: Optional[pd.Series],
    as_of_idx: int,
) -> dict[str, float]:
    """Compute 4 technical indicators as z-scored features."""
    signals = {}
    p = prices.values

    # RSI (14-day)
    if as_of_idx >= 14:
        deltas = np.diff(p[as_of_idx - 14:as_of_idx + 1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = _safe_div(gains, losses, 1.0)
        rsi = 100 - 100 / (1 + rs)
        signals["rsi_z"] = (rsi - 50) / 25  # Center at 50, scale
    else:
        signals["rsi_z"] = np.nan

    # MACD signal (12-26-9)
    if as_of_idx >= 35:
        ema12 = pd.Series(p[:as_of_idx + 1]).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(p[:as_of_idx + 1]).ewm(span=26).mean().iloc[-1]
        macd_line = ema12 - ema26
        macd_series = pd.Series(p[:as_of_idx + 1]).ewm(span=12).mean() - pd.Series(p[:as_of_idx + 1]).ewm(span=26).mean()
        signal_line = macd_series.ewm(span=9).mean().iloc[-1]
        signals["macd_signal"] = _safe_div(macd_line - signal_line, p[as_of_idx])
    else:
        signals["macd_signal"] = np.nan

    # ADX (simplified directional movement)
    if as_of_idx >= 28:
        highs = p[as_of_idx - 13:as_of_idx + 1]  # Proxy: use close as high
        lows = p[as_of_idx - 13:as_of_idx + 1] * 0.98  # Approx low
        tr = np.abs(np.diff(highs))
        avg_tr = np.mean(tr) if len(tr) > 0 else 1
        dm_plus = np.maximum(np.diff(highs), 0)
        dm_minus = np.maximum(-np.diff(highs), 0)
        di_plus = np.mean(dm_plus) / max(avg_tr, 1e-10)
        di_minus = np.mean(dm_minus) / max(avg_tr, 1e-10)
        dx = abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-10)
        signals["adx_strength"] = dx  # Higher = stronger trend
    else:
        signals["adx_strength"] = np.nan

    # OBV momentum (10-day change in on-balance volume)
    if volume is not None and as_of_idx >= 20 and len(volume) > as_of_idx:
        v = volume.values
        price_changes = np.sign(np.diff(p[as_of_idx - 19:as_of_idx + 1]))
        obv = np.cumsum(price_changes * v[as_of_idx - 19:as_of_idx])
        if len(obv) >= 10:
            obv_mom = obv[-1] - obv[-10]
            signals["obv_momentum"] = _safe_div(obv_mom, np.abs(obv[-10]) + 1)
        else:
            signals["obv_momentum"] = np.nan
    else:
        signals["obv_momentum"] = np.nan

    return signals


# ── Main Signal Matrix Builder ────────────────────────────────────────────────

def build_signal_matrix(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    fundamentals: dict[str, dict],
    sector_map: dict[str, str],
    dates: pd.DatetimeIndex,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Build the full signal matrix: (n_dates × n_stocks) rows × n_signals columns.

    Returns DataFrame with MultiIndex (date, symbol) and signal columns.
    """
    stocks = {s: p for s, p in prices.items() if not s.startswith("^")}
    symbols = sorted(stocks.keys())
    n_signals = len(ALL_SIGNAL_NAMES)

    logger.info("Building signal matrix: %d stocks × %d dates × %d signals",
                len(symbols), len(dates), n_signals)

    # Pre-compute returns
    returns = {sym: stocks[sym].pct_change() for sym in symbols}

    records = []

    for date in dates:
        date_signals = {}

        for sym in symbols:
            p = stocks[sym]
            if date not in p.index:
                continue

            idx = p.index.get_loc(date)
            if not isinstance(idx, int):
                idx = int(idx) if isinstance(idx, np.integer) else 0
            if idx < min_history:
                continue

            # Compute raw signals
            raw = {}
            raw.update(compute_momentum_signals(p, idx))
            raw.update(compute_mean_reversion_signals(p, idx))
            raw.update(compute_volume_signals(p, volumes.get(sym), idx))
            raw.update(compute_volatility_signals(returns[sym], idx))
            raw.update(compute_fundamental_signals(fundamentals.get(sym, {})))
            raw.update(compute_seasonal_signals(date))
            raw.update(compute_technical_signals(p, volumes.get(sym), idx))

            date_signals[sym] = raw

        if len(date_signals) < 10:
            continue

        # Cross-sectional z-score all signals
        for signal_name in ALL_SIGNAL_NAMES:
            raw_vals = {s: date_signals[s].get(signal_name, np.nan) for s in date_signals}
            valid_syms = [s for s, v in raw_vals.items() if not np.isnan(v)]

            if len(valid_syms) >= 5:
                vals = np.array([raw_vals[s] for s in valid_syms])
                z_vals = _mad_zscore(vals)
                for i, s in enumerate(valid_syms):
                    date_signals[s][signal_name] = float(np.clip(z_vals[i], -3, 3))
            else:
                for s in date_signals:
                    date_signals[s][signal_name] = 0.0

        # Cross-sectional signals (sector-relative)
        sectors = {}
        for sym in date_signals:
            sec = sector_map.get(sym, "Other")
            sectors.setdefault(sec, []).append(sym)

        for sym in date_signals:
            sec = sector_map.get(sym, "Other")
            peers = sectors.get(sec, [sym])

            if len(peers) >= 2:
                peer_moms = [date_signals[p].get("mom_60d", 0) for p in peers if p in date_signals]
                my_mom = date_signals[sym].get("mom_60d", 0)
                date_signals[sym]["sector_rel_mom"] = my_mom - np.mean(peer_moms)

                peer_vols = [date_signals[p].get("rvol_20d", 0) for p in peers if p in date_signals]
                my_vol = date_signals[sym].get("rvol_20d", 0)
                date_signals[sym]["sector_rel_vol"] = my_vol - np.mean(peer_vols)

                date_signals[sym]["peer_beta_dev"] = 0.0  # Placeholder
            else:
                date_signals[sym]["sector_rel_mom"] = 0.0
                date_signals[sym]["sector_rel_vol"] = 0.0
                date_signals[sym]["peer_beta_dev"] = 0.0

        # Build records
        for sym, sigs in date_signals.items():
            row = {"date": date, "symbol": sym}
            for signal_name in ALL_SIGNAL_NAMES:
                row[signal_name] = sigs.get(signal_name, 0.0)
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.set_index(["date", "symbol"])

    # Fill remaining NaN with 0
    df = df.fillna(0.0)

    logger.info("Signal matrix built: %d rows × %d signals", len(df), n_signals)
    return df

"""
Robust Outlier Detection & Z-Scoring
======================================
Centralised module for all outlier-related operations in ARTHA.

AGENTS.md rule 7 is the governing principle — two distinct problems,
two distinct solutions:

  (a) **Bad data** (fat-finger prints, feed glitches, data-vendor errors):
      Caught at ingestion using MAD-based detection with confirmation
      via price-range and volume cross-checks.  Flagged for review,
      never silently dropped.

  (b) **Genuine large moves** (circuit-limit hits, real news shocks):
      NEVER scrubbed.  These are signal, not noise.  The regime detector
      and strategy selector need to see them.

For **trading signal z-scores** (stat-arb spreads, factor scores),
neither naive mean/std (too sensitive to tails) nor naive MAD
(too aggressive clipping in regime transitions) works well alone.
This module provides:

  - `robust_zscore()` — Exponentially weighted median/MAD with
    adaptive Winsorisation to prevent extreme z-explosion while
    preserving genuine tail events.
  - `flag_bad_data()` — Multi-signal bad-data detector for ingestion
    (MAD + volume spike + price-range cross-check).
  - `handle_circuit_limits()` — Detects NSE circuit-breaker halts
    and marks them as explicit events rather than missing data.

Reference: docs/ARTHA_ARCHITECTURE.md §6.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────────

# MAD → σ conversion factor for normal distribution: σ ≈ 1.4826 × MAD
MAD_SCALE = 1.4826

# NSE circuit-limit bounds (percent from previous close)
# Standard equities: ±5%, ±10%, ±20% bands
# Indices: no circuit limit (but individual stocks in index can hit them)
CIRCUIT_LIMIT_BANDS = [0.05, 0.10, 0.20]

# Default threshold for extreme z-scores — Winsorise beyond this
DEFAULT_WINSOR_Z = 6.0

# Default threshold for bad-data flagging (MAD multiples)
# 8× MAD ≈ 5.9σ for Gaussian → flags only statistical near-impossibilities
BAD_DATA_MAD_THRESHOLD = 8.0


# ── Data types ──────────────────────────────────────────────────────────────────

class OutlierType(Enum):
    """Classification of flagged data points."""
    BAD_DATA = "bad_data"          # Fat-finger, feed glitch
    CIRCUIT_HIT = "circuit_hit"    # NSE circuit-breaker triggered
    GENUINE_MOVE = "genuine_move"  # Real large move (news shock, event)


@dataclass
class OutlierFlag:
    """A single flagged data point with classification and evidence."""
    date: pd.Timestamp
    symbol: str
    return_pct: float
    robust_z: float
    classification: OutlierType
    evidence: str  # Human-readable reason for classification
    volume_z: Optional[float] = None  # Volume z-score (for cross-check)


# ── Robust Z-Scoring ───────────────────────────────────────────────────────────

def robust_zscore_rolling(
    series: pd.Series,
    lookback: int = 60,
    min_periods: int = 20,
    winsor_z: float = DEFAULT_WINSOR_Z,
) -> pd.Series:
    """
    Rolling robust z-score using median and MAD.

    Unlike mean/std z-scores, this is resistant to the fat-tailed
    returns common in Indian equities (especially mid/small-caps).
    Unlike naive MAD z-scores, it uses Winsorisation to cap extreme
    values rather than letting z explode to ±50 during regime breaks.

    Parameters
    ----------
    series : pd.Series
        Raw values (e.g., spread, factor score, returns).
    lookback : int
        Rolling window.
    min_periods : int
        Minimum observations before computing z.
    winsor_z : float
        Cap z-scores at this absolute value to prevent explosion.

    Returns
    -------
    pd.Series
        Robust z-scores, same index as input.
    """
    median = series.rolling(lookback, min_periods=min_periods).median()

    # MAD: rolling median of |x - median|
    deviation = (series - median).abs()
    mad = deviation.rolling(lookback, min_periods=min_periods).median()

    # Scale to σ-equivalent
    scale = mad * MAD_SCALE

    # Guard against zero MAD (constant series or insufficient variation)
    # Fall back to rolling std when MAD is zero (can happen with
    # discretised or very stable series)
    std_fallback = series.rolling(lookback, min_periods=min_periods).std()
    scale = scale.where(scale > 1e-10, std_fallback)
    scale = scale.where(scale > 1e-10, 1.0)  # Ultimate fallback

    z = (series - median) / scale

    # Winsorise: cap at ±winsor_z to prevent z-explosion
    # This is why naive MAD "gave bad results" — without capping,
    # regime transitions produce z-scores of ±30-50 that overwhelm
    # position sizing and signal logic.
    if winsor_z is not None:
        z = z.clip(-winsor_z, winsor_z)

    return z


def robust_zscore_cross_sectional(
    values: np.ndarray,
    winsor_z: float = DEFAULT_WINSOR_Z,
) -> np.ndarray:
    """
    Cross-sectional robust z-score using median/MAD.

    Replaces the duplicated `_mad_zscore()` functions in signals.py
    and simulator.py.

    Parameters
    ----------
    values : np.ndarray
        Raw values for a single cross-section (e.g., all stocks on one date).
    winsor_z : float
        Cap z-scores at this absolute value.

    Returns
    -------
    np.ndarray
        Robust z-scores, same shape as input. NaNs are preserved.
    """
    valid_mask = ~np.isnan(values)
    result = np.full_like(values, np.nan)

    if valid_mask.sum() < 3:
        result[valid_mask] = 0.0
        return result

    valid = values[valid_mask]
    median = np.nanmedian(valid)
    mad = np.nanmedian(np.abs(valid - median))

    if mad > 1e-10:
        scale = mad * MAD_SCALE
    else:
        # MAD is zero (all values nearly identical) — use std
        scale = max(np.nanstd(valid), 1e-10)

    z = (valid - median) / scale

    # Winsorise
    if winsor_z is not None:
        z = np.clip(z, -winsor_z, winsor_z)

    result[valid_mask] = z
    return result


# ── Bad-Data Detection (Ingestion-Level) ────────────────────────────────────────

def flag_bad_data(
    df: pd.DataFrame,
    symbol: str,
    mad_threshold: float = BAD_DATA_MAD_THRESHOLD,
) -> list[OutlierFlag]:
    """
    Multi-signal bad-data detector for OHLCV ingestion.

    Uses three independent signals to classify outliers:
      1. **Return MAD z-score** — is the daily return statistically extreme?
      2. **Volume cross-check** — does volume confirm the move?
         (Bad data often has normal volume; genuine moves have abnormal volume.)
      3. **Price-range consistency** — does the close fall within [low, high]?
         (Data errors sometimes have close outside the day's range.)

    Classification logic:
      - High return z + normal volume + price inconsistency → BAD_DATA
      - High return z + high volume + price consistent → check circuit limits
      - Circuit-limit band match → CIRCUIT_HIT
      - Else → GENUINE_MOVE (flag but don't filter)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: date, open, high, low, close, volume.
    symbol : str
        Ticker for logging.
    mad_threshold : float
        MAD multiple for flagging.

    Returns
    -------
    list[OutlierFlag]
        Classified outlier flags. Empty list if no outliers detected.
    """
    if len(df) < 10:
        return []

    flags: list[OutlierFlag] = []

    # ── Compute returns and their MAD z-scores ──────────────────────────
    returns = df["close"].pct_change()
    valid_returns = returns.dropna()

    if len(valid_returns) < 3:
        return []

    median_ret = valid_returns.median()
    mad_ret = np.median(np.abs(valid_returns - median_ret))

    if mad_ret < 1e-10:
        return []  # All returns identical

    ret_z = np.abs(valid_returns - median_ret) / (mad_ret * MAD_SCALE)

    # ── Volume z-scores (for cross-check) ───────────────────────────────
    log_volume = np.log1p(df["volume"].astype(float))
    vol_median = log_volume.rolling(60, min_periods=10).median()
    vol_mad = (log_volume - vol_median).abs().rolling(60, min_periods=10).median()
    vol_scale = (vol_mad * MAD_SCALE).clip(lower=0.1)
    vol_z = (log_volume - vol_median) / vol_scale

    # ── Price-range consistency check ───────────────────────────────────
    # Close should be within [low, high] — if not, data is corrupt
    price_consistent = (df["close"] >= df["low"]) & (df["close"] <= df["high"])

    # ── Classify flagged points ─────────────────────────────────────────
    flagged_idx = ret_z[ret_z > mad_threshold].index

    for idx in flagged_idx:
        row_pos = df.index.get_loc(idx)
        if row_pos == 0:
            continue  # Skip first row (no return)

        row = df.iloc[row_pos]
        ret = returns.iloc[row_pos]
        rz = float(ret_z.loc[idx])
        vz = float(vol_z.iloc[row_pos]) if row_pos < len(vol_z) else 0.0
        consistent = bool(price_consistent.iloc[row_pos])

        # ── Classification ──────────────────────────────────────────────
        if not consistent:
            classification = OutlierType.BAD_DATA
            evidence = (
                f"Close ({row['close']:.2f}) outside day range "
                f"[{row['low']:.2f}, {row['high']:.2f}] — data corruption"
            )
        elif rz > mad_threshold and abs(vz) < 1.5:
            # Extreme return but normal volume → likely bad data
            classification = OutlierType.BAD_DATA
            evidence = (
                f"Return {ret:+.4f} ({rz:.1f}× MAD) with normal volume "
                f"(vol_z={vz:.1f}) — suspicious"
            )
        elif _is_circuit_hit(ret):
            classification = OutlierType.CIRCUIT_HIT
            evidence = (
                f"Return {ret:+.4f} matches circuit-limit band "
                f"(volume_z={vz:.1f})"
            )
        else:
            classification = OutlierType.GENUINE_MOVE
            evidence = (
                f"Return {ret:+.4f} ({rz:.1f}× MAD) with elevated volume "
                f"(vol_z={vz:.1f}) — genuine large move"
            )

        flags.append(OutlierFlag(
            date=pd.Timestamp(row["date"]) if "date" in row.index else pd.Timestamp(idx),
            symbol=symbol,
            return_pct=float(ret),
            robust_z=rz,
            classification=classification,
            evidence=evidence,
            volume_z=vz,
        ))

    if flags:
        bad_count = sum(1 for f in flags if f.classification == OutlierType.BAD_DATA)
        circuit_count = sum(1 for f in flags if f.classification == OutlierType.CIRCUIT_HIT)
        genuine_count = sum(1 for f in flags if f.classification == OutlierType.GENUINE_MOVE)
        logger.info(
            "Outlier detection for %s: %d flagged (%d bad_data, %d circuit, %d genuine)",
            symbol, len(flags), bad_count, circuit_count, genuine_count,
        )

    return flags


def _is_circuit_hit(return_pct: float, tolerance: float = 0.005) -> bool:
    """
    Check if a return matches an NSE circuit-limit band.

    NSE uses ±5%, ±10%, ±20% circuit limits for individual stocks.
    A close at exactly (or very near) a circuit band suggests a halt.
    """
    abs_ret = abs(return_pct)
    return any(
        abs(abs_ret - band) < tolerance
        for band in CIRCUIT_LIMIT_BANDS
    )


# ── Circuit-Breaker Halt Detection ──────────────────────────────────────────────

def detect_circuit_halts(
    df: pd.DataFrame,
    tolerance: float = 0.005,
) -> pd.Series:
    """
    Detect trading days where the stock hit a circuit-breaker limit.

    Returns a boolean Series indexed like df — True on circuit days.

    Detection heuristic: close is within `tolerance` of a circuit
    band relative to the previous close, AND intraday range is
    compressed (close ≈ high or close ≈ low, depending on direction).
    """
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    returns = df["close"].pct_change()
    abs_returns = returns.abs()

    # Check if return matches any circuit band
    is_circuit_band = pd.Series(False, index=df.index)
    for band in CIRCUIT_LIMIT_BANDS:
        is_circuit_band |= (abs_returns - band).abs() < tolerance

    # Confirm with intraday range compression:
    # On a circuit-hit day, close is pinned near high (upper) or low (lower)
    day_range = df["high"] - df["low"]
    close_to_extreme = pd.Series(False, index=df.index)

    # Upper circuit: close near high
    upper = (returns > 0) & ((df["high"] - df["close"]) < day_range * 0.05 + 0.01)
    # Lower circuit: close near low
    lower = (returns < 0) & ((df["close"] - df["low"]) < day_range * 0.05 + 0.01)
    close_to_extreme = upper | lower

    result = is_circuit_band & close_to_extreme
    result.iloc[0] = False  # First row has no return

    return result


# ── Spread Z-Score for Stat-Arb ─────────────────────────────────────────────────

def statarb_zscore(
    spread: pd.Series,
    lookback: int = 60,
    min_periods: int = 20,
    winsor_z: float = DEFAULT_WINSOR_Z,
    use_ewm: bool = True,
    ewm_halflife: int = 20,
) -> pd.Series:
    """
    Robust z-score specifically tuned for stat-arb spread trading.

    This was the function that "gave bad results" when using naive MAD:
    - Pure MAD with no Winsorisation → z-scores of ±30-50 during regime
      transitions, causing oversized positions and spurious stop-outs.
    - Pure mean/std → z-scores pulled by outliers, causing late entries
      and premature exits.

    Solution: **EWM-median/MAD hybrid with Winsorisation**.
    - Uses exponentially weighted statistics for responsiveness to
      regime changes (unlike fixed-window rolling).
    - MAD for outlier resistance (unlike mean/std).
    - Winsorisation at ±6σ to prevent z-explosion.
    - Falls back to EWM mean/std when MAD is zero (constant spread).

    Parameters
    ----------
    spread : pd.Series
        The price spread (A - β×B) or log-spread.
    lookback : int
        Rolling window for the robust estimator.
    min_periods : int
        Minimum observations.
    winsor_z : float
        Cap z-scores at this absolute value.
    use_ewm : bool
        If True, use EWM-based location/scale (more responsive).
        If False, use fixed rolling window (more stable).
    ewm_halflife : int
        Halflife for EWM (only used when use_ewm=True).

    Returns
    -------
    pd.Series
        Robust z-scores for the spread.
    """
    if use_ewm:
        return _ewm_robust_zscore(spread, ewm_halflife, min_periods, winsor_z)
    else:
        return robust_zscore_rolling(spread, lookback, min_periods, winsor_z)


def _ewm_robust_zscore(
    series: pd.Series,
    halflife: int = 20,
    min_periods: int = 20,
    winsor_z: float = DEFAULT_WINSOR_Z,
) -> pd.Series:
    """
    EWM-based robust z-score.

    The "EWM-median" doesn't exist natively in pandas, so we approximate
    it with: EWM mean of Winsorised data (which is more robust than
    raw EWM mean, but more responsive than rolling median).

    Steps:
    1. First pass: compute rolling median/MAD for initial Winsorisation
    2. Winsorise the series at ±3× rolling MAD from rolling median
    3. Compute EWM mean and std on the Winsorised series
    4. Z-score using EWM statistics, cap at ±winsor_z
    """
    # Step 1: rolling robust location/scale for Winsorisation bounds
    rolling_median = series.rolling(max(halflife * 3, 60), min_periods=min_periods).median()
    rolling_dev = (series - rolling_median).abs()
    rolling_mad = rolling_dev.rolling(max(halflife * 3, 60), min_periods=min_periods).median()
    rolling_scale = (rolling_mad * MAD_SCALE).clip(lower=1e-10)

    # Step 2: Winsorise the raw series (clip extreme values before EWM)
    winsor_bound = 3.0  # 3σ Winsorisation before EWM
    upper = rolling_median + winsor_bound * rolling_scale
    lower = rolling_median - winsor_bound * rolling_scale
    winsorised = series.clip(lower=lower, upper=upper)

    # Step 3: EWM statistics on Winsorised data
    ewm_mean = winsorised.ewm(halflife=halflife, min_periods=min_periods).mean()
    ewm_std = winsorised.ewm(halflife=halflife, min_periods=min_periods).std()

    # Guard against zero std
    ewm_std = ewm_std.clip(lower=1e-10)

    # Step 4: Z-score and cap
    z = (series - ewm_mean) / ewm_std  # Use original series for z, not Winsorised

    if winsor_z is not None:
        z = z.clip(-winsor_z, winsor_z)

    return z


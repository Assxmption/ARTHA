"""
India VIX Data Pipeline
=========================
Fetches, caches, and processes India VIX historical data for the ARTHA
options engine.

India VIX is computed by NSE from the order book of NIFTY 50 options
using the CBOE methodology. It represents the market's expectation of
annualized volatility over the next 30 calendar days.

Data Sources (in order of preference):
  1. Local cache (Parquet file in data_cache/)
  2. yfinance (^INDIAVIX ticker)
  3. jugaad-data NSE module

The VIX data feeds:
  - VRP computation (IV proxy when per-stock IV isn't available)
  - VIX regime classification (LOW/MEDIUM/HIGH)
  - Term structure analysis (via multi-expiry NIFTY option IV)
  - Strategy Selector (regime × VIX → options strategy)

Reference: docs/ARTHA_ARCHITECTURE.md §4.2, Implementation Plan §Component 1
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────────

CACHE_DIR = Path("data_cache")
VIX_CACHE_FILE = CACHE_DIR / "india_vix_history.parquet"
VIX_CACHE_STALENESS_DAYS = 1  # Re-fetch if cache is older than this

# India VIX historical characteristics (for reference, not hardcoded thresholds):
# Range: ~9 (extreme calm, Jan 2020) to ~86 (COVID crash, March 2020)
# Long-term median: ~15-17
# Mean: ~18-20 (skewed by crisis spikes)


# ── Data Fetching ───────────────────────────────────────────────────────────────


def fetch_vix_yfinance(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch India VIX historical data via yfinance.

    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume.
    The 'Close' column is the India VIX closing value.

    Parameters
    ----------
    start_date : start date (default: 5 years ago)
    end_date : end date (default: today)

    Returns
    -------
    DataFrame or None if fetch failed.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, cannot fetch VIX data")
        return None

    if start_date is None:
        start_date = date.today() - timedelta(days=5 * 365)
    if end_date is None:
        end_date = date.today()

    try:
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        if df.empty:
            logger.warning("yfinance returned empty data for ^INDIAVIX")
            return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"

        logger.info(
            "Fetched India VIX: %d days from %s to %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df

    except Exception as e:
        logger.error("Failed to fetch India VIX via yfinance: %s", e)
        return None


def fetch_vix_jugaad(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch India VIX via jugaad-data's NSE module (fallback).

    jugaad-data fetches index data from NSE directly.
    India VIX is available as an index with symbol 'INDIA VIX'.
    """
    try:
        from jugaad_data.nse import index_df
    except ImportError:
        logger.warning("jugaad-data not installed, cannot fetch VIX")
        return None

    if start_date is None:
        start_date = date.today() - timedelta(days=5 * 365)
    if end_date is None:
        end_date = date.today()

    try:
        df = index_df(
            symbol="INDIA VIX",
            from_date=start_date,
            to_date=end_date,
        )

        if df.empty:
            logger.warning("jugaad-data returned empty VIX data")
            return None

        # Normalize column names
        df = df.rename(columns={
            "HistoricalDate": "Date",
            "OPEN": "Open",
            "HIGH": "High",
            "LOW": "Low",
            "CLOSE": "Close",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        logger.info(
            "Fetched India VIX via jugaad-data: %d days", len(df),
        )
        return df

    except Exception as e:
        logger.error("Failed to fetch VIX via jugaad-data: %s", e)
        return None


# ── Cache Management ────────────────────────────────────────────────────────────


def _load_cache() -> Optional[pd.DataFrame]:
    """Load cached VIX data from Parquet file."""
    if not VIX_CACHE_FILE.exists():
        return None

    try:
        df = pd.read_parquet(VIX_CACHE_FILE)
        if df.empty:
            return None

        # Check staleness
        last_date = pd.Timestamp(df.index.max()).date()
        days_old = (date.today() - last_date).days

        if days_old > VIX_CACHE_STALENESS_DAYS:
            logger.info("VIX cache is %d days old, will refresh", days_old)
            return df  # Return stale data, caller will refresh

        logger.info("VIX cache hit: %d days, last=%s", len(df), last_date)
        return df

    except Exception as e:
        logger.warning("Failed to load VIX cache: %s", e)
        return None


def _save_cache(df: pd.DataFrame) -> None:
    """Save VIX data to Parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(VIX_CACHE_FILE, engine="pyarrow")
        logger.info("Saved VIX cache: %d rows", len(df))
    except Exception as e:
        logger.warning("Failed to save VIX cache: %s", e)


# ── Public API ──────────────────────────────────────────────────────────────────


def get_vix_history(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Get India VIX historical data with caching.

    Tries cache first, then yfinance, then jugaad-data.
    Returns a DataFrame with DatetimeIndex and at minimum a 'Close' column
    containing the VIX closing values.

    Parameters
    ----------
    start_date : start date (default: 5 years ago)
    end_date : end date (default: today)
    force_refresh : bypass cache

    Returns
    -------
    DataFrame with VIX data. Empty DataFrame if all sources fail.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            last_date = pd.Timestamp(cached.index.max()).date()
            days_old = (date.today() - last_date).days
            if days_old <= VIX_CACHE_STALENESS_DAYS:
                # Cache is fresh enough
                if start_date:
                    cached = cached[cached.index >= pd.Timestamp(start_date)]
                if end_date:
                    cached = cached[cached.index <= pd.Timestamp(end_date)]
                return cached

    # Fetch fresh data
    df = fetch_vix_yfinance(start_date, end_date)
    if df is None:
        df = fetch_vix_jugaad(start_date, end_date)

    if df is not None and not df.empty:
        # Merge with cached data if available
        cached = _load_cache()
        if cached is not None:
            df = pd.concat([cached, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()
        _save_cache(df)
        return df

    # All fetches failed — return cache even if stale
    cached = _load_cache()
    if cached is not None:
        logger.warning("Using stale VIX cache (all live fetches failed)")
        return cached

    logger.error("No VIX data available from any source")
    return pd.DataFrame()


def get_vix_close(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.Series:
    """
    Get India VIX closing values as a clean Series.

    Convenience wrapper over get_vix_history() that returns just the
    Close column, with any NaN values forward-filled.
    """
    df = get_vix_history(start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float, name="vix")

    vix = df["Close"].copy()
    vix.name = "vix"
    vix = vix.ffill()  # Forward-fill holidays/gaps
    return vix


# ── VIX Analytics ───────────────────────────────────────────────────────────────


def compute_vix_percentile(
    vix_series: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Compute rolling percentile rank of VIX.

    Returns a Series of values [0, 100] indicating what fraction of
    the trailing `lookback` days had a lower VIX.

    VIX < 33rd percentile → LOW regime (complacency, good for selling)
    VIX 33-67th percentile → MEDIUM regime (normal)
    VIX > 67th percentile → HIGH regime (fear, be cautious selling)
    """
    def _pct_rank(window):
        if len(window) < 2:
            return 50.0
        current = window.iloc[-1]
        rank = (window.iloc[:-1] < current).sum()
        return (rank / (len(window) - 1)) * 100

    return vix_series.rolling(lookback, min_periods=30).apply(
        _pct_rank, raw=False,
    )


def compute_vix_regime_series(
    vix_series: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Classify each day into a VIX regime: LOW, MEDIUM, HIGH.

    Uses trailing percentile rank (dynamic thresholds, not fixed VIX levels).
    """
    pct = compute_vix_percentile(vix_series, lookback)

    def _classify(p):
        if np.isnan(p):
            return "MEDIUM"
        if p < 33:
            return "LOW"
        elif p < 67:
            return "MEDIUM"
        else:
            return "HIGH"

    return pct.apply(_classify)


def compute_vix_statistics(vix_series: pd.Series) -> dict:
    """
    Compute summary statistics for the VIX series.

    Returns a dict of metrics useful for reporting and strategy calibration.
    """
    if vix_series.empty or len(vix_series) < 10:
        return {}

    return {
        "current": float(vix_series.iloc[-1]),
        "mean": float(vix_series.mean()),
        "median": float(vix_series.median()),
        "std": float(vix_series.std()),
        "min": float(vix_series.min()),
        "max": float(vix_series.max()),
        "percentile_25": float(np.percentile(vix_series.dropna(), 25)),
        "percentile_75": float(np.percentile(vix_series.dropna(), 75)),
        "current_percentile": float(
            (vix_series < vix_series.iloc[-1]).sum() / len(vix_series) * 100
        ),
        "regime": (
            "LOW" if (vix_series < vix_series.iloc[-1]).sum() / len(vix_series) < 0.33
            else "HIGH" if (vix_series < vix_series.iloc[-1]).sum() / len(vix_series) > 0.67
            else "MEDIUM"
        ),
        "days_in_series": len(vix_series),
    }


def compute_vix_mean_reversion_signal(
    vix_series: pd.Series,
    fast_window: int = 5,
    slow_window: int = 60,
) -> pd.Series:
    """
    VIX mean-reversion signal: z-score of fast MA vs slow MA.

    VIX is one of the most mean-reverting assets in finance. When it spikes
    far above its slow average, it tends to revert — this is the exact
    moment when selling premium is most profitable (high IV → high VRP).

    Positive z-score → VIX elevated → sell premium opportunity.
    Negative z-score → VIX depressed → cheap protection opportunity.

    Uses MAD-robust z-scoring (AGENTS.md rule 7).
    """
    fast_ma = vix_series.rolling(fast_window, min_periods=3).mean()
    slow_ma = vix_series.rolling(slow_window, min_periods=20).mean()

    spread = fast_ma - slow_ma
    median = spread.rolling(252, min_periods=60).median()
    mad = (spread - median).abs().rolling(252, min_periods=60).median()
    mad_std = mad * 1.4826

    return (spread - median) / mad_std.clip(lower=0.001)


# ── CLI ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    force = "--refresh" in sys.argv

    print("Fetching India VIX data...")
    vix = get_vix_close(force_refresh=force)

    if vix.empty:
        print("ERROR: No VIX data available")
        sys.exit(1)

    stats = compute_vix_statistics(vix)
    print(f"\n{'='*50}")
    print(f"India VIX Summary ({stats.get('days_in_series', 0)} trading days)")
    print(f"{'='*50}")
    print(f"Current:     {stats.get('current', 0):.2f}")
    print(f"Mean:        {stats.get('mean', 0):.2f}")
    print(f"Median:      {stats.get('median', 0):.2f}")
    print(f"Std Dev:     {stats.get('std', 0):.2f}")
    print(f"Range:       {stats.get('min', 0):.2f} – {stats.get('max', 0):.2f}")
    print(f"25th pct:    {stats.get('percentile_25', 0):.2f}")
    print(f"75th pct:    {stats.get('percentile_75', 0):.2f}")
    print(f"Current pct: {stats.get('current_percentile', 0):.1f}th percentile")
    print(f"Regime:      {stats.get('regime', 'UNKNOWN')}")

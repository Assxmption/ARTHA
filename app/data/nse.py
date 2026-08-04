"""
NSE Data Ingestion
==================
Historical OHLCV price data for NSE equities, with local caching.

Sources (in priority order):
  1. jugaad-data — purpose-built for NSE, actively maintained.
  2. yfinance with `.NS` suffix — fallback for long-range history or
     when jugaad-data is temporarily blocked by NSE website changes.

All data is cached locally as Parquet files after first fetch.
Subsequent calls read from cache unless `force_refresh=True`.

Outlier handling (AGENTS.md rule 7):
  (a) Bad data (fat-finger prints, feed glitches) is flagged at ingestion
      using MAD-based robust detection.  These are NOT silently dropped —
      they are flagged for review.
  (b) Genuine large market moves (circuit-limit hits, real news shocks)
      are NEVER scrubbed — they are signal, not noise.

Reference: docs/ARTHA_ARCHITECTURE.md §4.2
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path("data_cache/nse")

# MAD multiplier for outlier flagging.
# 6× MAD ≈ 4σ for Gaussian data — flags only extreme outliers.
# Chosen high to avoid flagging genuine large moves (rule 7b).
_MAD_THRESHOLD = 6.0


# ── Symbol normalization ────────────────────────────────────────────────────────

def normalize_nse_symbol(symbol: str) -> str:
    """
    Normalize a symbol to the canonical NSE form.

    Strips whitespace, uppercases, removes `.NS` suffix if present,
    handles common BSE ↔ NSE aliases for blue-chips.
    """
    s = symbol.strip().upper()
    if s.endswith(".NS"):
        s = s[:-3]
    # Common aliases (extend as needed)
    _ALIASES = {
        "HDFCBANK": "HDFCBANK",
        "HDFC BANK": "HDFCBANK",
        "ICICI BANK": "ICICIBANK",
        "RELIANCE INDUSTRIES": "RELIANCE",
        "TATA CONSULTANCY": "TCS",
        "INFOSYS": "INFY",
    }
    return _ALIASES.get(s, s)


def _yfinance_symbol(nse_symbol: str) -> str:
    """Convert an NSE symbol to yfinance format."""
    return f"{nse_symbol}.NS"


# ── MAD-based outlier detection ─────────────────────────────────────────────────

def flag_outliers_mad(
    returns: pd.Series,
    threshold: float = _MAD_THRESHOLD,
) -> pd.Series:
    """
    Flag potential bad-data outliers using Median Absolute Deviation.

    Returns a boolean Series — True where the return is flagged.
    Uses MAD (not mean/std) because financial returns are heavy-tailed;
    mean/std would flag genuine large moves as outliers (rule 7 violation).

    Genuine large moves (circuit-limit hits, real news shocks) at
    moderate MAD multiples are NOT flagged — only extreme statistical
    implausibilities are.
    """
    if len(returns) < 3:
        # Too few observations to compute MAD meaningfully
        return pd.Series(False, index=returns.index)

    median = returns.median()
    mad = np.median(np.abs(returns - median))

    if mad == 0:
        # All returns are identical — no outliers by definition
        return pd.Series(False, index=returns.index)

    # Robust z-score (using MAD scaling factor 1.4826 for Gaussian consistency)
    robust_z = np.abs(returns - median) / (mad * 1.4826)
    return robust_z > threshold


# ── Data fetching ───────────────────────────────────────────────────────────────

def fetch_nse_ohlcv(
    symbol: str,
    start_date: date,
    end_date: Optional[date] = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for an NSE equity.

    Returns a DataFrame with columns:
        date, open, high, low, close, volume, symbol

    The data is cached as a Parquet file.  If cached data exists and
    covers the requested range, it is returned directly.

    Parameters
    ----------
    symbol : str
        NSE symbol (e.g. 'HDFCBANK').
    start_date : date
        Start of the requested range (inclusive).
    end_date : date, optional
        End of the requested range (inclusive).  Defaults to today.
    cache_dir : Path
        Directory for Parquet cache files.
    force_refresh : bool
        If True, bypass cache and re-fetch.

    Returns
    -------
    pd.DataFrame
        OHLCV data, sorted by date ascending.
    """
    symbol = normalize_nse_symbol(symbol)
    if end_date is None:
        end_date = date.today()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}.parquet"

    # ── Try cache ───────────────────────────────────────────────────────
    if not force_refresh and cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            if not cached.empty:
                cached_start = pd.Timestamp(cached["date"].min()).date()
                cached_end = pd.Timestamp(cached["date"].max()).date()
                if cached_start <= start_date and cached_end >= end_date:
                    mask = (
                        (cached["date"] >= pd.Timestamp(start_date))
                        & (cached["date"] <= pd.Timestamp(end_date))
                    )
                    logger.info(
                        "Cache hit for %s [%s–%s]", symbol, start_date, end_date
                    )
                    return cached.loc[mask].reset_index(drop=True)
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", symbol, e)

    # ── Fetch from jugaad-data (primary) ────────────────────────────────
    df = _fetch_jugaad(symbol, start_date, end_date)

    # ── Fallback to yfinance ────────────────────────────────────────────
    if df is None or df.empty:
        logger.info("jugaad-data returned empty for %s, trying yfinance", symbol)
        df = _fetch_yfinance(symbol, start_date, end_date)

    if df is None or df.empty:
        logger.warning("No data fetched for %s [%s–%s]", symbol, start_date, end_date)
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "symbol"]
        )

    # ── Normalize columns ───────────────────────────────────────────────
    df["symbol"] = symbol
    df = df.sort_values("date").reset_index(drop=True)

    # ── Flag outliers at ingestion (rule 7a) ────────────────────────────
    if len(df) >= 3:
        returns = df["close"].pct_change().dropna()
        flags = flag_outliers_mad(returns)
        n_flagged = flags.sum()
        if n_flagged > 0:
            flagged_dates = df.loc[flags[flags].index + 1, "date"].tolist()
            logger.warning(
                "MAD outlier detection flagged %d observations for %s "
                "(NOT dropped — review manually): %s",
                n_flagged,
                symbol,
                flagged_dates[:5],
            )

    # ── Cache ───────────────────────────────────────────────────────────
    try:
        # Merge with existing cache to extend coverage
        if cache_file.exists():
            existing = pd.read_parquet(cache_file)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["date", "symbol"]
            ).sort_values("date").reset_index(drop=True)
        df.to_parquet(cache_file, index=False)
        logger.info("Cached %d rows for %s", len(df), symbol)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", symbol, e)

    # Return only the requested range
    mask = (
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    )
    return df.loc[mask].reset_index(drop=True)


# ── Private fetch helpers ───────────────────────────────────────────────────────

def _fetch_jugaad(
    symbol: str, start_date: date, end_date: date
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from jugaad-data.  Returns None on failure."""
    try:
        from jugaad_data.nse import stock_df

        df = stock_df(
            symbol=symbol,
            from_date=start_date,
            to_date=end_date,
            series="EQ",
        )
        if df is None or df.empty:
            return None

        # jugaad-data column normalization (columns vary by version)
        col_map = {}
        for col in df.columns:
            cl = col.strip().upper()
            if cl == "DATE" or cl == "CH_TIMESTAMP":
                col_map[col] = "date"
            elif cl in ("OPEN", "CH_OPENING_PRICE"):
                col_map[col] = "open"
            elif cl in ("HIGH", "CH_TRADE_HIGH_PRICE"):
                col_map[col] = "high"
            elif cl in ("LOW", "CH_TRADE_LOW_PRICE"):
                col_map[col] = "low"
            elif cl in ("CLOSE", "CH_CLOSING_PRICE"):
                col_map[col] = "close"
            elif cl in ("VOLUME", "CH_TOT_TRADED_QTY"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)
        needed = {"date", "open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            logger.warning(
                "jugaad-data columns mismatch for %s: got %s", symbol, list(df.columns)
            )
            return None

        df = df[list(needed)].copy()
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except ImportError:
        logger.info("jugaad-data not installed, skipping")
        return None
    except Exception as e:
        logger.warning("jugaad-data fetch failed for %s: %s", symbol, e)
        return None


def _fetch_yfinance(
    symbol: str, start_date: date, end_date: date
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance.  Returns None on failure."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(_yfinance_symbol(symbol))
        # yfinance end is exclusive, so add 1 day
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,  # adjust for splits/dividends
        )
        if df is None or df.empty:
            return None

        df = df.reset_index()
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    except ImportError:
        logger.info("yfinance not installed, skipping")
        return None
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None


# ── Convenience ─────────────────────────────────────────────────────────────────

def get_nse_returns(
    symbol: str,
    start_date: date,
    end_date: Optional[date] = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> pd.Series:
    """
    Fetch close prices and compute log-returns.

    Returns a named Series indexed by date.
    """
    df = fetch_nse_ohlcv(symbol, start_date, end_date, cache_dir)
    if df.empty:
        return pd.Series(dtype=float, name=symbol)
    prices = df.set_index("date")["close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = symbol
    return returns

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
from zoneinfo import ZoneInfo

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


_YAHOO_INDEX_SYMBOLS = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY 50": "^NSEI", "^NSEI": "^NSEI",
    "BANKNIFTY": "^NSEBANK", "NIFTY BANK": "^NSEBANK", "^NSEBANK": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX", "INDIA VIX": "^INDIAVIX", "^INDIAVIX": "^INDIAVIX",
}


def _yfinance_symbol(nse_symbol: str) -> str:
    """Convert an NSE symbol (or index alias) to yfinance format."""
    from app.data.symbols import resolve_symbol

    s = nse_symbol.strip().upper()
    if s in _YAHOO_INDEX_SYMBOLS:
        return _YAHOO_INDEX_SYMBOLS[s]
    return f"{resolve_symbol(s)}.NS"  # former tickers map to the current listing


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


# ── Session-date handling ───────────────────────────────────────────────────────

_IST = ZoneInfo("Asia/Kolkata")
# NSE equity cash session closes at 15:30 IST; a bar for "today" fetched
# before then is still forming and must not be cached as final.
_SESSION_CLOSE_IST = (15, 30)
# Cache files written before session dates were normalised (no `source`
# column) mixed shifted jugaad timestamps with yfinance rows and are
# discarded on read rather than trusted.
_CACHE_REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume", "symbol", "source"}
_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol", "source", "provisional"]


def _now_ist() -> datetime:
    return datetime.now(_IST)


def ist_today() -> date:
    """Today's date on the NSE calendar (independent of the server timezone)."""
    return _now_ist().date()


def _session_closed(now: Optional[datetime] = None) -> bool:
    now = now or _now_ist()
    return (now.hour, now.minute) >= _SESSION_CLOSE_IST


def normalize_session_dates(values: pd.Series) -> pd.Series:
    """
    Map provider timestamps to the NSE trading date (naive midnight).

    jugaad-data returns each IST session date as a *naive UTC instant*:
    the 2026-11-13 session arrives as ``2026-11-12 18:30:00``.  Taking
    ``.date()`` of that labels every bar one day early (Friday bars land on
    Thursday, Monday bars on Sunday).  Any naive timestamp that carries a
    time-of-day is therefore read as UTC and converted to IST before the
    date is taken.  Timezone-aware inputs (yfinance) are converted to IST.
    """
    ts = pd.to_datetime(values)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(_IST).dt.tz_localize(None)
    else:
        has_time = ts.notna() & (ts != ts.dt.normalize())
        if has_time.any():
            shifted = ts.dt.tz_localize("UTC").dt.tz_convert(_IST).dt.tz_localize(None)
            ts = ts.where(~has_time, shifted)
    return ts.dt.normalize().astype("datetime64[ns]")


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLUMNS)


# ── Data fetching ───────────────────────────────────────────────────────────────

def fetch_nse_ohlcv(
    symbol: str,
    start_date: date,
    end_date: Optional[date] = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for an NSE equity, current up to the latest session.

    Returns a DataFrame with columns:
        date, open, high, low, close, volume, symbol, source, provisional

    * ``date`` is the NSE session date at naive midnight.
    * Prices are as traded (unadjusted) from both providers so the two
      sources agree bar-for-bar; corporate-action adjustment is a separate,
      explicit step for return calculations.
    * ``source`` records which provider produced each bar (rule 10).
    * ``provisional`` is True for today's bar while the session is open.

    Cache policy: a cached range is served directly only if it fully covers
    the request *and* the request ends before today; otherwise the tail is
    re-fetched from the last cached session, so a bar cached mid-session is
    replaced by its final values and new sessions are appended.
    """
    symbol = normalize_nse_symbol(symbol)
    today = ist_today()
    if end_date is None or end_date > today:
        end_date = today
    if start_date > end_date:
        return _empty_ohlcv()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}.parquet"

    cached = None if force_refresh else _read_cache(cache_file, symbol)
    fetch_start = start_date
    if cached is not None and not cached.empty:
        cached_start = cached["date"].min().date()
        cached_end = cached["date"].max().date()
        if cached_start <= start_date and cached_end >= end_date and end_date < today:
            logger.info("Cache hit for %s [%s–%s]", symbol, start_date, end_date)
            return _slice(cached, start_date, end_date)
        if cached_start <= start_date:
            fetch_start = max(start_date, cached_end)

    fresh = _fetch_range(symbol, fetch_start, end_date)

    if fresh.empty:
        if cached is not None and not cached.empty:
            logger.warning(
                "No fresh data for %s [%s–%s]; serving cached bars only",
                symbol, fetch_start, end_date,
            )
            return _slice(cached, start_date, end_date)
        logger.warning("No data fetched for %s [%s–%s]", symbol, start_date, end_date)
        return _empty_ohlcv()

    fresh["symbol"] = symbol
    _flag_ingestion_outliers(fresh, symbol)

    frames = [f for f in (cached, fresh) if f is not None and not f.empty]
    merged = (
        pd.concat(frames, ignore_index=True)
        # Fresh rows win: they replace provisional bars and repaired dates.
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if "provisional" not in merged.columns:
        merged["provisional"] = False
    merged["provisional"] = merged["provisional"].fillna(False).astype(bool)

    try:
        merged.to_parquet(cache_file, index=False)
        logger.info("Cached %d rows for %s", len(merged), symbol)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", symbol, e)

    return _slice(merged, start_date, end_date)


def _slice(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    return df.loc[mask].reset_index(drop=True)


def _read_cache(cache_file: Path, symbol: str) -> Optional[pd.DataFrame]:
    if not cache_file.exists():
        return None
    try:
        cached = pd.read_parquet(cache_file)
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", symbol, e)
        return None
    if not _CACHE_REQUIRED_COLUMNS.issubset(cached.columns):
        logger.warning(
            "Discarding legacy cache for %s (no provenance column; dates may be "
            "shifted by one session) — re-fetching", symbol,
        )
        return None
    cached = cached.copy()
    cached["date"] = normalize_session_dates(cached["date"])
    if "provisional" not in cached.columns:
        cached["provisional"] = False
    return cached.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def _fetch_range(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Kite Connect first when a session is active (exchange data, current to
    the live session); otherwise jugaad-data (NSE archives) with yfinance
    filling whatever jugaad lacks.

    jugaad-data typically lags the live market by one or more sessions, so a
    non-empty jugaad result is not proof of freshness: any sessions after its
    last bar are requested from yfinance, which carries the current session.
    """
    kite_df = _fetch_kite(symbol, start_date, end_date)
    if kite_df is not None and not kite_df.empty:
        today = ist_today()
        kite_df["provisional"] = (kite_df["date"].dt.date == today) & (not _session_closed())
        return kite_df

    parts = []
    jug = _fetch_jugaad(symbol, start_date, end_date)
    yf_start = start_date
    if jug is not None and not jug.empty:
        parts.append(jug)
        yf_start = jug["date"].max().date() + timedelta(days=1)
    if yf_start <= end_date:
        yfd = _fetch_yfinance(symbol, yf_start, end_date)
        if yfd is not None and not yfd.empty:
            parts.append(yfd)
    if not parts:
        return _empty_ohlcv()

    df = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="first")  # exchange archive wins
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df.dropna(subset=["open", "high", "low", "close"])
    today = ist_today()
    df["provisional"] = (df["date"].dt.date == today) & (not _session_closed())
    return df


def _flag_ingestion_outliers(df: pd.DataFrame, symbol: str) -> None:
    """Flag (never drop) suspicious bars at ingestion — AGENTS.md rule 7."""
    if len(df) >= 10:
        from app.quant.robust_outliers import flag_bad_data, detect_circuit_halts, OutlierType
        outlier_flags = flag_bad_data(df, symbol)
        if outlier_flags:
            bad = [f for f in outlier_flags if f.classification == OutlierType.BAD_DATA]
            circuit = [f for f in outlier_flags if f.classification == OutlierType.CIRCUIT_HIT]
            genuine = [f for f in outlier_flags if f.classification == OutlierType.GENUINE_MOVE]
            if bad:
                logger.warning(
                    "BAD DATA flagged for %s (%d points, NOT dropped — review manually): %s",
                    symbol, len(bad),
                    [(str(f.date.date()), f.evidence) for f in bad[:3]],
                )
            if circuit:
                logger.info("Circuit-limit halts detected for %s: %d days", symbol, len(circuit))
            if genuine:
                logger.info("Genuine large moves for %s: %d (preserved as signal)", symbol, len(genuine))

        circuit_days = detect_circuit_halts(df)
        if circuit_days.any():
            df["circuit_halt"] = circuit_days
            logger.info("Marked %d circuit-halt days for %s", circuit_days.sum(), symbol)
    elif len(df) >= 3:
        returns = df["close"].pct_change().dropna()
        n_flagged = flag_outliers_mad(returns).sum()
        if n_flagged > 0:
            logger.warning(
                "MAD outlier detection flagged %d observations for %s "
                "(NOT dropped — review manually)", n_flagged, symbol,
            )


# ── Private fetch helpers ───────────────────────────────────────────────────────

def _fetch_kite(symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """Daily candles from Kite Connect (as traded). None if not connected."""
    try:
        from app.data import kite
    except ImportError:
        return None
    if not kite.is_configured():
        return None
    try:
        client = kite.get_client()
        exchange, tradingsymbol = kite.kite_symbol(symbol)
        token = client.instrument_token(tradingsymbol, exchange)
        df = client.historical(
            token, "day",
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time().replace(microsecond=0)),
        )
    except kite.KiteNotConnected:
        return None
    except Exception as e:
        logger.warning("Kite daily fetch failed for %s: %s", symbol, e)
        return None
    if df.empty:
        return None
    df["date"] = normalize_session_dates(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["source"] = f"kite:{exchange}:{tradingsymbol}"
    return df


def _fetch_jugaad(
    symbol: str, start_date: date, end_date: date
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from jugaad-data (NSE archives).  Returns None on failure."""
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
        needed = ["date", "open", "high", "low", "close", "volume"]
        if not set(needed).issubset(set(df.columns)):
            logger.warning(
                "jugaad-data columns mismatch for %s: got %s", symbol, list(df.columns)
            )
            return None

        df = df[needed].copy()
        df["date"] = normalize_session_dates(df["date"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = "jugaad-data:NSE_EQ"
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
    """Fetch OHLCV from yfinance (`.NS`).  Returns None on failure."""
    try:
        import yfinance as yf

        yf_symbol = _yfinance_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        # yfinance end is exclusive, so add 1 day
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            # As-traded prices, matching NSE archives (jugaad-data) exactly.
            auto_adjust=False,
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
        df["date"] = normalize_session_dates(df["date"])
        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out["source"] = f"yfinance:{yf_symbol}"
        return out

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

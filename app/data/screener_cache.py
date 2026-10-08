"""
Screener Cache — Batch Fundamentals for Multi-Stock Screening
==============================================================
Fetches, caches, and queries fundamental metrics across the NIFTY 200
universe for Screener.in-style stock filtering.

Uses yfinance `.info` for pre-computed ratios (PE, ROE, D/E, etc.)
rather than re-deriving from financial statements — this is both faster
and more consistent with what users see on other platforms.

Data flow:
  1. build_screener_cache() → batch fetch yfinance .info → SQLite
  2. query_screener(filters) → read SQLite → apply filters → return

AGENTS.md Rule 1: All numeric values are from yfinance, not LLM-generated.
AGENTS.md Rule 11: Screener results are informational, not investment advice.
"""

from __future__ import annotations

import logging
import sqlite3
import time as time_mod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Symbol Universe ─────────────────────────────────────────────────────────────

# NIFTY 200 constituents (top by free-float market cap)
# This is a static snapshot; refresh periodically from NSE website.
NIFTY_200_SYMBOLS = [
    # NIFTY 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "BHARTIARTL", "SBIN", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO",
    "HCLTECH", "BAJAJFINSV", "NTPC", "POWERGRID", "ONGC",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "TATASTEEL",
    "HINDALCO", "COALINDIA", "M&M", "INDUSINDBK", "GRASIM",
    "CIPLA", "DRREDDY", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO",
    "BPCL", "DIVISLAB", "TATACONSUM", "BRITANNIA", "TECHM",
    "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN",
    # NIFTY NEXT 50
    "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IOB",
    "INDIANB", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "AUBANK",
    "GODREJCP", "DABUR", "MARICO", "COLPAL", "PIDILITIND",
    "BERGEPAINT", "HAVELLS", "VOLTAS", "CROMPTON", "POLYCAB",
    "ABB", "SIEMENS", "CUMMINSIND", "BEL", "HAL",
    "BHEL", "IRCTC", "CONCOR", "PIIND", "ATUL",
    "SOLARINDS", "COFORGE", "MPHASIS", "PERSISTENT", "LTTS",
    "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY",
    "TRENT", "PAGEIND", "RELAXO", "ABFRL", "RAYMOND",
    "VEDL", "NMDC", "NATIONALUM", "HINDZINC", "JINDALSTEL",
    # More mid-caps for broader coverage
    "IPCALAB", "LAURUSLABS", "BIOCON", "AUROPHARMA", "TORNTPHARM",
    "ALKEM", "GLENMARK", "LUPIN", "NATCOPHARM", "SYNGENE",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
    "PHOENIXLTD", "LODHA", "SOBHA", "SUNTV", "PVRINOX",
    "MUTHOOTFIN", "MANAPPURAM", "CHOLAFIN", "SUNDARMFIN", "LICHSGFIN",
    "RECLTD", "PFC", "IREDA", "CANFINHOME", "IIFL",
    "MAXHEALTH", "FORTIS", "METROPOLIS", "LALPATHLAB", "MEDANTA",
    "MCX", "CDSL", "BSE", "KPITTECH", "TATAELXSI",
    "DEEPAKNTR", "AARTIIND", "CLEAN", "NAVINFLUOR", "FLUOROCHEM",
    "INDIGO", "TATAPOWER", "ADANIGREEN", "NHPC", "SJVN",
    "TORNTPOWER", "CESC", "JSL", "APLAPOLLO", "RATNAMANI",
    "ASTRAL", "SUPREMEIND", "FINPIPE", "KEI", "TIMKEN",
    "SCHAEFFLER", "SKFINDIA", "GRINDWELL", "CARBORUNIV",
    "SAIL", "TATACOMM", "IDEA", "GAIL", "PETRONET",
    "IGL", "MGL", "GSPL", "SUMICHEM",
    "UPL", "COROMANDEL", "GNFC", "CHAMBLFERT", "DHANUKA",
    "SRF", "ENDURANCE", "ESCORTS", "ASHOKLEY", "BALKRISIND",
    "EXIDEIND", "MOTHERSON", "BOSCHLTD", "MRF",
    "CEATLTD", "TVSMOTOR", "SONACOMS", "STARHEALTH", "NIACL",
    "GICRE", "ICICIGI", "ICICIPRULI", "HDFCAMC", "CAMS",
]


# ── Metric Catalog ──────────────────────────────────────────────────────────────

@dataclass
class MetricInfo:
    """Describes a screenable metric."""
    key: str                 # Internal key (column name in SQLite)
    display_name: str        # Human-readable name
    yfinance_field: str      # Key in yfinance .info dict
    unit: str                # "ratio", "percent", "inr", "inr_crore", "text"
    transform: str = "none"  # "none", "pct_to_100", "to_crore"
    is_filterable: bool = True


# The full catalog — adding a new metric is just one line here.
METRIC_CATALOG: list[MetricInfo] = [
    MetricInfo("market_cap",       "Market Cap",       "marketCap",        "inr_crore",  "to_crore"),
    MetricInfo("current_price",    "CMP",              "currentPrice",     "inr"),
    MetricInfo("pe",               "PE Ratio (TTM)",   "trailingPE",       "ratio"),
    MetricInfo("forward_pe",       "Forward PE",       "forwardPE",        "ratio"),
    MetricInfo("pb",               "Price/Book",       "priceToBook",      "ratio"),
    MetricInfo("eps",              "EPS (TTM)",        "trailingEps",      "inr"),
    MetricInfo("book_value",       "Book Value",       "bookValue",        "inr"),
    MetricInfo("dividend_yield",   "Div. Yield",       "dividendYield",    "percent"),
    MetricInfo("roe",              "ROE",              "returnOnEquity",   "percent",    "pct_to_100"),
    MetricInfo("roa",              "ROA",              "returnOnAssets",   "percent",    "pct_to_100"),
    MetricInfo("debt_to_equity",   "D/E Ratio",        "debtToEquity",     "ratio"),
    MetricInfo("revenue_growth",   "Rev. Growth",      "revenueGrowth",    "percent",    "pct_to_100"),
    MetricInfo("profit_margin",    "Profit Margin",    "profitMargins",    "percent",    "pct_to_100"),
    MetricInfo("operating_margin", "OPM",              "operatingMargins", "percent",    "pct_to_100"),
    MetricInfo("free_cashflow",    "Free Cash Flow",   "freeCashflow",     "inr_crore",  "to_crore"),
    MetricInfo("beta",             "Beta",             "beta",             "ratio"),
    MetricInfo("sector",           "Sector",           "sector",           "text",       is_filterable=False),
    MetricInfo("industry",         "Industry",         "industry",         "text",       is_filterable=False),
    MetricInfo("fifty_two_wk_high","52W High",         "fiftyTwoWeekHigh", "inr"),
    MetricInfo("fifty_two_wk_low", "52W Low",          "fiftyTwoWeekLow",  "inr"),
]

METRIC_BY_KEY = {m.key: m for m in METRIC_CATALOG}
NUMERIC_METRICS = [m.key for m in METRIC_CATALOG if m.unit != "text"]

# Valid operators for the filter API
VALID_OPERATORS = {"gt", "lt", "gte", "lte", "eq", "between"}


# ── Data Models ─────────────────────────────────────────────────────────────────

@dataclass
class ScreenerFilter:
    """A single filter condition."""
    metric: str       # Must be in METRIC_BY_KEY
    operator: str     # gt, lt, gte, lte, eq, between
    value: float
    value2: Optional[float] = None  # Only for "between"


@dataclass
class ScreenerResult:
    """One stock's screener data."""
    symbol: str
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    current_price: float = 0.0
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    pb: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    free_cashflow: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_wk_high: Optional[float] = None
    fifty_two_wk_low: Optional[float] = None
    cached_at: str = ""


# ── SQLite Persistence ──────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path("data_cache/screener_cache.db")


def _get_connection(db_path: Path = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get or create SQLite connection with WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection):
    """Create screener_data table if it doesn't exist."""
    cols = ["symbol TEXT PRIMARY KEY", "cached_at TEXT"]
    for m in METRIC_CATALOG:
        if m.unit == "text":
            cols.append(f"{m.key} TEXT DEFAULT ''")
        else:
            cols.append(f"{m.key} REAL")
    conn.execute(f"CREATE TABLE IF NOT EXISTS screener_data ({', '.join(cols)})")
    conn.commit()


# ── Cache Builder ───────────────────────────────────────────────────────────────

def _transform_value(raw_value, metric_info: MetricInfo) -> Optional[float]:
    """Apply unit transformation to a raw yfinance value."""
    if raw_value is None:
        return None
    try:
        val = float(raw_value)
        if np.isnan(val) or np.isinf(val):
            return None
        if metric_info.transform == "pct_to_100":
            val = val * 100
        elif metric_info.transform == "to_crore":
            val = val / 1e7  # INR to crore
        return round(val, 4)
    except (ValueError, TypeError):
        return None


def _fetch_single_info(symbol: str, max_retries: int = 3) -> dict:
    """
    Fetch yfinance .info for a single NSE symbol with retry + backoff.

    yfinance uses Yahoo Finance's unofficial API. Rate limits are not
    documented but empirically: ~2000 requests/hour is safe, aggressive
    bursts of >50 concurrent requests trigger 429s or empty responses.

    Retry strategy: exponential backoff (1s, 2s, 4s) on failure.
    """
    import random

    for attempt in range(max_retries):
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info or {}

            # yfinance returns an empty dict or a dict with only 'trailingPegRatio'
            # when rate-limited — detect this as a soft failure.
            if not info or len(info) < 5:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.debug(
                        "Sparse response for %s (attempt %d), retrying in %.1fs",
                        symbol, attempt + 1, wait,
                    )
                    time_mod.sleep(wait)
                    continue
                return {}

            return info

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                logger.debug(
                    "Error fetching %s (attempt %d): %s, retrying in %.1fs",
                    symbol, attempt + 1, e, wait,
                )
                time_mod.sleep(wait)
            else:
                logger.debug("Failed to fetch info for %s after %d attempts: %s",
                             symbol, max_retries, e)
    return {}


def _fetch_and_transform(symbol: str, now_iso: str) -> dict | None:
    """Fetch + transform a single symbol. Returns a row dict or None."""
    info = _fetch_single_info(symbol)
    if not info or info.get("regularMarketPrice") is None:
        return None

    row = {"symbol": symbol, "cached_at": now_iso}
    for m in METRIC_CATALOG:
        raw = info.get(m.yfinance_field)
        if m.unit == "text":
            row[m.key] = str(raw) if raw else ""
        else:
            row[m.key] = _transform_value(raw, m)
    return row


def build_screener_cache(
    symbols: list[str] | None = None,
    db_path: Path = _DEFAULT_DB_PATH,
    max_workers: int = 8,
    batch_size: int = 20,
    delay_between_batches: float = 1.0,
) -> int:
    """
    Batch-fetch fundamentals for all symbols and store in SQLite.

    Uses ThreadPoolExecutor for concurrent I/O-bound fetches.
    Rate-limit safety:
      - max_workers=8 limits concurrent Yahoo Finance requests
      - delay_between_batches adds cooldown every `batch_size` symbols
      - Per-request retry with exponential backoff in _fetch_single_info

    Performance: ~200 symbols in ~30-45 seconds (vs ~5 min sequential).

    Args:
        symbols: List of NSE symbols. Defaults to NIFTY_200_SYMBOLS.
        db_path: Path to SQLite database.
        max_workers: Max concurrent fetch threads.
        batch_size: Symbols per batch before cooldown pause.
        delay_between_batches: Seconds to pause between batches.

    Returns:
        Number of symbols successfully cached.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if symbols is None:
        symbols = NIFTY_200_SYMBOLS

    conn = _get_connection(db_path)
    _ensure_table(conn)

    cached_count = 0
    failed_symbols = []
    now_iso = datetime.now().isoformat()
    total = len(symbols)

    logger.info("Building screener cache: %d symbols, %d workers", total, max_workers)

    for batch_start in range(0, total, batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        batch_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {
                executor.submit(_fetch_and_transform, sym, now_iso): sym
                for sym in batch
            }

            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    row = future.result()
                    if row:
                        batch_results.append(row)
                    else:
                        failed_symbols.append(sym)
                except Exception as e:
                    logger.debug("Exception fetching %s: %s", sym, e)
                    failed_symbols.append(sym)

        # Write batch to SQLite
        for row in batch_results:
            cols = list(row.keys())
            placeholders = ", ".join(["?"] * len(cols))
            updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "symbol")
            conn.execute(
                f"INSERT INTO screener_data ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(symbol) DO UPDATE SET {updates}",
                [row.get(c) for c in cols],
            )
        conn.commit()
        cached_count += len(batch_results)

        progress = batch_start + len(batch)
        logger.info(
            "Screener cache progress: %d/%d fetched, %d cached, %d failed",
            progress, total, cached_count, len(failed_symbols),
        )

        # Cooldown between batches to avoid rate limits
        if batch_start + batch_size < total:
            time_mod.sleep(delay_between_batches)

    conn.close()
    logger.info(
        "Screener cache built: %d/%d cached, %d failed",
        cached_count, total, len(failed_symbols),
    )
    return cached_count


def refresh_cache(
    db_path: Path = _DEFAULT_DB_PATH,
    max_age_hours: float = 24.0,
) -> int:
    """
    Refresh stale entries in the screener cache.

    Only re-fetches symbols whose cached_at is older than max_age_hours.
    Returns the number of refreshed symbols.
    """
    conn = _get_connection(db_path)
    _ensure_table(conn)

    cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

    # Find stale or missing symbols
    existing = set()
    fresh = set()
    for row in conn.execute("SELECT symbol, cached_at FROM screener_data"):
        existing.add(row["symbol"])
        if row["cached_at"] and row["cached_at"] > cutoff:
            fresh.add(row["symbol"])

    stale = [s for s in NIFTY_200_SYMBOLS if s not in fresh]

    conn.close()

    if not stale:
        logger.info("Cache is fresh — nothing to refresh")
        return 0

    logger.info("Refreshing %d stale symbols", len(stale))
    return build_screener_cache(symbols=stale, db_path=db_path)


# ── Query Engine ────────────────────────────────────────────────────────────────

def query_screener(
    filters: list[ScreenerFilter] | None = None,
    sort_by: str = "market_cap",
    sort_desc: bool = True,
    limit: int = 50,
    sector: str | None = None,
    db_path: Path = _DEFAULT_DB_PATH,
) -> tuple[list[dict], int]:
    """
    Query the screener cache with filters.

    Args:
        filters: List of ScreenerFilter conditions (AND-combined).
        sort_by: Metric key to sort by.
        sort_desc: Sort descending if True.
        limit: Max results to return.
        sector: Optional sector filter (exact match).
        db_path: Path to SQLite database.

    Returns:
        (results, total_matches) where results is a list of dicts.
    """
    conn = _get_connection(db_path)
    _ensure_table(conn)

    where_clauses = []
    params = []

    # Sector filter
    if sector:
        where_clauses.append("sector = ?")
        params.append(sector)

    # Metric filters
    if filters:
        for f in filters:
            if f.metric not in METRIC_BY_KEY:
                logger.warning("Unknown metric: %s", f.metric)
                continue
            if f.operator not in VALID_OPERATORS:
                logger.warning("Invalid operator: %s", f.operator)
                continue

            col = f.metric
            op_map = {
                "gt": ">", "lt": "<", "gte": ">=",
                "lte": "<=", "eq": "=",
            }

            if f.operator == "between":
                if f.value2 is None:
                    continue
                where_clauses.append(f"{col} BETWEEN ? AND ?")
                params.extend([f.value, f.value2])
            else:
                sql_op = op_map[f.operator]
                where_clauses.append(f"{col} {sql_op} ?")
                params.append(f.value)

    # Build query
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Validate sort column
    if sort_by not in METRIC_BY_KEY:
        sort_by = "market_cap"
    sort_dir = "DESC" if sort_desc else "ASC"

    # Count total matches
    count_row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM screener_data WHERE {where_sql}",
        params,
    ).fetchone()
    total = count_row["cnt"] if count_row else 0

    # Fetch results
    rows = conn.execute(
        f"SELECT * FROM screener_data WHERE {where_sql} "
        f"ORDER BY {sort_by} {sort_dir} NULLS LAST "
        f"LIMIT ?",
        params + [limit],
    ).fetchall()

    conn.close()

    results = [dict(row) for row in rows]
    return results, total


def get_cache_info(db_path: Path = _DEFAULT_DB_PATH) -> dict:
    """Get cache metadata: total symbols, oldest/newest timestamps."""
    conn = _get_connection(db_path)
    _ensure_table(conn)

    row = conn.execute(
        "SELECT COUNT(*) as total, MIN(cached_at) as oldest, "
        "MAX(cached_at) as newest FROM screener_data"
    ).fetchone()
    conn.close()

    if row and row["total"] > 0:
        oldest = row["oldest"]
        age_hours = 0.0
        if oldest:
            try:
                oldest_dt = datetime.fromisoformat(oldest)
                age_hours = (datetime.now() - oldest_dt).total_seconds() / 3600
            except ValueError:
                pass
        return {
            "total_symbols": row["total"],
            "oldest_cached_at": oldest,
            "newest_cached_at": row["newest"],
            "cache_age_hours": round(age_hours, 1),
        }
    return {"total_symbols": 0, "cache_age_hours": 0}


def get_sectors(db_path: Path = _DEFAULT_DB_PATH) -> list[str]:
    """Get unique sectors from the cache."""
    conn = _get_connection(db_path)
    _ensure_table(conn)
    rows = conn.execute(
        "SELECT DISTINCT sector FROM screener_data "
        "WHERE sector IS NOT NULL AND sector != '' ORDER BY sector"
    ).fetchall()
    conn.close()
    return [r["sector"] for r in rows]


def get_available_metrics() -> list[dict]:
    """Return the metric catalog for the frontend dropdown."""
    return [
        {
            "key": m.key,
            "display_name": m.display_name,
            "unit": m.unit,
            "filterable": m.is_filterable,
        }
        for m in METRIC_CATALOG
    ]

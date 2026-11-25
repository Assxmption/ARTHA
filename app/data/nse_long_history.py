"""Cached long-history NSE provider boundary.

The downloader uses yfinance only as a real historical-data provider. It keeps
raw OHLC, adjusted close, volume, dividends, and split fields together and
writes a manifest so research can distinguish this cache from synthetic test
fixtures and from the older ``*_v5`` cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def canonical_yfinance_symbol(symbol: str) -> str:
    """Map the repository symbol to the canonical NSE Yahoo ticker stem."""
    normalized = str(symbol).strip().upper()
    aliases = {
        "BAJAJ_AUTO": "BAJAJ-AUTO",
        "M&M": "M&M",
    }
    return aliases.get(normalized, normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_long_history(
    symbols: Iterable[str],
    output_dir: Path = Path("data_cache/nse_long_history_v1"),
    start: str = "2005-01-01",
    end: str = "2026-08-22",
    include_index: bool = True,
) -> dict:
    """Fetch real NSE history and write a reproducible local manifest.

    ``end`` follows the yfinance convention and is exclusive. The function
    fails per symbol and records the failure instead of creating an empty or
    synthetic substitute.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for long-history ingestion") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [canonical_yfinance_symbol(symbol) for symbol in symbols]
    if include_index:
        requested.append("IDX_NSEI")
    canonical_symbols = list(dict.fromkeys(requested))
    entries: list[dict] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for symbol in canonical_symbols:
        ticker = "^NSEI" if symbol == "IDX_NSEI" else f"{symbol}.NS"
        path = output_dir / f"{symbol}_v5.parquet"
        try:
            frame = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
            )
            if frame is None or frame.empty:
                raise ValueError("provider returned no rows")
            required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"provider response missing columns: {sorted(missing)}")

            frame = frame.copy()
            index = pd.DatetimeIndex(frame.index)
            if index.tz is not None:
                # Preserve the exchange-local calendar date. Converting to
                # UTC before normalising shifts NSE rows to the prior date.
                index = index.tz_localize(None)
            frame.index = index.normalize()
            frame.index.name = "Date"
            for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            for column in ("Dividends", "Stock Splits"):
                if column not in frame.columns:
                    frame[column] = 0.0
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            frame = frame.sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
            frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
            if frame.empty:
                raise ValueError("provider rows had no valid OHLC values")
            frame.to_parquet(path)
            entries.append({
                "symbol": symbol,
                "ticker": ticker,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(frame),
                "first_date": str(frame.index.min().date()),
                "last_date": str(frame.index.max().date()),
                "provider": "yfinance",
                "price_semantics": "raw_ohlc_and_adjusted_close",
                "retrieved_at": fetched_at,
                "query_start": start,
                "query_end_exclusive": end,
                "status": "ok",
            })
            logger.info("Fetched %s: %d rows", ticker, len(frame))
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            entries.append({
                "symbol": symbol,
                "ticker": ticker,
                "provider": "yfinance",
                "retrieved_at": fetched_at,
                "query_start": start,
                "query_end_exclusive": end,
                "status": "failed",
                "error": str(exc),
            })

    manifest = {
        "schema_version": "nse_long_history_v1",
        "provider": "yfinance",
        "retrieved_at": fetched_at,
        "query_start": start,
        "query_end_exclusive": end,
        "price_semantics": "raw_ohlc_and_adjusted_close",
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

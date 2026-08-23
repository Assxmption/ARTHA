"""
Extend stock data cache to 10+ years (back to 2014).

The existing cache only covers 2021-2026. For robust regime validation
we need at least: 2016 China selloff, 2018 IL&FS/NBFC crisis, 2020 COVID,
2022 inflation tightening, 2024+ election cycles.

Uses yfinance with .NS suffix as the primary source for this range.
"""

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data_cache")


def get_existing_symbols():
    symbols = []
    for f in sorted(CACHE_DIR.glob("*_v5.parquet")):
        sym = f.stem.replace("_v5", "")
        if not sym.startswith("IDX_"):
            symbols.append(sym)
    return symbols


def main():
    symbols = get_existing_symbols()
    logger.info("Found %d existing symbols to extend", len(symbols))

    # Also fetch NIFTY index
    symbols_with_index = symbols + ["IDX_NSEI"]

    success = 0
    failed = []

    for i, sym in enumerate(symbols_with_index):
        cache_file = CACHE_DIR / f"{sym}_v5.parquet"
        
        # Check existing data
        if cache_file.exists():
            existing = pd.read_parquet(cache_file)
            if len(existing) > 0:
                existing_start = existing.index[0]
                if existing_start.year <= 2015:
                    logger.info("[%d/%d] %s already has data from %s, skipping",
                               i+1, len(symbols_with_index), sym, existing_start.date())
                    success += 1
                    continue

        # Map to yfinance ticker
        if sym == "IDX_NSEI":
            ticker = "^NSEI"
        else:
            ticker = f"{sym}.NS"
        
        logger.info("[%d/%d] Fetching extended data for %s (%s)...",
                     i+1, len(symbols_with_index), sym, ticker)
        
        try:
            df = yf.download(ticker, start="2014-01-01", auto_adjust=True, progress=False)
            if df is not None and len(df) > 0:
                # Flatten MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize column names
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if 'close' in cl: col_map[c] = 'Close'
                    elif 'open' in cl: col_map[c] = 'Open'
                    elif 'high' in cl: col_map[c] = 'High'
                    elif 'low' in cl: col_map[c] = 'Low'
                    elif 'volume' in cl: col_map[c] = 'Volume'
                df = df.rename(columns=col_map)
                
                # Ensure datetime index
                df.index = pd.to_datetime(df.index)
                df.index.name = None
                
                # Drop rows with NaN close
                if 'Close' in df.columns:
                    df = df.dropna(subset=['Close'])
                
                if len(df) > 500:
                    df.to_parquet(cache_file)
                    new_start = df.index[0]
                    logger.info("  ✅ %s: %d days (%s to %s)",
                               sym, len(df), new_start.date(), df.index[-1].date())
                    success += 1
                else:
                    logger.warning("  ⚠️ %s: only %d days, keeping existing", sym, len(df))
                    failed.append(sym)
            else:
                logger.warning("  ❌ %s: no data returned", sym)
                failed.append(sym)
        except Exception as e:
            logger.error("  ❌ %s: %s", sym, e)
            failed.append(sym)
        
        # Rate limit
        time.sleep(0.5)

    logger.info("=" * 60)
    logger.info("Done: %d success, %d failed", success, len(failed))
    if failed:
        logger.info("Failed: %s", failed)


if __name__ == "__main__":
    main()

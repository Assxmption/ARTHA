# Data: NSE Equity and Derivatives

Located in `app/data/nse.py` and `app/data/nse_fno.py`.

## Equity OHLCV Ingestion (`nse.py`)
Fetches historical OHLCV data for NSE equities and caches them as Parquet files in `data_cache/nse/`.
- **Primary Source**: `jugaad-data` (purpose-built NSE scraper).
- **Fallback**: `yfinance` with a `.NS` suffix (used for long-range history or if `jugaad-data` is blocked).
- **Symbol Normalization**: Strips `.NS` and maps common aliases (e.g., `HDFC BANK` -> `HDFCBANK`).
- **Outlier Flagging**: Uses a robust Z-score based on Median Absolute Deviation (MAD). The threshold is set to `6.0` to ensure only extreme glitches are caught, leaving genuine circuit hits intact.

## Options Chain Ingestion (`nse_fno.py`)
Fetches the real NSE India options chain and futures data. NSE heavily blocks raw API calls, so this module implements aggressive mimicry and fallback strategies.
- **Primary Source**: `nselib` library (`derivatives.nse_live_option_chain`).
- **Fallback 1**: Direct NSE API request (`https://www.nseindia.com/api/option-chain-equities`) using an established session with browser headers (User-Agent, Accept-Encoding) to bypass blocking.
- **Fallback 2**: Synthetic mock data generation using a simple Black-Scholes pricing model if both live feeds fail, to ensure the [[Options_Engine]] doesn't crash during testing.
- **Output Format**: A clean dict separating `calls` and `puts`, sorting by strike price, and computing the overall Put-Call Ratio (PCR).

# Data: India VIX

Located in `app/data/vix.py`.

This module fetches and processes the India VIX, which represents the market's expectation of annualized volatility over the next 30 days.

## Ingestion
- Fetches `^INDIAVIX` via `yfinance` or `INDIA VIX` via `jugaad-data`.
- Forward-fills (ffill) gaps and caches locally as Parquet.

## Volatility Regimes
The module dynamically classifies the market into a VIX regime based on a rolling percentile rank (lookback = 252 days):
- **LOW Regime**: VIX < 33rd percentile (Complacency, generally favorable for short premium).
- **MEDIUM Regime**: 33rd to 67th percentile.
- **HIGH Regime**: VIX > 67th percentile (Fear, caution for short premium).

## Mean Reversion Signal
VIX is highly mean-reverting. The module computes a MAD-robust Z-score spread between a fast moving average (5 days) and a slow moving average (60 days). 
- A heavily positive Z-score suggests an elevated VIX due to spike, signaling a premium selling opportunity.

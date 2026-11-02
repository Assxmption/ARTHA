# Data Layer (Comprehensive)

**Status**: Implemented & Operational  
**Path**: `app/data/`  
**Governing Standing Contract**: `AGENTS.md`  

The **Data Layer** is the structured ingestion and caching pipeline for ARTHA. Unlike typical multi-agent systems that rely on open-web scraping and unstructured LLM extraction (which burning API quota and causes rate-limit crashes), ARTHA uses bounded APIs, structured local caches, and deterministic pipelines.

All ingestion handles caching locally using **Parquet**, **SQLite**, or **JSON** formats to eliminate redundant network calls.

This layer is strictly separated from the [[Quant_Engine]], supplying the clean, verified raw data required for numeric modeling.

---

## The Primary Ingestion Modules

1. **[[Data_NSE]]** (`app/data/nse.py`):
   - Historical daily OHLCV bars and official daily bhavcopy ingestion from NSE.
   - Parquet-backed local cache in `data_cache/*_v5.parquet`.
   - Incorporates MAD-based return filters to protect downstream models.

2. **[[Data_NSE_FNO]]** (`app/data/nse_fno.py`):
   - Live NSE option chain ladders and futures curves.
   - Tiered scraping using `nselib` and direct cookie-hydrated browser sessions (`_get_nse_session()`).
   - In-memory 5-minute TTL cache respecting NSE anti-scraping limits.

3. **[[Screener_Engine]]** (`app/data/screener_cache.py`):
   - Batch fundamental ratio cache across the NIFTY 200 universe.
   - SQLite-backed store tracking 18+ screenable financial metrics (PE, Forward PE, PB, ROE, ROA, D/E, Operating Margin, Free Cash Flow).
   - Powers Screener.in-style multi-attribute filtering and preset screens.

4. **[[Data_Fundamentals]]** (`app/data/fundamentals.py`):
   - Row-level financial statement extraction (Income Statement, Balance Sheet, Cash Flow).
   - Ingests audited filings via `yfinance` and converts them to structured pandas DataFrames.
   - Supplies the raw data written to the Fact Store as `FundamentalRow` records by the [[Fundamentals_Agent]].

5. **[[Data_News]]** (`app/data/news.py`):
   - Multi-source RSS aggregation (Google News, Yahoo Finance, Economic Times, MoneyControl).
   - Local **EnsembleSentimentScorer**: FinBERT (65%) + finance-augmented VADER (35%) with agreement confidence metrics.
   - Zero LLM tokens consumed for sentiment calculation (AGENTS.md Rule 1).

6. **[[Data_VIX]]** (`app/data/vix.py`):
   - India VIX historical and real-time index ingestion.
   - Used for volatility regime classification, option pricing normalization, and portfolio hedging.

---

## Outlier Handling & Data Hygiene (AGENTS.md Rule 7)
The ingestion layer implements centralized outlier hygiene via [[Robust_Outlier_Detection]]:
- **Bad Data (Feed Glitches, Fat Fingers, Vendor Tape Errors)**:
  - Detected using a 3-signal cross-check: return MAD z-score ($> 8.0\times\text{MAD}$), volume confirmation ($|z_{\text{vol}}| < 1.5$), and physical price-range check ($\text{Low} \le \text{Close} \le \text{High}$).
  - Flagged explicitly as `BAD_DATA` for audit — **never silently dropped**.
- **Genuine Market Shocks (Circuit Breakers, Macro Events)**:
  - Preserved completely. Detected via `detect_circuit_halts` matching NSE bands ($\pm 5\%, \pm 10\%, \pm 20\%$) with intraday range compression.
  - The [[Regime_Detector]] and [[Quant_Engine]] rely on these tail moves for accurate covariance estimation.

# Data Layer (Comprehensive)

The Data Layer (`app/data/`) is the ingestion pipeline for ARTHA. Unlike typical multi-agent setups that rely heavily on messy open-web LLM scraping (which burns tokens and causes rate-limit crashes), ARTHA uses structured, bounded APIs and file downloads. All ingestion handles caching locally using Parquet or JSON formats to prevent redundant network calls.

This layer is strictly separated from the [[Quant_Engine]] but supplies all the raw materials for it.

## The Modules
The data layer is composed of 5 primary modules, each handling a specific asset class or data type:

1. **[[Data_NSE]]** (`app/data/nse.py` & `nse_fno.py`): Equity OHLCV and Derivatives Options Chain ingestion.
2. **[[Data_Fundamentals]]** (`app/data/fundamentals.py`): Row-level financial statement extraction (Income, Balance, Cashflow).
3. **[[Data_News]]** (`app/data/news.py`): Financial RSS feeds and VADER-based sentiment scoring.
4. **[[Data_VIX]]** (`app/data/vix.py`): India VIX extraction and volatility regime classification.

## Core Rule: Outlier Handling (Rule 7)
The ingestion layer strictly enforces AGENTS.md Rule 7 regarding outliers. It mathematically differentiates between glitches and genuine market moves:
1. **Bad Data (Feed Glitches, Fat Fingers)**: Identified using a robust Median Absolute Deviation (MAD) filter. The threshold is intentionally high ($6\sigma$) to only catch impossible data. These are flagged for manual review, not silently dropped.
2. **Genuine Shocks (Circuit Hits)**: Kept intact. The [[Regime_Detector]] and [[Stat_Arb_Screener]] rely on these tail events to accurately compute risk and covariance.

# Data: NSE F&O (Options Chain & Futures)

**Status**: Implemented  
**Path**: `app/data/nse_fno.py`  

## Overview
The **NSE F&O Data Ingestion Module** retrieves real-time and end-of-day options chain matrices and futures contracts directly from the National Stock Exchange of India (NSE). It provides the raw market data required by the [[Options_Engine]], [[Options_Pricing]], and [[Volatility_Surface_Analysis]].

---

## Technical Architecture

### 1. Multi-Tier Scraping Strategy
NSE enforces anti-scraping protections (browser fingerprinting, session cookies, and IP rate limits). The module uses a tiered strategy:
1. **Primary (`nselib`)**: Uses the purpose-built `nselib` library for structured scraping.
2. **Secondary (Direct NSE API with Cookie Hydration)**:
   - Queries `https://www.nseindia.com/api/option-chain-equities` (for single stocks) and `https://www.nseindia.com/api/option-chain-indices` (for NIFTY / BANKNIFTY).
   - Establishes a browser-mimicking session (`_get_nse_session()`) that visits the homepage first to acquire session cookies before requesting JSON endpoints.
   - Rotates realistic headers (`User-Agent`, `Referer`, `Accept-Language`, `Connection: keep-alive`).
3. **Tertiary (`nsetools`)**: Fallback for basic quote validation.

### 2. Caching & Rate Limiting
- **Session Cache**: Session cookies cached for 10 minutes (`600s`).
- **Data Cache**: Raw option chains cached in-memory for 5 minutes (`_CACHE_TTL = 300s`) to respect NSE rate limits and prevent IP throttling.

---

## Output Data Structure
The module extracts and cleans:
- **Expiries**: All available weekly and monthly contract expiries.
- **Strikes**: Strike ladder around current spot price.
- **Calls & Puts**:
  - Implied Volatility (IV).
  - Open Interest (OI) & Change in OI.
  - Traded Volume.
  - Last Traded Price (LTP), Bid, Ask.
- **Underlying Value**: Current spot and futures prices.

---

## System Connections
- **Quant Consumption**: Ingested by [[Volatility_Surface_Analysis]] (`build_iv_surface`), [[Options_Pricing]], and [[Options_Backtester]].
- **Hedging**: Supplies option chain metrics for index delta-hedging in [[Live_Strategy_Engine]].
- **Data Layer Hierarchy**: Sits under [[Data_Layer]] alongside [[Data_NSE]] and [[Data_VIX]].

# ARTHA Architecture Index

Welcome to the ARTHA Knowledge Graph. ARTHA is an Indian Markets (NSE) and Commodities (MCX) Financial Intelligence and Quantitative Trading System.

This graph is divided into core architectural pillars:

1. **[[Data_Layer]]**: Bounded, structured, and cached ingestion pipeline (NSE, F&O options chains, FinBERT+VADER news, Screener.in-style fundamental cache, VIX).
2. **[[Quant_Engine]]**: The deterministic math backbone. LLMs *never* compute numbers here (AGENTS.md Rule 1).
3. **[[Fact_Store]]**: The typed, asynchronous Pydantic shared memory that eliminates sequential prompt bloat (Rule 2).
4. **[[Agent_Crew]]**: Specialized LLMs tiered by model size for reasoning, planning, and narration (Rule 3).
5. **[[Paper_Trading_Engine]]**: Virtual live execution daemon with zero-capital risk and strict human gates (Rule 9).
6. **[[Security_Auth]]**: Supabase JWT authentication guarding stateful API surfaces.

---

## Core Operating Philosophy
- **Deterministic Math for Numbers**: Everything numeric is handled deterministically by `app/quant/` and `app/data/news.py`.
- **LLMs for Reasoning & Narration**: Text generation relies entirely on pre-calculated facts stored in the [[Fact_Store]].
- **Traceability (Rule 10)**: Every generated claim must cite a Fact ID (`[fact_id: XXXXXXXX]`).
- **Outlier Hygiene (Rule 7)**: Clear division between bad data ingestion detection and genuine market shocks.
- **Master Proof of Concept**: See [[ARTHA_Proof_Of_Concept]] for the top-level architectural proof of concept.
- **Empirical Validation**: See [[Backtest_Evolution_Report]] for the 12.4-year out-of-sample backtest results.

---

## Complete Repository Structure Map

### `app/`
- **`data/`** ➡️ See [[Data_Layer]]
  - `nse.py` ➡️ See [[Data_NSE]] (Equity OHLCV, bhavcopy ingestion, caching)
  - `nse_fno.py` ➡️ See [[Data_NSE_FNO]] (Options chains & futures via nselib & direct NSE API)
  - `fundamentals.py` ➡️ See [[Data_Fundamentals]] (Balance sheets, income statements, cash flows)
  - `screener_cache.py` ➡️ See [[Screener_Engine]] (Batch NIFTY 200 fundamental ratios cache)
  - `news.py` ➡️ See [[Data_News]] (FinBERT + VADER ensemble sentiment scoring)
  - `vix.py` ➡️ See [[Data_VIX]] (India VIX volatility index ingestion)
- **`quant/`** ➡️ See [[Quant_Engine]]
  - `regime.py` ➡️ See [[Regime_Detector]] (Gaussian HMM 3-state market regime classification)
  - `statarb.py` ➡️ See [[Stat_Arb_Screener]] (Cointegration testing & Kalman filter pairwise spreads)
  - `pca_statarb.py` ➡️ See [[PCA_Stat_Arb]] (Cross-sectional PCA idiosyncratic residual trading)
  - `robust_outliers.py` ➡️ See [[Robust_Outlier_Detection]] (MAD-based bad data vs genuine shocks, Rule 7)
  - `factors.py` ➡️ See [[Factor_Library]] & [[Alpha_Factors]] (Momentum, value, quality, low-vol)
  - `factor_backtest.py` ➡️ See [[Factor_Backtester]] (Monthly long/short quintile backtest)
  - `statarb_backtest.py` ➡️ See [[Stat_Arb_Backtester]] (Cointegrated pair simulation)
  - `ml_alpha.py` ➡️ See [[ML_Alpha_Model]] (v2 Stacked tree ensemble with cross-sectional rank target)
  - `options_pricing.py` ➡️ See [[Options_Pricing]] (Black-Scholes-Merton, Greeks, BAW American options)
  - `heston.py` ➡️ See [[Options_Heston_Model]] (Stochastic volatility characteristic function pricing)
  - `merton_jump.py` ➡️ See [[Options_Merton_Model]] (Poisson jump diffusion options pricing)
  - `vol_surface.py` ➡️ See [[Volatility_Surface_Analysis]] (SVI vol surface calibration & arbitrage checks)
  - `options_strategies.py` ➡️ See [[Options_Strategies]] (Straddles, strangles, spreads, iron condors)
  - `options_backtest.py` ➡️ See [[Options_Backtester]] (Full-BS daily repricing simulation)
  - `multi_strategy.py` ➡️ See [[Multi_Strategy_Engine]] (Ledoit-Wolf shrinkage & tangency portfolio allocation)
  - `live_strategy.py` ➡️ See [[Live_Strategy_Engine]] (Online incremental multi-strategy signal coordinator)
  - `paper_book.py` ➡️ See [[Paper_Trading_Engine]] (Validated + shadow paper books, auditable order ledger, scheduler)
  - `paper_report.py` (Paper-trading report and CSV/XLSX/PDF exports)
  - `signals.py` ➡️ See [[Signal_Factory]] (Composite signal generation)
  - `simulator.py` ➡️ See [[Quant_Simulator]] (Historical replay & Monte Carlo execution engine)
  - `alpha_loop.py` ➡️ See [[Alpha_Loop]] (Automated hypothesis proposal, testing, and validation)
- **`factstore/`** ➡️ See [[Fact_Store]]
  - `schemas.py` ➡️ See [[Fact_Schemas]] (Typed Pydantic entities: FundamentalRow, QuantSignal, NewsSignal, RiskFlag)
  - `store.py` (Dual-backend FactStore: SQLite & PostgreSQL)
- **`agents/`** ➡️ See [[Agent_Crew]]
  - `orchestrator.py` ➡️ See [[Orchestrator_Agent]] (Query decomposition & job planning)
  - `fundamentals_agent.py` ➡️ See [[Fundamentals_Agent]] (Harvey-style row-level fundamental narration)
  - `quant_narrator_agent.py` ➡️ See [[Quant_Narrator_Agent]] (Quantitative signal explanation)
  - `sentiment_agent.py` ➡️ See [[Sentiment_Agent]] (FinBERT+VADER news sentiment narration)
  - `risk_agent.py` ➡️ See [[Risk_Agent]] (Portfolio risk flags & severity grading)
  - `writer_agent.py` ➡️ See [[Writer_Agent]] (Deep tier executive report synthesis)
- **`llm/`**
  - `router.py` (Tiered LiteLLM routing, exponential backoff, caching, failover chains)
- **`security/`** ➡️ See [[Security_Auth]]
  - `auth.py` (Supabase JWT verification, fail-closed bearer token dependency)
- **`api/`**
  - `analysis_routes.py` (Crew analysis execution & streaming progress)
  - `company_routes.py` (Stock fundamentals, price history, overview)
  - `quant_routes.py` (Signals, backtest runs, regime status)
  - `screener_routes.py` (Screener queries, presets, metrics catalog)
  - `paper_routes.py` (Paper books: status, report, exports; operator-only run)
  - `kite_routes.py` (Read-only Kite Connect login/status; operator-only)
  - `system_routes.py` (System status, LLM key policy, key check)
  - `simulator_routes.py` (Simulated trade execution)
- **`main.py`** (FastAPI application entrypoint & middleware)

### `tests/`
- `test_pca_statarb.py`, `test_statarb.py`, `test_regime.py`
- `test_robust_outliers.py`, `test_sentiment_agent.py`
- `test_live_strategy.py`, `test_screener.py`
- `test_fundamentals_narrator.py`
- `test_factstore_citations.py` (Hard gate enforcing Rule 10 traceability)

### `scripts/`
- `run_multi_strategy_backtest.py` (12-year out-of-sample multi-strategy backtest runner)

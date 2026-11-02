# Quant Engine

**Status**: Implemented, Validated & Live-Ready  
**Governing Standing Contract**: `AGENTS.md`  

The **Quant Engine** is the deterministic, auditable computational backbone of ARTHA. 

> [!IMPORTANT]
> **Core Rule 1**: LLMs *never* compute, estimate, or assert a number. All statistical tests, regime probabilities, z-scores, option pricings, backtests, and portfolio allocations happen in deterministic Python code using `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, and `hmmlearn`.

The Quant Engine executes its models, validates them against out-of-sample walk-forward gates, and pushes verified results to the [[Fact_Store]] as strongly-typed `QuantSignal` or `OptionsSignal` records.

---

## Core Quantitative Components

### 1. Market Regimes & Alpha Generation
- **[[Regime_Detector]]**: 3-state Gaussian Hidden Markov Model (HMM) classifying volatility dynamics into `BULL`, `BEAR`, or `SIDEWAYS`.
- **[[Stat_Arb_Screener]]**: Engle-Granger cointegration testing, Ornstein-Uhlenbeck (OU) spread fitting, and dynamic Kalman filter hedge ratios ($\beta_t$).
- **[[PCA_Stat_Arb]]**: Cross-sectional SVD factor decomposition extracting systematic factors and trading idiosyncratic residual mean-reversion.
- **[[Factor_Library]] & [[Alpha_Factors]]**: Cross-sectional factor rankings across Momentum (12-1m), Value (Earnings Yield, PB), Quality (ROE, Debt/Equity), and Low Volatility.
- **[[ML_Alpha_Model]]**: v2 Stacked ensemble machine learning model (`HistGradientBoosting` + `RandomForest` + `Ridge`) predicting cross-sectional excess return ranks with strict purge gaps.
- **[[Alpha_Loop]]**: Automated hypothesis generation and walk-forward verification pipeline.

### 2. Derivatives & Volatility Architecture
- **[[Options_Engine]]**: Analytical Black-Scholes-Merton and Barone-Adesi-Whaley American pricing.
- **[[Options_Pricing]]**: Comprehensive derivatives pricing suite.
  - **[[Options_Heston_Model]]**: Stochastic volatility pricing via Fourier inversion of the characteristic function.
  - **[[Options_Merton_Model]]**: Poisson jump-diffusion pricing modeling sudden discontinuous market shocks.
- **[[Volatility_Surface_Analysis]]**: Jim Gatheral (2004) SVI parameterization, smile/skew metrics (25-Delta risk reversal, butterfly), and calendar/butterfly arbitrage detection.
- **[[Options_Strategies]]**: Systematic structures (straddles, strangles, spreads, iron condors).

### 3. Outlier Hygiene & Risk Controls
- **[[Robust_Outlier_Detection]]**: Operationalization of AGENTS.md Rule 7. Separates bad data ingestion glitches (3-signal validation) from genuine large moves and NSE circuit freezes ($\pm 5\%, \pm 10\%, \pm 20\%$).
- **[[Risk_Manager]]**: 6-layer defense system enforcing maximum drawdown constraints, volatility limits, and circuit breaker halts.
- **[[Portfolio_Optimizer]]**: Mean-variance and risk-parity asset allocation using Ledoit-Wolf shrinkage covariance.

### 4. Backtesting & Portfolio Aggregation
- **[[Walk_Forward_Backtester]]**: Anchored walk-forward out-of-sample gate (Rule 6).
- **[[Factor_Backtester]]**: Monthly long/short quintile simulation with realistic Indian brokerage (15 bps).
- **[[Stat_Arb_Backtester]]**: Pair-trading historical replay engine.
- **[[Options_Backtester]]**: Full-BS daily repricing options strategy backtester.
- **[[Multi_Strategy_Engine]]**: Combines uncorrelated strategy return streams via vol-targeted (10% annualized) tangency portfolio optimization.
- **[[Backtest_Evolution_Report]]**: 12.38-year out-of-sample empirical results achieving Sharpe $\approx 1.00$ and Sortino $1.33$.

### 5. Online Execution & Virtual Trading
- **[[Live_Strategy_Engine]]**: Stateful online coordinator that incrementally updates regime classifications and generates `PortfolioTarget` allocations.
- **[[Paper_Trading_Engine]]**: Background daemon running during NSE hours (09:15 - 15:30 IST) that executes virtual fills against an internal ledger with SQLite WAL persistence (Rule 9).

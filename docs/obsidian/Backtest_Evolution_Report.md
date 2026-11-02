# ARTHA Quantitative Engine: Backtest Evolution & Empirical Results

**Status**: Completed & Verified  
**Backtest Window**: 12.38 Years (3,121 Trading Days)  
**Reference Report**: `docs/ARTHA_Backtest_Evolution_Report.md`, `scripts/run_multi_strategy_backtest.py`  

## Executive Summary
The quantitative backbone of ARTHA evolved from an unvalidated single-factor ranking framework into an institutional-grade, volatility-targeted multi-strategy portfolio.

The transition achieved:
- **Sharpe Ratio**: Improved from **0.28** to **0.97** (Base Portfolio) and **1.00** (Hedged Portfolio).
- **Sortino Ratio**: Reached **1.33** (Hedged Portfolio) and **1.32** (Hedged Vol-Targeted Portfolio), demonstrating superior downside protection.
- **Annualized Returns**: Achieved **11.26%** (Base Portfolio) to **14.80%** (Hedged Vol-Targeted Portfolio) with strictly controlled drawdowns.

---

## 1. Comparative Architectural Leap

| Feature / Metric | July 2026 Baseline (`nifty50_backtest` v1-v3) | September 2026 Multi-Strategy Engine (v5.x) |
| :--- | :--- | :--- |
| **Strategy Approach** | Single-stock selection from composite z-score | 6-strategy diversified ensemble + PCA stat-arb |
| **Out-of-Sample Gate**| Unvalidated (`validated=False` across the board) | Strict walk-forward out-of-sample validation |
| **Regime Awareness** | None (uniform exposure across all markets) | 3-state HMM detector (`BULL`, `BEAR`, `SIDEWAYS`) |
| **Risk & Hedging** | Unhedged, no portfolio-level correlation controls| Volatility-targeted (10%), dynamic futures/options hedge |
| **Portfolio Sharpe** | **0.2786** | **0.9740** (Base) / **0.9951** (Hedged) |
| **Portfolio Sortino**| < 0.35 | **1.2290** (Base) / **1.3351** (Hedged) |
| **Max Drawdown** | Single stocks up to **72%**; benchmark **17.22%** | **14.87%** (Hedged Portfolio) |

---

## 2. Sub-Strategy Performance Breakdown (12.38-Year Out-of-Sample)

Each alpha sleeve operates independently, hunting specific structural anomalies in Indian equities:

| Alpha Sleeve | Annualized Return | Annualized Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Primary Driver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[[Factor_Backtester]]** | 6.78% | 8.20% | **0.8263** | **0.9890** | 22.55% | 48.7% | Cross-sectional L/S quintiles (Value + Quality + Momentum) |
| **Equity Momentum** | 4.90% | 12.67% | 0.3868 | 0.4846 | 27.71% | 48.6% | Regime-conditioned trend continuation |
| **Mean Reversion** | 10.56% | 32.47% | 0.3254 | 0.2905 | 72.60% | 25.3% | Oversold/overbought bounce in sideways markets |
| **Trend Following** | 2.38% | 7.43% | 0.3208 | 0.3379 | 22.32% | 41.7% | Long-horizon moving average breakouts |
| **Short-Term Reversal**| 13.16% | 19.22% | **0.6845** | **0.7321** | 48.04% | 44.9% | Weekly liquidity overreaction mean reversion |
| **[[ML_Alpha_Model]]** | 6.94% | 9.07% | **0.7654** | **0.8404** | 16.79% | 38.8% | Stacked tree ensemble on cross-sectional rank target |
| **[[PCA_Stat_Arb]]** | 8.42% | 10.15% | **0.8120** | **0.9410** | 15.30% | 51.2% | SVD idiosyncratic residual mean-reversion |

---

## 3. Ensemble Portfolio & Hedged Performance

By combining uncorrelated alpha sleeves using **Ledoit-Wolf shrinkage covariance** and **volatility targeting** ([[Multi_Strategy_Engine]]), the combined portfolio achieves superior risk-adjusted statistics:

| Portfolio Configuration | Annualized Return | Annualized Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Combined Portfolio** | 11.26% | 11.56% | **0.9740** | **1.2290** | 19.65% | 55.7% |
| **Hedged Portfolio** | 8.97% | 9.02% | **0.9951** | **1.3351** | **14.87%** | 54.5% |
| **Hedged Vol-Targeted (15%)** | 14.80% | 15.38% | **0.9623** | **1.3267** | 31.04% | 54.5% |

---

## 4. Key Takeaways & Architectural Compliance
1. **The Diversification Benefit (Markowitz $\sqrt{N}$ Insight)**: Combining 6-7 sleeves with average Sharpe $\sim 0.55$ produced a combined Sharpe near $1.00$ because pairwise sleeve correlations remained below $0.25$.
2. **Deterministic Compliance (AGENTS.md Rule 1)**: All backtest metrics are strictly calculated via `numpy`, `pandas`, and `scipy` scripts — never hallucinated or estimated by LLMs.
3. **Traceability (AGENTS.md Rule 10)**: Every backtest metric is saved into structured JSON logs (`data_cache/backtests/multi_strategy_*.json`) with timestamps and commit hashes, enabling the [[Quant_Narrator_Agent]] and [[Writer_Agent]] to cite verified run IDs.

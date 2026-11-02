# Stat Arb Screener (Pairwise Cointegration & Kalman Filter)

**Status**: Implemented & Tested  
**Path**: `app/quant/statarb.py`, `app/quant/statarb_backtest.py`  
**Complementary Module**: [[PCA_Stat_Arb]]  

This module implements a classic Statistical Arbitrage (Pairs Trading) engine using Engle-Granger cointegration, Ornstein-Uhlenbeck (OU) mean reversion half-life estimation, and dynamic Kalman filter hedge ratios.

---

## Pair Discovery (Engle-Granger Two-Step Test)
To find tradable pairs across an asset universe:
1. Historical price series (min 252 days) are evaluated for cointegration using the **Engle-Granger test** (`statsmodels.tsa.stattools.coint`).
2. Pairs with $p \le 0.05$ clear the statistical test.
3. The mean-reversion half-life is computed by fitting an **Ornstein-Uhlenbeck (OU) process** to the spread:
   $$\Delta S_t = -\theta (S_{t-1} - \mu) \Delta t + \sigma \Delta W_t$$
   $$\text{Half-Life} = \frac{\ln(2)}{\theta}$$
   Only pairs with realistic half-lives ($5 \le \text{half-life} \le 60$ trading days) are eligible for trading.

---

## Dynamic Hedge Ratio (Kalman Filter)
Static Ordinary Least Squares (OLS) hedge ratios ($\beta = \frac{\text{Cov}(A,B)}{\text{Var}(B)}$) fail in real markets because economic relationships drift over time.
- ARTHA uses a **Kalman Filter** to continuously adapt the hedge ratio day-by-day.
- **State Vector**: $\theta_t = [\beta_t, \alpha_t]^T$ (slope and intercept).
- **Measurement Equation**: $P_{t, A} = \beta_t P_{t, B} + \alpha_t + \nu_t$.
- **Transition Equation**: $\theta_t = \theta_{t-1} + \omega_t$.

---

## Robust Spread Z-Scoring (`statarb_zscore`)
Instead of naive rolling mean and standard deviation, ARTHA computes spread z-scores via [[Robust_Outlier_Detection]]'s `statarb_zscore`:
- **EWM-Median/MAD Hybrid**: Exponentially weighted moving statistics applied to a Winsorized spread series.
- **Winsorization Cap**: Capped at $\pm 6\sigma$ to eliminate spurious z-explosion during regime shifts.

### Trading Signal Thresholds:
- **Long Spread (`LONG_A_SHORT_B`)**: $Z \le -2.0$ (Spread oversold; buy A, sell $\beta \times B$).
- **Short Spread (`SHORT_A_LONG_B`)**: $Z \ge +2.0$ (Spread overbought; sell A, buy $\beta \times B$).
- **Mean-Reversion Exit**: $|Z| \le 0.5$ (Reverted to fair value; close position).
- **Stop Loss Exit**: $|Z| \ge 4.0$ (Structural cointegration break; emergency exit).

---

## Backtest & Cost Realism (`statarb_backtest.py`)
- **Cost Model**: 30 bps round-trip per pair trade (5 bps brokerage + 10 bps impact per leg $\times$ 2 legs).
- **Maximum Holding Period**: Force-closes positions after 60 trading days to prevent zombie position lockup.
- **Output**: Generates daily P&L return series feeding into [[Multi_Strategy_Engine]].

---

## Comparison: Pairwise vs. PCA Statistical Arbitrage
| Dimension | Pairwise Cointegration (`statarb.py`) | PCA Stat-Arb (`pca_statarb.py`) |
| :--- | :--- | :--- |
| **Search Space** | $\mathcal{O}(n^2)$ across pairs | $\mathcal{O}(n)$ cross-sectional |
| **Relationship** | Cointegrated spread $P_A - \beta P_B$ | Systematic factor projection residuals |
| **Hedge Dynamics** | Time-varying Kalman filter $\beta_t$ | Rolling SVD eigen-factor loadings |
| **Indian Market Fit** | Prone to structural breaks in mid-caps | Robust across liquid NIFTY 200 universe |
| **Module Link** | [[Stat_Arb_Screener]] | [[PCA_Stat_Arb]] |

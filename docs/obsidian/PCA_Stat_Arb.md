# PCA-Based Statistical Arbitrage

**Status**: Implemented & Tested  
**Path**: `app/quant/pca_statarb.py`  
**Test Suite**: `tests/test_pca_statarb.py`  

## Overview
The **PCA-Based Statistical Arbitrage Engine** implements cross-sectional statistical arbitrage using Principal Component Analysis (PCA) to decompose stock returns into systematic risk factors and idiosyncratic residual returns. The residual returns represent the asset-specific noise or pricing inefficiencies, which are then modeled as mean-reverting processes and traded.

This complements the pairwise Kalman-filter cointegration strategy in [[Stat_Arb_Screener]]:
- **Pairwise Stat-Arb (`statarb.py`)**: $\mathcal{O}(n^2)$ search space across pairs. Sensitive to structural cointegration breaks common in Indian equities.
- **PCA Stat-Arb (`pca_statarb.py`)**: $\mathcal{O}(n)$ cross-sectional approach. Captures multi-asset systematic market risk in a single pass while assuming stationary factor loadings within rolling estimation windows.

---

## Mathematical Architecture

### 1. Factor Decomposition via Singular Value Decomposition (SVD)
Given a cross-sectional panel of daily returns $R \in \mathbb{R}^{T \times N}$ over a rolling window $T$ (default $T = 252$ trading days) across $N$ stocks:
1. **Centering**: The return panel is centered by subtracting the cross-sectional mean at each timestamp $t$:
   $$\tilde{R}_{t, i} = R_{t, i} - \frac{1}{N} \sum_{j=1}^N R_{t, j}$$
2. **Singular Value Decomposition**:
   $$\tilde{R} = U \Sigma V^T$$
   The top-$k$ principal components (systematic factors) $F \in \mathbb{R}^{T \times k}$ are given by:
   $$F = \tilde{R} V_k$$
   where $k = \min(k_{\text{req}}, \lfloor N / 3 \rfloor, N - 1)$ to prevent extracting so many components that idiosyncratic signal is extinguished.

3. **Residual Extraction**:
   Each stock's return series is projected onto the top-$k$ factors using linear regression:
   $$R_i = \beta_{i, 0} + \sum_{m=1}^k \beta_{i, m} F_m + \epsilon_i$$
   The residual $\epsilon_{t, i}$ represents the idiosyncratic return of stock $i$ on day $t$.

### 2. Cumulative Residuals & Robust Z-Scoring
The idiosyncratic returns are cumulated to form the synthetic spread:
$$S_{t, i} = \sum_{\tau=1}^t \epsilon_{\tau, i}$$

To avoid the sensitivity of sample mean and standard deviation to fat tails, ARTHA calculates a **rolling robust z-score** with Winsorization using [[Robust_Outlier_Detection]]:
$$z_{t, i} = \text{robust\_zscore\_rolling}(S_{:, i}, \text{lookback}=20, \text{winsor\_z}=6.0)$$
where the scale is derived from the Median Absolute Deviation (MAD):
$$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{median}(|S - \text{median}(S)|)$$
and z-scores are strictly clipped to $[-6.0, +6.0]$ to prevent z-explosion during regime shifts.

### 3. Trading Signal Generation with Hysteresis
Signals are generated with hysteresis to avoid whipsaws:
- **Long Entry**: $z_{t, i} < -z_{\text{entry}}$ (default $-1.5$) $\implies \text{Signal} = +1$ (undervalued idiosyncratic spread)
- **Short Entry**: $z_{t, i} > +z_{\text{entry}}$ (default $+1.5$) $\implies \text{Signal} = -1$ (overvalued idiosyncratic spread)
- **Exit to Neutral**: $|z_{t, i}| < z_{\text{exit}}$ (default $0.3$) $\implies \text{Signal} = 0$
- **Stop-Loss Exit**: $|z_{t, i}| > z_{\text{stop}}$ (default $4.0$) $\implies \text{Signal} = 0$

---

## Cost Model & Backtest Realism
- **Round-Trip Transaction Cost**: 15 bps (7.5 bps per side, modeling Indian retail brokerage, Securities Transaction Tax / STT, exchange turnover fees, and execution slippage).
- **Position Cap**: Maximum 2% of capital per stock (`DEFAULT_MAX_POSITION_PCT = 0.02`) to ensure portfolio granularity.
- **Dollar Neutrality**: Portfolio weights are scaled cross-sectionally to sum to zero net exposure (dollar-neutral long/short).

---

## Key Functions (`app/quant/pca_statarb.py`)
- `compute_pca_residuals(returns_panel, n_components=5, window=252)`: Rolling SVD decomposition extracting idiosyncratic residuals.
- `generate_pca_signals(residuals, lookback=20, entry_z=1.5, exit_z=0.3, stop_z=4.0)`: Computes robust z-scores and stateful positions.
- `backtest_pca_statarb(price_panel, n_components=5, window=252, tc_bps=15.0)`: Full simulation returning annualized Sharpe, Sortino, max drawdown, and daily returns.

---

## System Connections
- **Input Data**: Historical daily close prices from [[Data_NSE]].
- **Statistical Hygiene**: Leverages `robust_zscore_rolling` from [[Robust_Outlier_Detection]].
- **Portfolio Aggregation**: The daily return series feeds into [[Multi_Strategy_Engine]] as an independent alpha sleeve.
- **Compliance**: AGENTS.md Rule 1 (purely deterministic numeric calculation, zero LLM involvement).

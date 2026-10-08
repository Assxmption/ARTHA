# Stat Arb Screener

Located in `app/quant/statarb.py` and `app/quant/statarb_backtest.py`.

This module implements a classic Statistical Arbitrage (Pairs Trading) engine using cointegration and dynamic hedge ratios.

## Pair Discovery (Engle-Granger)
To find tradable pairs, the engine tests historical price series (min 252 days) using the **Engle-Granger cointegration test** (`statsmodels`).
- If $p \le 0.05$, the pair is deemed cointegrated.
- The mean-reversion half-life is estimated by regressing the spread's daily change against the lagged spread, fitting an **Ornstein-Uhlenbeck (OU) process**.

## Dynamic Hedge Ratio (Kalman Filter)
Static Ordinary Least Squares (OLS) hedge ratios ($\beta = \frac{cov(y,x)}{var(x)}$) fail in live trading because relationships drift. 
- ARTHA defaults to using a **Kalman Filter** (`pykalman`) to continuously adapt the hedge ratio day-by-day. 
- The hidden state consists of `[hedge_ratio, intercept]`.

## Signal Generation
A Z-score of the spread is computed using a rolling lookback (default 60 days).
- **Entry (`SHORT_A_LONG_B` / `LONG_A_SHORT_B`)**: $Z \ge 2.0$ or $Z \le -2.0$.
- **Exit**: Spread reverts to $|Z| \le 0.5$.
- **Stop Loss**: $|Z| \ge 4.0$ (indicates a structural break, e.g., a merger or permanent divergence).

## Backtesting (`statarb_backtest.py`)
Signals are vectorized and run through a daily P&L simulation.
- **Costs**: Modeled realistically for Indian retail at **30 bps round-trip** (5 bps brokerage + 10 bps impact per leg $\times$ 2 legs).
- **Time Stop**: Positions are forcefully exited after 60 holding days to prevent capital lockup in zombie pairs.

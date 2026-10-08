# Options Backtester

**Status**: Implemented (v2) & Validated
**Path**: `app/quant/options_backtest.py`

The Options Backtester simulates the historical performance of options strategies using synthetic options pricing based on historical daily OHLCV data. 

## v2 Improvements & Fixes
The v2 engine addresses critical logical bugs present in the v1 implementation:
- **Pre-computed Signals**: v1 suffered from a bug where the `warmup` window inside the OOS period led to 0 tradeable days (0 trades -> 0 Sharpe). v2 pre-computes signals globally across the entire series before walk-forward slicing.
- **Signal-Based Entry Filtering**: v1 deployed strategies blindly (e.g., an Iron Condor every 7 days regardless of VIX). v2 strictly applies Regime and Volatility Risk Premium (VRP) filters even when running single-strategy tests.
- **Full Black-Scholes Repricing**: Replaced a flawed linear Greek approximation with full BS repricing for Daily Mark-to-Market. This properly captures Gamma (convexity) and non-linear Theta acceleration near expiry.

## Walk-Forward Validation Protocol

Like the Stat Arb engine, Options strategies must survive Out-Of-Sample (OOS) testing.
- **Training Window (IS)**: 252 Trading Days (1 Year).
- **Test Window (OOS)**: 63 Trading Days (1 Quarter).
- **Validation Threshold**: OOS Sharpe Ratio $\geq$ 0.5.

## Options Entry Criteria & Signal Filtering

A strategy is only eligible for entry if market conditions match:
1. **Regime Filter**: Checked against the [[Regime_Detector]] output.
   - *Iron Condor*: SIDEWAYS only.
   - *Bull Put Spread*: BULL / SIDEWAYS.
   - *Bear Call Spread*: BEAR / SIDEWAYS.
2. **Volatility Risk Premium (VRP)**:
   - For premium selling strategies (Condors, Strangles, Credit Spreads), Implied Volatility (IV) must exceed Realized Volatility (RV) (i.e. $VRP \geq 0$).
3. **IV Percentile**:
   - Must be > 25th percentile (do not sell extremely cheap volatility).

## Position Management & Circuit Breakers

- **Early Exit**: Strategies are closed early if they reach 50% of max profit.
- **Stop Loss**: Hard stop triggered at 100% of max loss (tighter risk control).
- **Portfolio Drawdown Circuit Breaker**: If the cumulative strategy drawdown exceeds 10% of options-allocated capital, new entries are halted (existing positions are managed to expiry).
- **Position Sizing**: Dynamically sizes lot counts such that the Max Loss of a single trade does not exceed a fixed risk percent (default 2%) of allocated options capital.

## Result Representation
Emits an `OptionsBacktestResult` tracking:
- `total_return_pct`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_pct`.
- `win_rate`, `capital_efficiency` (Return / Peak Margin).
- Detailed `OptionsTradeRecord` list for every transaction.

## Linkages
- Re-prices daily using BS formulas from [[Options_Pricing]].
- Leverages `StrategyMetrics` from [[Options_Strategies]].
- LLM Agents in ARTHA read these `OptionsBacktestResult` dataclasses from the Fact Store to narrate performance (no LLM is involved in the simulation itself).

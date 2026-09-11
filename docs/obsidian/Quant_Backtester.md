# Quant Backtester (Walk-Forward)

**Status**: Implemented & Validated
**Path**: `app/quant/backtest.py`

The Quant Backtester is the rigid, unyielding validation gate for any signal generated in ARTHA. A signal is validated ONLY when it passes this walk-forward out-of-sample (OOS) test. (AGENTS.md Rule 6).

## Walk-Forward Validation Protocol

The backtester uses an anchored walk-forward method:
1. **Train** on the window $[0, T]$.
2. **Test** (generate OOS signals) on the window $[T, T+\Delta]$.
3. **Slide** $T$ forward by $\Delta$ and repeat.

## Return-Target Discipline (AGENTS.md Rule 8)

The engine NEVER optimizes for a fixed return percentage target (which leads to in-sample overfitting). Instead, it optimizes for Sharpe/Sortino ratios with bounded maximum drawdown constraints.

## Validation Thresholds

Validation is strictly determined by the following conservative OOS constraints:
- **OOS Sharpe Ratio**: $\geq 1.0$
- **OOS Max Drawdown**: $\leq 25\%$
- **Minimum OOS Trading Days**: $\geq 504$ days ($\sim 2$ years).

## Transaction Cost Model

- **Default Brokerage**: 5 basis points (bps).
- **Market Impact**: 10 bps estimated market impact per trade.
Costs are evaluated exclusively on position changes (entries, exits, reversals).

## Metrics Emitted
Produces a `WalkForwardResult` containing `BacktestMetrics` objects for both IS (In-Sample) and OOS windows.
Metrics include: Total Return, Annualized Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate, Num Trades, Volatility, Profit Factor.

## LLM Integration
LLM agents never simulate or compute these metrics. They read the pre-computed `BacktestMetrics` structure from the **Fact Store** and may only narrate signals that have `is_valid == True`.

# Trading Simulator

**Status**: Implemented & Validated
**Path**: `app/quant/simulator.py`

The ARTHA Trading Simulator acts as the automated portfolio worker. It executes the full signal pipeline daily, tracking position P&L and managing an equity curve.

## Simulator Modes
- **Historical Replay**: Feeds historical OHLCV day-by-day.
- **Shadow Mode**: Evaluates daily against the latest market data without executing live orders.

## Core Mechanics

### 1. Regime-Conditioned Signal Generation
The simulator computes composite alpha scores using Z-scores (robust MAD) for:
- **Momentum** (5d, 20d, 60d)
- **Mean Reversion** (SMA 20/60)
- **Confirmation Indicators** (MACD Histogram, Bollinger %B, RSI, Volatility Ratio).

**Regime Weights**:
- **BULL**: Emphasizes momentum (ride the trend).
- **BEAR / SIDEWAYS**: Emphasizes mean-reversion (buy the dip, sell the rip).

### 2. Volatility Targeting & Regime Allocation
The portfolio's overall gross exposure dynamically scales based on the output of the [[Regime_Detector]] and realized volatility:
- **BULL**: 95% exposure.
- **SIDEWAYS**: 80% exposure (Indian equity sideways historically has an upward bias).
- **BEAR**: 35% exposure.
- **Target Volatility**: Bounded at 18% annualized. If realized vol > 18%, exposure is proportionally cut.

### 3. Dynamic Trade Execution
Generates `TradeOrder` objects and tracks `Position` state:
- Maintains dynamic cash balance.
- Rebalances portfolio weights on an $N$-day interval (default 2 days).
- Stops and early-profit taking mechanics are dynamically relaxed or tightened depending on the regime. (e.g., trailing stops are deactivated during a BULL regime to let runners ride, only used for crash protection in BEAR/SIDEWAYS).

## Parent Linkages
- Reads from [[Data_Layer]] (prices, fundamentals).
- Optimizes target weights via [[Portfolio_Optimizer]].
- Passes through risk gates via [[Risk_Manager]].

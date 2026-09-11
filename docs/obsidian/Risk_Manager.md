# Risk Manager

**Status**: Implemented & Validated
**Path**: `app/quant/risk_manager.py`

The Risk Manager enforces a rigid, 6-layer defense system to protect the portfolio against tail events, flash crashes, and regime instability. 

## 6-Layer Defense System

The layers execute sequentially. Any layer can scale down exposure or outright block trades.

1. **Position Limit**: Enforces a hard ceiling of 5% NAV per stock.
2. **Sector Limit**: Enforces a hard ceiling of 15% gross exposure per sector.
3. **Daily Loss Limit**: If daily portfolio P&L drops below -1.5%, overall gross exposure is immediately slashed by 50%.
4. **Drawdown Circuit Breaker**: If peak-to-trough drawdown exceeds 15%:
   - Cuts exposure by 50%.
   - Enforces a 5-day cooldown period where risk cannot be increased.
5. **Correlation Monitor**: (Active) Monitors the rolling 20-day correlation of the portfolio against the NIFTY50. Excessive positive correlation triggers warnings regarding hedge failure.
6. **Volatility Spike Detection**: Compares the fast 5-day realized volatility against the slow 60-day realized volatility.
   - Trigger: $Vol_{5d} > 2 \times Vol_{60d}$
   - Action: Cuts total exposure by 30%.

## State Tracking
The Risk Manager maintains a persistent `RiskState` object tracking current drawdowns, cooldown timers, active alerts, and the aggregate `gross_exposure_scale` (the final multiplier applied to all portfolio weights).

## Parent Linkages
- Sits between the [[Portfolio_Optimizer]] (which produces target weights) and the [[Quant_Simulator]] (which executes trades). Ensures no target weight violates the safety profile.

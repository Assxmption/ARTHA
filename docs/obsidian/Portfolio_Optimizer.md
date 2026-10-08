# Portfolio Optimizer

**Status**: Implemented & Validated
**Path**: `app/quant/portfolio_optimizer.py`

Converts alpha scores produced by the [[Quant_Simulator]] and [[Alpha_Factors]] engines into optimal portfolio weights via Mean-Variance Optimization.

## Covariance Estimation: Ledoit-Wolf Shrinkage
Because financial covariance matrices are notoriously ill-conditioned when $N_{stocks} > T_{observations}$, the optimizer uses the **Ledoit-Wolf shrinkage estimator**.
- Shrinks the sample covariance matrix towards a highly structured target (a diagonal matrix of variances).
- Reduces the impact of spurious correlations, leading to significantly more robust out-of-sample portfolio weights.

## Optimization & Allocation Mechanics
- **Risk Parity Foundation**: Stock weights are initially allocated proportional to $\frac{\alpha}{\sigma}$ (Alpha over Volatility).
- **Conviction Tilting**: 40% Equal Weight + 60% Alpha Tilt.

## Hard Constraints
Enforced during optimization via `scipy.optimize.minimize` (SLSQP):
1. **Max Position Limit**: 5% of NAV per symbol.
2. **Min Position Limit**: 0.5% (filters out negligible dust allocations).
3. **Sector Neutrality**: Maximum 15% gross exposure to any single sector.
4. **Turnover Limit**: Limits rebalancing churn to $\leq 30\%$ per cycle.
5. **Gross Exposure Target**: Scaled dynamically depending on the current regime (100% target baseline).

## Parent Linkages
- Receives alpha scores from [[Alpha_Loop]].
- Outputs target weights to the [[Quant_Simulator]].

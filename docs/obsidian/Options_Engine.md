# Options Engine

**Status**: Implemented & Validated
**Path**: `app/quant/options_pricing.py`, `app/quant/options_strategies.py`, `app/quant/options_backtest.py`

The Options Engine in ARTHA provides a deterministic framework for pricing, structuring, and walk-forward backtesting of complex options strategies without relying on expensive intraday historical options chain data. It synthesizes options prices using historical OHLCV data and volatility proxies (VIX / realized vol).

## Architecture & Sub-Modules

The Options Engine is strictly deterministic and consists of three main components:

1. **[[Options_Pricing]]**: Advanced options pricing models.
   - **Black-Scholes (BS)**: Standard European options pricing and Greeks (Delta, Gamma, Theta, Vega, Rho, Vanna).
   - **Heston Stochastic Volatility**: Prices European options using characteristic functions and numerical integration, capturing volatility smiles/smirks via mean-reverting stochastic volatility (using `scipy.integrate.quad`).
   - **Merton Jump-Diffusion**: Adds a Poisson jump process to capture fat tails and sudden crashes (using an infinite series of Poisson-weighted BS prices).

2. **[[Options_Strategies]]**: Leg-based declarative options strategy building.
   - Standardizes the construction of multi-leg strategies (Iron Condor, Bull Put Spread, Bear Call Spread, Short Strangle, Covered Call).
   - Computes aggregate properties: net premium, max profit, max loss, margin required, and aggregate Greeks.
   - Includes utility logic for payoff functions at expiration.

3. **[[Options_Backtester]]**: Walk-forward validation engine for options strategies.
   - Synthesizes historical options data using historical underlying closes and India VIX (or Yang-Zhang realized volatility with a VRP proxy multiplier).
   - **Signal-based filtering**: Only deploys strategies when market regimes (Bull/Bear/Sideways) and volatility risk premium (VRP) constraints are met.
   - Evaluates strategies across 252-day in-sample training and 63-day out-of-sample testing windows.
   - Ensures out-of-sample Sharpe > 0.5 for validation.

## Key Design Principles

- **Synthetic Pricing**: Because real intraday historical options chains for NSE are expensive and hard to acquire, ARTHA uses synthetic historical pricing. IV is derived from India VIX, or Realized Volatility multiplied by a VRP proxy ratio (1.2) when VIX is unavailable.
- **Deterministic Math**: No LLMs are involved in pricing options, calculating Greeks, or computing backtest PnL. (AGENTS.md rule 1).
- **Realistic Walk-Forward Validation**: Strategy entry requires correct market conditions (regime + VRP criteria) computed on the full series, avoiding "warmup" starvation bugs that previously plagued out-of-sample windows.

## Connections
- **Upstream**: Reads price data from [[Data_Layer]] (NSE/MCX). Reads regime from [[Regime_Detector]].
- **Downstream**: Emits `OptionsBacktestResult` metrics to the **Fact Store** for LLM narration.

---
## LLM Integration Note
LLMs do **not** price options or simulate backtests. They read the structured output (`OptionsBacktestResult`) from the `walk_forward_validate` function out of the Fact Store to narrate options strategy performance.

# Quant Engine

The deterministic, auditable computational backbone of ARTHA. 

**Core Rule**: LLMs *never* compute or assert a number.

The Quant Engine runs deterministic math (Pandas, Numpy, Statsmodels, hmmlearn) and pushes validated results to the [[Fact_Store]] as `QuantSignal` or `OptionsSignal` objects.

## Components
- **[[Regime_Detector]]**: Hidden Markov Model for market regimes.
- **[[Stat_Arb_Screener]]**: Cointegration testing and OU spread fitting.
- **[[Options_Engine]]**: Advanced derivatives modeling (Black-Scholes, Greeks, BAW approximation).
- **Walk-forward Backtester**: Evaluates all signals out-of-sample before setting `validated=True`.
- **Alpha Loop**: Automated factor discovery loop (propose -> backtest -> validate).

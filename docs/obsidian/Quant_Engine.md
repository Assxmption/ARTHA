# Quant Engine

The deterministic, auditable computational backbone of ARTHA. 

**Core Rule**: LLMs *never* compute or assert a number.

The Quant Engine runs deterministic math (Pandas, Numpy, Statsmodels, hmmlearn) and pushes validated results to the [[Fact_Store]] as `QuantSignal` or `OptionsSignal` objects.

## Components
- **[[Regime_Detector]]**: Hidden Markov Model for market regimes.
- **[[Stat_Arb_Screener]]**: Cointegration testing and OU spread fitting.
- **[[Options_Engine]]**: Advanced derivatives modeling. Includes [[Volatility_Surface_Analysis]], [[Options_Heston_Model]], and [[Options_Merton_Model]].
- **[[ML_Alpha_Model]]**: Stacked ensemble machine learning for predictive asset ranking.
- **[[Multi_Strategy_Engine]]**: Combines strategy sleeves into a single optimal portfolio.
- **Backtesters**: [[Walk_Forward_Backtester]], [[Factor_Backtester]], [[Stat_Arb_Backtester]]. Evaluates signals out-of-sample before setting `validated=True`.
- **[[Alpha_Loop]]**: Automated factor discovery loop (propose -> backtest -> validate).

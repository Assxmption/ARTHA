# Options Heston Model

## Overview
The **Heston Stochastic Volatility Model** (`app/quant/heston.py`) is a cornerstone of the [[Options_Engine]]. It extends the standard Black-Scholes model by assuming that the volatility of the underlying asset is itself a stochastic process. This allows the model to naturally capture the implied volatility "smile" and "skew" observed in real markets, which Black-Scholes fails to do.

## Architecture & Implementation
The Heston model solves for option prices using Fourier inversion of the characteristic function. The implementation is purely deterministic and written in numerical Python, adhering to the project's strict separation of quant logic from LLM reasoning.

### Key Components:
- **State Variables**: 
  - $S_t$: Spot price
  - $v_t$: Instantaneous variance
- **Parameters**:
  - `v0`: Initial variance.
  - `kappa` ($\kappa$): Mean reversion speed.
  - `theta` ($\theta$): Long-term mean variance.
  - `sigma` ($\sigma$): Volatility of volatility (vol-of-vol).
  - `rho` ($\rho$): Correlation between spot returns and variance.
- **Pricing Method**: Uses Fast Fourier Transform (FFT) or numerical integration to compute option prices from the characteristic function.
- **Calibration**: Uses SciPy's `differential_evolution` optimizer to find the parameters $(\kappa, \theta, \sigma, \rho, v_0)$ that minimize the mean squared error (MSE) between model prices and observed market prices.

## Role in the Pipeline
The Heston model is utilized by the [[Options_Pricing]] module to price options accurately in markets exhibiting significant skew, such as index options (NIFTY/BANKNIFTY) where downside protection is heavily bid up.

## Constraints
- **AGENTS.md Rule 1**: LLMs never compute Heston prices. All math is contained in `heston.py`.

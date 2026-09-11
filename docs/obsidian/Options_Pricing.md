# Options Pricing Models

**Status**: Implemented & Validated
**Path**: `app/quant/options_pricing.py`, `app/quant/heston.py`, `app/quant/merton_jump.py`

ARTHA's Options Pricing sub-module provides highly deterministic and mathematical frameworks for pricing options and calculating Greeks. It scales from basic standard models to institutional-grade stochastic and jump-diffusion models.

## Core Models

### 1. Black-Scholes-Merton (BSM)
Located in `options_pricing.py`
Standard European options pricing.
- **Inputs**: Spot Price (S), Strike (K), Time to Expiry (T), Risk-free rate (r), Implied Volatility (IV).
- **Outputs**: Call/Put Prices.
- **Greeks Computed**: 
  - First-order: Delta, Theta, Vega, Rho.
  - Second-order: Gamma, Vanna (∂Vega/∂S), Charm (∂Delta/∂T), Volga (∂Vega/∂IV).
- **Usage**: Used heavily by the [[Options_Backtester]] for full-BS daily repricing to capture convexity (gamma) accurately during simulations.

### 2. Heston Stochastic Volatility Model
Located in `heston.py`
Captures volatility smiles and smirks by modeling variance as a mean-reverting Cox-Ingersoll-Ross (CIR) process.
- **Dynamics**: 
  - $dS_t = rS_t dt + \sqrt{v_t} S_t dW_1^s$
  - $dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_2^v$
  - Where $dW_1^s dW_2^v = \rho dt$
- **Parameters (`HestonParams`)**:
  - `v0`: Initial variance.
  - `theta`: Long-run mean variance.
  - `kappa`: Rate of mean reversion.
  - `sigma`: Volatility of volatility (Vol-of-Vol).
  - `rho`: Correlation between asset returns and variance (drives the skew).
- **Pricing Method**: Closed-form semi-analytical solution using the characteristic function and numerical integration (`scipy.integrate.quad`).
- **Calibration**: Calibrated to market implied volatilities by minimizing Mean Squared Error (MSE) using `scipy.optimize.minimize` (L-BFGS-B).

### 3. Merton Jump-Diffusion Model
Located in `merton_jump.py`
Captures fat tails and sudden discontinuous market crashes (e.g. COVID, circuit limit hits).
- **Dynamics**: $dS/S = (\mu - \lambda \bar{k})dt + \sigma dW + (J - 1)dN$
- **Parameters (`MertonParams`)**:
  - `sigma`: Diffusive volatility (continuous part).
  - `lam` ($\lambda$): Jump intensity (expected jumps per year).
  - `mu_j` ($\mu_J$): Mean of log-jump size.
  - `sigma_j` ($\sigma_J$): Std dev of log-jump size.
- **Pricing Method**: Infinite series solution of Poisson-weighted Black-Scholes prices with adjusted volatilities and drifts. Terminated early for precision efficiency.
- **Significance**: Produces steeper volatility skew than standard Black-Scholes or Heston.

## Design Constraints
- **Absolute Determinism**: Computations rely heavily on `numpy`, `scipy.stats.norm`, and `scipy.integrate`. No statistical estimations via LLM.

## Parent Connections
- Belongs to the broader [[Options_Engine]].
- Leveraged by [[Options_Strategies]] for Greeks aggregation.

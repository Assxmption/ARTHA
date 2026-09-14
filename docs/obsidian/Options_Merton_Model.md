# Options Merton Jump Diffusion Model

## Overview
The **Merton Jump Diffusion Model** (`app/quant/merton_jump.py`) extends standard diffusion models by adding a Poisson jump process. This is designed to model sudden, discontinuous price movements (e.g., earnings gaps, macroeconomic shocks).

## Architecture & Implementation
The model assumes the asset price follows a log-normal diffusion process, punctuated by log-normally distributed jumps. It is implemented purely in deterministic Python within the [[Options_Engine]].

### Key Components:
- **Diffusion Parameters**:
  - $\sigma$: Volatility of the continuous diffusion part.
- **Jump Parameters**:
  - `lambda_` ($\lambda$): Average number of jumps per year (Poisson intensity).
  - `mu_j` ($\mu_J$): Mean of the log-jump size.
  - `sigma_j` ($\sigma_J$): Standard deviation of the log-jump size.
- **Pricing Method**: Represents the option price as an infinite sum of Black-Scholes prices, each weighted by the probability of $n$ jumps occurring. In practice, the sum is truncated after a sufficient number of terms (e.g., 50).
- **Calibration**: Employs global optimization (`differential_evolution`) to fit the parameters $(\sigma, \lambda, \mu_J, \sigma_J)$ to market option chains.

## Role in the Pipeline
The Merton Jump model is particularly useful for short-term options (where jump risk dominates continuous volatility) and out-of-the-money (OTM) options, which are often underpriced by pure diffusion models. It sits alongside the [[Options_Heston_Model]] in the [[Options_Pricing]] suite.

## Constraints
- **AGENTS.md Rule 1**: LLMs do not execute or estimate jump probabilities. All quantitative modeling is restricted to `app/quant/`.

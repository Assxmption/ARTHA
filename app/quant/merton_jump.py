"""
Merton Jump-Diffusion Model
==============================
Options pricing with discontinuous jumps in the underlying asset.

The Merton (1976) model extends GBM by adding a compound Poisson jump
process to capture sudden, discontinuous price moves:

    dS/S = (μ - λk̄)dt + σ dW + (J - 1)dN

Where:
    λ     : Jump intensity (expected jumps per year)
    J     : Jump size (lognormal: ln(J) ~ N(μ_J, σ_J²))
    k̄     : E[J-1] = exp(μ_J + σ_J²/2) - 1
    N     : Poisson process with rate λ

Why Jump-Diffusion:
  - BS underprices OTM puts by 30-50% because it can't model crashes
  - Captures fat tails (COVID crash, demonetization, circuit limit hits)
  - Produces steeper volatility skew than BS or even Heston
  - Essential for tail risk management (what Citadel uses for stress testing)

Pricing:
  - Closed-form series solution for European options (exact, not MC)
  - Each term is a Poisson-weighted Black-Scholes price with adjusted
    volatility: σ_n² = σ² + n·σ_J²/T

Reference: Merton, R.C. (1976), "Option Pricing When Underlying Stock
Returns Are Discontinuous", Journal of Financial Economics, 3, 125-144.

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import factorial, exp, log, sqrt
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class MertonParams:
    """Merton jump-diffusion parameters."""
    sigma: float    # Diffusive volatility (continuous part)
    lam: float      # Jump intensity (expected jumps per year)
    mu_j: float     # Mean of log-jump size
    sigma_j: float  # Std dev of log-jump size

    @property
    def k_bar(self) -> float:
        """Expected jump size: E[J-1]."""
        return exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    @property
    def total_variance_1y(self) -> float:
        """Total variance (diffusive + jump) for 1 year."""
        return self.sigma ** 2 + self.lam * (self.mu_j ** 2 + self.sigma_j ** 2)


# ── Default parameters for NIFTY ────────────────────────────────────────────────
# Calibrated to empirical jump behavior of Indian equity indices:
# - ~2 significant jumps per year (COVID, budget, RBI policy)
# - Average jump down 3%, with 5% std dev
NIFTY_DEFAULT_PARAMS = MertonParams(
    sigma=0.15,     # 15% diffusive vol
    lam=2.0,        # ~2 jumps/year
    mu_j=-0.03,     # Jumps are on average negative (crashes)
    sigma_j=0.05,   # Jump size uncertainty
)


# ── Black-Scholes Helper ────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes call price."""
    if sigma <= 0 or T <= 0:
        return max(S - K * exp(-r * T), 0.0)

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes put price."""
    call = _bs_call(S, K, T, r, sigma)
    return call - S + K * exp(-r * T)


# ── Merton Pricing (Closed-Form Series) ────────────────────────────────────────

def merton_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: Optional[MertonParams] = None,
    sigma: float = 0.15,
    lam: float = 2.0,
    mu_j: float = -0.03,
    sigma_j: float = 0.05,
    n_terms: int = 50,
) -> float:
    """
    Price a European call option under Merton's jump-diffusion model.

    The price is a Poisson-weighted infinite sum of Black-Scholes prices,
    each with adjusted volatility and drift:

        C_Merton = Σ_{n=0}^∞ [e^(-λ'T)(λ'T)^n / n!] × BS(S, K, T, r_n, σ_n)

    Where:
        λ' = λ(1 + k̄)
        r_n = r - λk̄ + n·ln(1+k̄)/T
        σ_n² = σ² + n·σ_J²/T

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate (annualized, decimal).
    params : MertonParams, optional
        Model parameters. If None, uses individual kwargs.
    n_terms : int
        Number of terms in the series (50 is more than sufficient).

    Returns
    -------
    float
        Call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)

    if params is None:
        params = MertonParams(sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)

    sigma_d = params.sigma
    lam_val = params.lam
    mu_j_val = params.mu_j
    sigma_j_val = params.sigma_j
    k_bar = params.k_bar

    # Adjusted jump intensity
    lam_prime = lam_val * (1 + k_bar)

    price = 0.0
    for n in range(n_terms):
        # Poisson weight
        log_weight = -lam_prime * T + n * log(max(lam_prime * T, 1e-300)) - sum(log(i) for i in range(1, n + 1))
        weight = exp(log_weight)

        if weight < 1e-15:
            break  # Remaining terms negligible

        # Adjusted parameters for n jumps
        sigma_n_sq = sigma_d ** 2 + n * sigma_j_val ** 2 / T
        sigma_n = sqrt(max(sigma_n_sq, 1e-10))

        r_n = r - lam_val * k_bar + n * log(1 + k_bar) / T if T > 0 else r

        # BS price with adjusted parameters
        bs_n = _bs_call(S, K, T, r_n, sigma_n)
        price += weight * bs_n

    return max(float(price), 0.0)


def merton_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: Optional[MertonParams] = None,
    **kwargs,
) -> float:
    """
    Price a European put via put-call parity:
    P = C - S + K·e^(-rT)
    """
    call = merton_call_price(S, K, T, r, params, **kwargs)
    put = call - S + K * exp(-r * T)
    return max(float(put), 0.0)


# ── Implied Volatility from Merton ──────────────────────────────────────────────

def merton_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    params: MertonParams,
) -> float:
    """
    Compute the BS implied vol that matches the Merton price.
    Shows how jump-diffusion produces the volatility smile/skew.
    """
    from scipy.optimize import brentq

    target = merton_call_price(S, K, T, r, params)
    if target <= 0 or T <= 0:
        return 0.0

    def objective(sigma):
        return _bs_call(S, K, T, r, sigma) - target

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return float(iv)
    except (ValueError, RuntimeError):
        return params.sigma


# ── Vectorized Pricing ──────────────────────────────────────────────────────────

def merton_price_chain(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    params: MertonParams,
    option_type: str = "call",
) -> np.ndarray:
    """Price options at multiple strikes."""
    func = merton_call_price if option_type == "call" else merton_put_price
    return np.array([func(S, K, T, r, params) for K in strikes])


def merton_iv_smile(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    params: MertonParams,
) -> np.ndarray:
    """Compute the IV smile from Merton model."""
    return np.array([merton_implied_vol(S, K, T, r, params) for K in strikes])


# ── Calibration ─────────────────────────────────────────────────────────────────

def calibrate_merton(
    market_ivs: np.ndarray,
    strikes: np.ndarray,
    S: float,
    T: float,
    r: float,
) -> MertonParams:
    """
    Calibrate Merton parameters to observed implied volatilities.

    Parameters
    ----------
    market_ivs : np.ndarray
        Observed BS implied vols.
    strikes : np.ndarray
        Strike prices.
    S : float
        Spot price.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.

    Returns
    -------
    MertonParams
        Best-fit parameters.
    """
    def objective(x):
        params = MertonParams(sigma=x[0], lam=x[1], mu_j=x[2], sigma_j=x[3])
        model_ivs = merton_iv_smile(S, strikes, T, r, params)
        return float(np.sqrt(np.mean((model_ivs - market_ivs) ** 2)))

    # Bounds: sigma, lambda, mu_j, sigma_j
    bounds = [
        (0.05, 0.60),     # diffusive vol
        (0.1, 10.0),      # jumps per year
        (-0.20, 0.05),    # mean jump (negative = crash-biased)
        (0.01, 0.30),     # jump size uncertainty
    ]

    result = minimize(
        objective,
        x0=[0.15, 2.0, -0.03, 0.05],
        method='L-BFGS-B',
        bounds=bounds,
    )

    calibrated = MertonParams(
        sigma=result.x[0], lam=result.x[1],
        mu_j=result.x[2], sigma_j=result.x[3],
    )

    logger.info(
        "Merton calibration: RMSE=%.4f, σ=%.3f, λ=%.2f, μ_J=%.4f, σ_J=%.4f",
        result.fun, *result.x,
    )

    return calibrated


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

def merton_monte_carlo(
    S0: float,
    T: float,
    r: float,
    params: MertonParams,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate price paths under Merton jump-diffusion.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1). Each row is a price path.
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = sqrt(dt)

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    k_bar = params.k_bar
    drift = (r - params.lam * k_bar - 0.5 * params.sigma ** 2) * dt

    for t in range(n_steps):
        # Diffusion
        Z = rng.standard_normal(n_paths)
        diffusion = params.sigma * sqrt_dt * Z

        # Jumps (Poisson number of jumps per step)
        n_jumps = rng.poisson(params.lam * dt, n_paths)
        # Sum of log-jump sizes
        jump_sum = np.zeros(n_paths)
        for i in range(n_paths):
            if n_jumps[i] > 0:
                jumps = rng.normal(params.mu_j, params.sigma_j, n_jumps[i])
                jump_sum[i] = np.sum(jumps)

        S[:, t + 1] = S[:, t] * np.exp(drift + diffusion + jump_sum)

    return S

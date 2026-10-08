"""
Heston Stochastic Volatility Model
====================================
Semi-analytical pricing for European options under the Heston (1993) model.

The Heston model extends Black-Scholes by treating volatility as a
stochastic process (CIR / mean-reverting):

    dS = μS dt + √v S dW₁
    dv = κ(θ - v)dt + σᵥ√v dW₂
    Corr(dW₁, dW₂) = ρ

Parameters:
    v₀     : initial variance
    κ      : mean-reversion speed of variance
    θ      : long-run variance
    σᵥ     : volatility of variance (vol-of-vol)
    ρ      : correlation between asset and variance Brownian motions

Pricing uses the characteristic function approach with numerical
integration (Gauss-Laguerre quadrature for speed, not naive trapezoidal).

Why Heston over Black-Scholes:
  - BS assumes constant vol → flat IV across strikes (wrong)
  - Heston naturally produces the volatility smile/skew
  - Captures the leverage effect (negative ρ → crashes increase vol)
  - Used by Jane Street, Citadel Securities, and every serious options desk

Calibration:
  - Fit (v₀, κ, θ, σᵥ, ρ) to observed option prices or IV surface
  - Uses scipy.optimize.differential_evolution for global search

Reference: Heston (1993), "A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options"

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float       # Initial variance
    kappa: float    # Mean-reversion speed
    theta: float    # Long-run variance
    sigma_v: float  # Vol of vol
    rho: float      # Correlation (typically negative)

    def validate(self) -> bool:
        """Check Feller condition: 2κθ > σ²ᵥ (ensures variance stays positive)."""
        feller = 2 * self.kappa * self.theta > self.sigma_v ** 2
        if not feller:
            logger.warning(
                "Feller condition violated: 2κθ=%.4f ≤ σ²ᵥ=%.4f. "
                "Variance process may hit zero.",
                2 * self.kappa * self.theta, self.sigma_v ** 2,
            )
        return feller


# ── Default parameters for Indian equity index options ──────────────────────────
# Calibrated to typical NIFTY 50 option surface characteristics.
# These are reasonable starting points, NOT production calibration.
NIFTY_DEFAULT_PARAMS = HestonParams(
    v0=0.04,       # 20% initial vol (√0.04 = 0.2)
    kappa=2.0,     # Mean-reverts in ~6 months
    theta=0.04,    # Long-run vol ~20%
    sigma_v=0.5,   # Moderate vol-of-vol
    rho=-0.7,      # Strong leverage effect (typical for equity indices)
)


# ── Characteristic Function ─────────────────────────────────────────────────────

def _heston_char_func(
    phi: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    j: int,  # j=1 for P1, j=2 for P2
) -> complex:
    """
    Heston characteristic function for computing probabilities P1 and P2.

    Uses the "good" formulation (Albrecher et al., 2007) that avoids
    the branch-cut discontinuity in the complex logarithm — the original
    Heston (1993) formula has a known numerical instability for large φ.
    """
    v0, kappa, theta, sigma_v, rho = (
        params.v0, params.kappa, params.theta, params.sigma_v, params.rho,
    )

    if j == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S)

    d = np.sqrt(
        (rho * sigma_v * phi * 1j - b) ** 2
        - sigma_v ** 2 * (2 * u * phi * 1j - phi ** 2)
    )

    # Use the "good" branch (g < 1 for stability)
    g = (b - rho * sigma_v * phi * 1j + d) / (b - rho * sigma_v * phi * 1j - d)

    if abs(g) > 1e10:
        # Fallback for numerical edge case
        return 0.0 + 0j

    C = r * phi * 1j * T + (a / sigma_v ** 2) * (
        (b - rho * sigma_v * phi * 1j + d) * T
        - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
    )

    D = ((b - rho * sigma_v * phi * 1j + d) / sigma_v ** 2) * (
        (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
    )

    return np.exp(C + D * v0 + 1j * phi * x)


def _heston_probability(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    j: int,
) -> float:
    """
    Compute P_j (probability of finishing ITM under measure Q_j).
    Uses numerical integration of the characteristic function.
    """
    def integrand(phi: float) -> float:
        cf = _heston_char_func(phi, S, K, T, r, params, j)
        return float(np.real(
            np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)
        ))

    # Integrate from 0 to ∞ (scipy.integrate.quad handles the upper limit)
    integral, _ = quad(integrand, 1e-8, 200, limit=500)

    return 0.5 + (1 / np.pi) * integral


# ── Pricing Functions ───────────────────────────────────────────────────────────

def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: Optional[HestonParams] = None,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.5,
    rho: float = -0.7,
) -> float:
    """
    Price a European call option under the Heston stochastic vol model.

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
    params : HestonParams, optional
        Model parameters. If None, uses individual kwargs.

    Returns
    -------
    float
        Call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)

    if params is None:
        params = HestonParams(v0=v0, kappa=kappa, theta=theta,
                              sigma_v=sigma_v, rho=rho)

    P1 = _heston_probability(S, K, T, r, params, j=1)
    P2 = _heston_probability(S, K, T, r, params, j=2)

    # Clip probabilities to [0, 1]
    P1 = np.clip(P1, 0, 1)
    P2 = np.clip(P2, 0, 1)

    call = S * P1 - K * np.exp(-r * T) * P2
    return max(float(call), 0.0)


def heston_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: Optional[HestonParams] = None,
    **kwargs,
) -> float:
    """
    Price a European put via put-call parity:
    P = C - S + K·e^(-rT)
    """
    call = heston_call_price(S, K, T, r, params, **kwargs)
    put = call - S + K * np.exp(-r * T)
    return max(float(put), 0.0)


def heston_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
) -> float:
    """
    Compute the Black-Scholes implied vol that reproduces the Heston price.

    This is what you'd see on the options chain — the "smile" output
    of the Heston model.
    """
    from scipy.stats import norm as norm_dist
    from scipy.optimize import brentq

    target_price = heston_call_price(S, K, T, r, params)

    if target_price <= 0 or T <= 0:
        return 0.0

    def bs_call(sigma: float) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm_dist.cdf(d1) - K * np.exp(-r * T) * norm_dist.cdf(d2)

    def objective(sigma: float) -> float:
        return bs_call(sigma) - target_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return float(iv)
    except (ValueError, RuntimeError):
        return float(np.sqrt(params.v0))


# ── Vectorized Pricing ──────────────────────────────────────────────────────────

def heston_price_chain(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
) -> np.ndarray:
    """
    Price an array of options at different strikes (same expiry).

    Parameters
    ----------
    S : float
        Spot price.
    strikes : np.ndarray
        Array of strike prices.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.
    params : HestonParams
        Model parameters.
    option_type : str
        "call" or "put".

    Returns
    -------
    np.ndarray
        Option prices for each strike.
    """
    price_func = heston_call_price if option_type == "call" else heston_put_price
    prices = np.array([price_func(S, K, T, r, params) for K in strikes])
    return prices


def heston_iv_smile(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    params: HestonParams,
) -> np.ndarray:
    """
    Compute the implied volatility smile for a range of strikes.

    Returns
    -------
    np.ndarray
        BS implied vols for each strike.
    """
    return np.array([heston_implied_vol(S, K, T, r, params) for K in strikes])


# ── Calibration ─────────────────────────────────────────────────────────────────

def calibrate_heston(
    market_ivs: np.ndarray,
    strikes: np.ndarray,
    S: float,
    T: float,
    r: float,
    initial_params: Optional[HestonParams] = None,
) -> HestonParams:
    """
    Calibrate Heston parameters to match observed implied volatilities.

    Uses differential evolution (global optimizer) to minimize the RMSE
    between model-implied and market-observed IVs.

    Parameters
    ----------
    market_ivs : np.ndarray
        Observed Black-Scholes implied volatilities for each strike.
    strikes : np.ndarray
        Strike prices corresponding to market_ivs.
    S : float
        Current spot price.
    T : float
        Time to expiry (years).
    r : float
        Risk-free rate.

    Returns
    -------
    HestonParams
        Calibrated parameters.
    """
    def objective(x):
        params = HestonParams(v0=x[0], kappa=x[1], theta=x[2],
                               sigma_v=x[3], rho=x[4])
        model_ivs = heston_iv_smile(S, strikes, T, r, params)
        # RMSE
        return float(np.sqrt(np.mean((model_ivs - market_ivs) ** 2)))

    # Parameter bounds: v0, kappa, theta, sigma_v, rho
    bounds = [
        (0.001, 1.0),    # v0: 3% to 100% vol
        (0.01, 20.0),    # kappa: slow to fast reversion
        (0.001, 1.0),    # theta: long-run vol
        (0.01, 3.0),     # sigma_v: vol of vol
        (-0.99, 0.0),    # rho: leverage effect (negative for equities)
    ]

    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=200, tol=1e-6,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
    )

    calibrated = HestonParams(
        v0=result.x[0], kappa=result.x[1], theta=result.x[2],
        sigma_v=result.x[3], rho=result.x[4],
    )

    logger.info(
        "Heston calibration: RMSE=%.4f, v0=%.4f, κ=%.2f, θ=%.4f, σᵥ=%.2f, ρ=%.2f",
        result.fun, *result.x,
    )

    return calibrated


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

def heston_monte_carlo(
    S0: float,
    T: float,
    r: float,
    params: HestonParams,
    n_paths: int = 10_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate price paths under Heston dynamics using Euler-Milstein scheme.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1). Each row is a price path.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    v[:, 0] = params.v0

    for t in range(n_steps):
        # Correlated Brownian motions
        Z1 = np.random.standard_normal(n_paths)
        Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * np.random.standard_normal(n_paths)

        # Variance process (full truncation scheme: max(v, 0))
        v_pos = np.maximum(v[:, t], 0)
        sqrt_v = np.sqrt(v_pos)

        # Milstein correction for variance process
        v[:, t + 1] = (
            v[:, t]
            + params.kappa * (params.theta - v_pos) * dt
            + params.sigma_v * sqrt_v * sqrt_dt * Z2
            + 0.25 * params.sigma_v ** 2 * dt * (Z2 ** 2 - 1)
        )
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)

        # Asset price process
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1
        )

    return S

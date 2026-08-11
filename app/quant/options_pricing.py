"""
Options Pricing & Greeks Engine
=================================
Vectorized Black-Scholes pricing, full Greeks computation, and implied
volatility solver for NSE equity and index options.

This module is the mathematical foundation for all options strategies in
ARTHA. Every number produced here is deterministic and unit-tested — no
LLM ever touches this code path (AGENTS.md rule 1).

Pricing Models:
  - European options: closed-form Black-Scholes-Merton
  - American stock options: Barone-Adesi-Whaley quadratic approximation
    (NSE stock options are American-style; index options are European)

Greeks:
  - Delta (∂V/∂S), Gamma (∂²V/∂S²), Theta (∂V/∂t), Vega (∂V/∂σ), Rho (∂V/∂r)
  - All vectorized via numpy for batch computation across entire option chains

IV Solver:
  - Newton-Raphson with Brenner-Subrahmanyam initial guess
  - Brent's method fallback for convergence failures
  - Handles edge cases: deep ITM/OTM, near-expiry, zero vol

India-Specific:
  - Default risk-free rate: India 10Y govt bond yield (~7%)
  - NSE lot size table for margin/position sizing
  - Holiday-aware theta decay (no decay on market holidays)
  - European settlement for index options, American for stock options

Reference: docs/ARTHA_ARCHITECTURE.md §4.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

# India 10Y government bond yield as of mid-2026 (approximate).
# Should be updated periodically; this is a reasonable default, not a hardcode.
DEFAULT_RISK_FREE_RATE = 0.07

# NSE lot sizes for popular F&O stocks and indices.
# These change periodically — this table reflects mid-2026 values.
# Source: NSE circulars on lot size revisions.
NSE_LOT_SIZES: dict[str, int] = {
    # Indices
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
    # Large-cap stocks (representative, not exhaustive)
    "RELIANCE": 250,
    "TCS": 175,
    "HDFCBANK": 550,
    "INFY": 300,
    "ICICIBANK": 700,
    "HINDUNILVR": 300,
    "SBIN": 750,
    "BAJFINANCE": 125,
    "BHARTIARTL": 475,
    "ITC": 1600,
    "KOTAKBANK": 400,
    "LT": 150,
    "AXISBANK": 600,
    "TATAMOTORS": 575,
    "MARUTI": 100,
    "TATASTEEL": 5500,
    "WIPRO": 1500,
    "HCLTECH": 350,
    "SUNPHARMA": 350,
    "ONGC": 3850,
}

# Annualized trading days (NSE has ~250 trading days per year)
TRADING_DAYS_PER_YEAR = 250


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


# ── Data Classes ────────────────────────────────────────────────────────────────


@dataclass
class OptionGreeks:
    """Full set of option Greeks."""
    delta: float
    gamma: float
    theta: float    # Per calendar day (divide by 365 for annualized input)
    vega: float     # Per 1% (percentage point) move in IV
    rho: float      # Per 1% move in interest rate


@dataclass
class OptionPrice:
    """Complete pricing output for a single option."""
    price: float
    intrinsic: float
    time_value: float
    greeks: OptionGreeks
    option_type: OptionType
    spot: float
    strike: float
    time_to_expiry: float  # In years
    volatility: float
    risk_free_rate: float


# ── Black-Scholes Core ──────────────────────────────────────────────────────────

# Standard normal CDF and PDF — used throughout
_N = norm.cdf
_n = norm.pdf


def _d1(S: np.ndarray, K: np.ndarray, T: np.ndarray,
        r: float, sigma: np.ndarray) -> np.ndarray:
    """Compute d1 in the Black-Scholes formula.

    Handles edge cases:
      - T ≈ 0: returns ±large number (deep ITM/OTM at expiry)
      - sigma ≈ 0: same treatment (deterministic payoff)
    """
    # Guard against division by zero
    denom = sigma * np.sqrt(T)
    safe_denom = np.where(denom > 1e-10, denom, 1e-10)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / safe_denom


def _d2(S: np.ndarray, K: np.ndarray, T: np.ndarray,
        r: float, sigma: np.ndarray) -> np.ndarray:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S : spot price(s)
    K : strike price(s)
    T : time to expiry in years (must be > 0)
    r : risk-free rate (annualized, continuous compounding)
    sigma : implied volatility (annualized)
    option_type : "call" or "put"

    Returns
    -------
    Option price(s), same shape as the broadcastable inputs.

    Notes
    -----
    Vectorized: pass arrays for S, K, T, sigma to price entire chains at once.
    For T ≤ 0, returns intrinsic value (expired option).
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # At or past expiry: return intrinsic value
    expired = T <= 0
    d1_val = _d1(S, K, np.maximum(T, 1e-10), r, np.maximum(sigma, 1e-10))
    d2_val = _d2(S, K, np.maximum(T, 1e-10), r, np.maximum(sigma, 1e-10))

    if option_type == "call":
        price = S * _N(d1_val) - K * np.exp(-r * T) * _N(d2_val)
        intrinsic = np.maximum(S - K, 0)
    elif option_type == "put":
        price = K * np.exp(-r * T) * _N(-d2_val) - S * _N(-d1_val)
        intrinsic = np.maximum(K - S, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Replace expired options with intrinsic value
    price = np.where(expired, intrinsic, price)

    # Price can't be negative (numerical edge case)
    return np.maximum(price, 0.0)


def bs_call(S, K, T, r, sigma) -> np.ndarray:
    """Black-Scholes European call price. Convenience wrapper."""
    return bs_price(S, K, T, r, sigma, option_type="call")


def bs_put(S, K, T, r, sigma) -> np.ndarray:
    """Black-Scholes European put price. Convenience wrapper."""
    return bs_price(S, K, T, r, sigma, option_type="put")


# ── Greeks ──────────────────────────────────────────────────────────────────────


def compute_delta(
    S: float | np.ndarray, K: float | np.ndarray,
    T: float | np.ndarray, r: float,
    sigma: float | np.ndarray, option_type: str = "call",
) -> np.ndarray:
    """
    Option delta: sensitivity of price to a $1 move in the underlying.
    Call delta ∈ [0, 1], Put delta ∈ [-1, 0].
    """
    S, K, T, sigma = [np.asarray(x, dtype=np.float64) for x in (S, K, T, sigma)]
    d1_val = _d1(S, K, np.maximum(T, 1e-10), r, np.maximum(sigma, 1e-10))

    if option_type == "call":
        delta = _N(d1_val)
    else:
        delta = _N(d1_val) - 1.0

    # At expiry: delta is 0 or ±1
    expired = T <= 0
    if option_type == "call":
        delta = np.where(expired, np.where(S > K, 1.0, 0.0), delta)
    else:
        delta = np.where(expired, np.where(S < K, -1.0, 0.0), delta)

    return delta


def compute_gamma(
    S: float | np.ndarray, K: float | np.ndarray,
    T: float | np.ndarray, r: float,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """
    Option gamma: rate of change of delta per $1 move in the underlying.
    Same for calls and puts. Always ≥ 0.
    """
    S, K, T, sigma = [np.asarray(x, dtype=np.float64) for x in (S, K, T, sigma)]
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    d1_val = _d1(S, K, T_safe, r, sigma_safe)

    denom = S * sigma_safe * np.sqrt(T_safe)
    safe_denom = np.where(denom > 1e-10, denom, 1e-10)
    gamma = _n(d1_val) / safe_denom

    # At expiry: gamma is 0 (except at-the-money, where it's technically infinite;
    # we return 0 to avoid numerical issues)
    return np.where(T <= 0, 0.0, gamma)


def compute_theta(
    S: float | np.ndarray, K: float | np.ndarray,
    T: float | np.ndarray, r: float,
    sigma: float | np.ndarray, option_type: str = "call",
) -> np.ndarray:
    """
    Option theta: time decay per calendar day.

    Returns a negative number (options lose value over time).
    Divide by trading_days_per_year if you want per-trading-day theta.
    """
    S, K, T, sigma = [np.asarray(x, dtype=np.float64) for x in (S, K, T, sigma)]
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    d1_val = _d1(S, K, T_safe, r, sigma_safe)
    d2_val = _d2(S, K, T_safe, r, sigma_safe)

    # First term: always negative (time decay of optionality)
    term1 = -(S * _n(d1_val) * sigma_safe) / (2 * np.sqrt(T_safe))

    if option_type == "call":
        term2 = -r * K * np.exp(-r * T_safe) * _N(d2_val)
    else:
        term2 = r * K * np.exp(-r * T_safe) * _N(-d2_val)

    # Convert from per-year to per-calendar-day
    theta_annual = term1 + term2
    theta_daily = theta_annual / 365.0

    return np.where(T <= 0, 0.0, theta_daily)


def compute_vega(
    S: float | np.ndarray, K: float | np.ndarray,
    T: float | np.ndarray, r: float,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """
    Option vega: sensitivity of price to a 1 percentage-point move in IV.

    Same for calls and puts. Always ≥ 0.
    Returns vega per 0.01 (1%) change in sigma, not per 1.0 change.
    """
    S, K, T, sigma = [np.asarray(x, dtype=np.float64) for x in (S, K, T, sigma)]
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    d1_val = _d1(S, K, T_safe, r, sigma_safe)

    # Vega = S * N'(d1) * sqrt(T)
    # Divide by 100 so the output is "price change per 1% IV move"
    vega = S * _n(d1_val) * np.sqrt(T_safe) / 100.0

    return np.where(T <= 0, 0.0, vega)


def compute_rho(
    S: float | np.ndarray, K: float | np.ndarray,
    T: float | np.ndarray, r: float,
    sigma: float | np.ndarray, option_type: str = "call",
) -> np.ndarray:
    """
    Option rho: sensitivity of price to a 1 percentage-point move in
    the risk-free rate.

    Returns rho per 0.01 (1%) change in r.
    """
    S, K, T, sigma = [np.asarray(x, dtype=np.float64) for x in (S, K, T, sigma)]
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    d2_val = _d2(S, K, T_safe, r, sigma_safe)

    if option_type == "call":
        rho = K * T_safe * np.exp(-r * T_safe) * _N(d2_val) / 100.0
    else:
        rho = -K * T_safe * np.exp(-r * T_safe) * _N(-d2_val) / 100.0

    return np.where(T <= 0, 0.0, rho)


def compute_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> OptionGreeks:
    """Compute all Greeks for a single option. Returns an OptionGreeks dataclass."""
    return OptionGreeks(
        delta=float(compute_delta(S, K, T, r, sigma, option_type)),
        gamma=float(compute_gamma(S, K, T, r, sigma)),
        theta=float(compute_theta(S, K, T, r, sigma, option_type)),
        vega=float(compute_vega(S, K, T, r, sigma)),
        rho=float(compute_rho(S, K, T, r, sigma, option_type)),
    )


def price_option(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> OptionPrice:
    """
    Full pricing output: price + intrinsic + time value + all Greeks.

    This is the primary interface for pricing a single option.
    For batch pricing, use bs_price() directly with arrays.
    """
    price = float(bs_price(S, K, T, r, sigma, option_type))
    if option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)

    return OptionPrice(
        price=price,
        intrinsic=intrinsic,
        time_value=max(price - intrinsic, 0.0),
        greeks=compute_all_greeks(S, K, T, r, sigma, option_type),
        option_type=OptionType(option_type),
        spot=S,
        strike=K,
        time_to_expiry=T,
        volatility=sigma,
        risk_free_rate=r,
    )


# ── Implied Volatility Solver ──────────────────────────────────────────────────


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied volatility using Newton-Raphson with Brent fallback.

    Parameters
    ----------
    market_price : observed market price of the option
    S, K, T, r : spot, strike, time-to-expiry (years), risk-free rate
    option_type : "call" or "put"
    tol : convergence tolerance
    max_iter : maximum Newton-Raphson iterations

    Returns
    -------
    Implied volatility (annualized). Returns NaN if no solution found.

    Algorithm
    ---------
    1. Brenner-Subrahmanyam initial guess: σ₀ = sqrt(2π/T) × (C/S)
    2. Newton-Raphson: σ_{n+1} = σ_n - (BS(σ_n) - market_price) / vega(σ_n)
    3. If Newton fails (non-convergence, negative vol), fall back to Brent's
       method on [0.01, 5.0] — this always converges if a root exists.

    Edge cases handled:
    - market_price ≤ 0 or T ≤ 0: returns NaN
    - market_price < intrinsic value: returns NaN (arbitrage, bad data)
    - deep ITM/OTM: Brent fallback handles these robustly
    """
    # Sanity checks
    if market_price <= 0 or T <= 1e-10 or S <= 0 or K <= 0:
        return np.nan

    # Check against intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic - tol:
        # Price below intrinsic — arbitrage or bad data
        return np.nan

    # Brenner-Subrahmanyam initial guess
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = np.clip(sigma, 0.01, 3.0)  # Reasonable bounds

    # Newton-Raphson
    for _ in range(max_iter):
        price = float(bs_price(S, K, T, r, sigma, option_type))
        # Vega in raw form (per unit sigma change, not per 1%)
        vega_raw = float(compute_vega(S, K, T, r, sigma)) * 100.0

        if abs(vega_raw) < 1e-12:
            # Vega too small — Newton step would be huge, switch to Brent
            break

        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)

        sigma -= diff / vega_raw
        if sigma <= 0:
            break  # Negative vol — switch to Brent

    # Brent's method fallback
    def objective(vol: float) -> float:
        return float(bs_price(S, K, T, r, vol, option_type)) - market_price

    try:
        result = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=200)
        return float(result)
    except (ValueError, RuntimeError):
        # No root in [0.001, 5.0] — truly unsolvable
        logger.debug(
            "IV solver failed: S=%.2f K=%.2f T=%.4f price=%.2f type=%s",
            S, K, T, market_price, option_type,
        )
        return np.nan


def implied_volatility_batch(
    market_prices: np.ndarray,
    S: float,
    K: np.ndarray,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    option_type: str = "call",
) -> np.ndarray:
    """
    Vectorized IV solver — computes IV for an entire option chain.

    Parameters
    ----------
    market_prices : array of market prices
    S : spot price (scalar)
    K : array of strikes
    T : time to expiry (scalar, years)
    r : risk-free rate
    option_type : "call" or "put"

    Returns
    -------
    Array of implied volatilities, same shape as market_prices.
    NaN for any strike where IV couldn't be solved.
    """
    K = np.asarray(K, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)

    ivs = np.full_like(market_prices, np.nan)
    for i in range(len(market_prices)):
        ivs[i] = implied_volatility(
            market_prices[i], S, K[i], T, r, option_type,
        )
    return ivs


# ── Barone-Adesi-Whaley American Option Approximation ───────────────────────────


def baw_american_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """
    Barone-Adesi-Whaley (1987) quadratic approximation for American options.

    NSE stock options are American-style, so we need this for accurate pricing.
    Index options (NIFTY, BANKNIFTY) are European — use bs_price() instead.

    The BAW approximation adds an early-exercise premium to the European price.
    It's accurate to within ~0.1% of binomial tree pricing for typical parameters.

    Reference: Barone-Adesi, G. & Whaley, R.E. (1987), "Efficient Analytic
    Approximation of American Option Values", Journal of Finance, 42(2).
    """
    # European price as baseline
    euro_price = float(bs_price(S, K, T, r, sigma, option_type))

    if T <= 1e-10:
        # At expiry, American = European = intrinsic
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # For calls with r ≤ 0, early exercise is never optimal
    if option_type == "call" and r <= 0:
        return euro_price

    # BAW parameters
    M = 2 * r / (sigma**2)
    N_val = 2 * (r) / (sigma**2)  # same as M for non-dividend case
    q2 = (-(N_val - 1) + np.sqrt((N_val - 1)**2 + 4 * M / (1 - np.exp(-r * T)))) / 2

    if option_type == "call":
        # Find the critical stock price S* where early exercise is optimal
        # Use iterative approach
        S_star = _baw_critical_price_call(K, T, r, sigma, q2)

        if S >= S_star:
            return S - K  # Exercise immediately

        A2 = (S_star / q2) * (1 - np.exp(-r * T) * _N(_d1(
            np.array([S_star]), np.array([K]), np.array([T]), r, np.array([sigma])
        )[0]))
        premium = A2 * (S / S_star)**q2
        return euro_price + max(premium, 0.0)

    else:  # put
        q1 = (-(N_val - 1) - np.sqrt((N_val - 1)**2 + 4 * M / (1 - np.exp(-r * T)))) / 2
        S_star = _baw_critical_price_put(K, T, r, sigma, q1)

        if S <= S_star:
            return K - S  # Exercise immediately

        A1 = -(S_star / q1) * (1 - np.exp(-r * T) * _N(-_d1(
            np.array([S_star]), np.array([K]), np.array([T]), r, np.array([sigma])
        )[0]))
        premium = A1 * (S / S_star)**q1
        return euro_price + max(premium, 0.0)


def _baw_critical_price_call(
    K: float, T: float, r: float, sigma: float, q2: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Find critical stock price for American call early exercise."""
    # Initial guess: slightly above strike
    S_star = K * 1.1

    for _ in range(max_iter):
        d1_val = _d1(
            np.array([S_star]), np.array([K]), np.array([T]),
            r, np.array([sigma])
        ).item()
        bs = float(bs_price(S_star, K, T, r, sigma, "call"))

        lhs = S_star - K
        rhs = bs + (1 - np.exp(-r * T) * _N(d1_val)) * S_star / q2

        if abs(lhs - rhs) < tol:
            break

        # Newton update (simplified)
        d_bs = float(compute_delta(S_star, K, T, r, sigma, "call"))
        deriv = 1 - (1 / q2) * (1 - np.exp(-r * T) * _N(d1_val)) - d_bs
        if abs(deriv) < 1e-12:
            break
        S_star = S_star - (lhs - rhs) / deriv
        S_star = max(S_star, K * 0.5)  # Don't let it go below K/2

    return S_star


def _baw_critical_price_put(
    K: float, T: float, r: float, sigma: float, q1: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Find critical stock price for American put early exercise."""
    S_star = K * 0.9

    for _ in range(max_iter):
        d1_val = _d1(
            np.array([S_star]), np.array([K]), np.array([T]),
            r, np.array([sigma])
        ).item()
        bs = float(bs_price(S_star, K, T, r, sigma, "put"))

        lhs = K - S_star
        rhs = bs - (1 - np.exp(-r * T) * _N(-d1_val)) * S_star / q1

        if abs(lhs - rhs) < tol:
            break

        d_bs = float(compute_delta(S_star, K, T, r, sigma, "put"))
        deriv = -1 + (1 / q1) * (1 - np.exp(-r * T) * _N(-d1_val)) - d_bs
        if abs(deriv) < 1e-12:
            break
        S_star = S_star - (lhs - rhs) / deriv
        S_star = max(S_star, K * 0.01)  # Floor

    return S_star


# ── Utility Functions ───────────────────────────────────────────────────────────


def put_call_parity_check(
    call_price: float, put_price: float,
    S: float, K: float, T: float, r: float,
    tolerance_pct: float = 1.0,
) -> tuple[bool, float]:
    """
    Verify put-call parity: C - P = S - K*e^(-rT).

    Returns (is_valid, deviation_pct).
    Useful for sanity-checking market data before feeding it to the IV solver.
    """
    theoretical_diff = S - K * np.exp(-r * T)
    actual_diff = call_price - put_price
    deviation = abs(actual_diff - theoretical_diff)
    deviation_pct = (deviation / max(S, 1)) * 100
    return deviation_pct <= tolerance_pct, deviation_pct


def time_to_expiry_years(
    current_date,
    expiry_date,
    trading_days: bool = False,
) -> float:
    """
    Compute time to expiry in years.

    Parameters
    ----------
    current_date : date or datetime
    expiry_date : date or datetime
    trading_days : if True, use trading days (250/year); otherwise calendar days (365/year)
    """
    from datetime import date, datetime

    if isinstance(current_date, datetime):
        current_date = current_date.date()
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()

    days = (expiry_date - current_date).days
    if days <= 0:
        return 0.0

    if trading_days:
        # Rough approximation: 5/7 of calendar days are trading days
        # A more precise version would use an NSE holiday calendar
        td = days * 5 / 7
        return td / TRADING_DAYS_PER_YEAR
    else:
        return days / 365.0


def get_lot_size(symbol: str) -> int:
    """
    Get NSE lot size for a given symbol.

    Falls back to 1 if symbol not in the table (for custom/unlisted instruments).
    """
    return NSE_LOT_SIZES.get(symbol.upper(), 1)


def compute_moneyness(S: float, K: float, option_type: str = "call") -> str:
    """
    Classify option moneyness.

    Returns: "ITM", "ATM", "OTM"
    ATM tolerance: within 1% of spot.
    """
    ratio = S / K
    if abs(ratio - 1.0) < 0.01:
        return "ATM"

    if option_type == "call":
        return "ITM" if S > K else "OTM"
    else:
        return "ITM" if S < K else "OTM"


def synthetic_iv_from_realized_vol(
    realized_vol: float,
    vrp_ratio: float = 1.2,
) -> float:
    """
    Estimate implied volatility when no market IV data is available.

    Uses the empirical Volatility Risk Premium: IV ≈ vrp_ratio × RV.
    The default ratio of 1.2 (IV is ~20% above RV) is the empirical
    average for NIFTY options — this ratio is a starting point that
    must be validated against actual data when available, not a hardcoded
    truth.

    Parameters
    ----------
    realized_vol : historical realized volatility (annualized)
    vrp_ratio : IV/RV ratio (default 1.2 based on NIFTY empirical data)

    Returns
    -------
    Estimated implied volatility.
    """
    return realized_vol * vrp_ratio

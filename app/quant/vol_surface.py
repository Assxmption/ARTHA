"""
Volatility Surface Engine
===========================
Constructs, interpolates, and analyzes the implied volatility surface
σ(K, T) across strikes and expiries.

This is how Jane Street and Citadel Securities actually price options:
they don't use one flat IV number — they model the entire surface and
trade mispricings RELATIVE to the surface.

Surface Parameterization:
  - SVI (Stochastic Volatility Inspired) — Jim Gatheral (2004)
  - Fits total implied variance w(k, T) = a + b(ρ(k-m) + √((k-m)² + σ²))
  - Where k = log(K/F) is log-moneyness
  - 5 parameters: a, b, ρ, m, σ

Why SVI:
  - Analytically tractable (fast calibration)
  - Captures smile and skew with 5 parameters
  - No-arbitrage constraints are well-understood
  - Industry standard for equity vol surfaces

Features:
  - Build surface from option chain data (NSE format)
  - Interpolate IV at arbitrary (K, T) points
  - Detect butterfly/calendar spread arbitrage violations
  - Compute skew metrics (25Δ risk reversal, butterfly)
  - Synthetic IV surface from VIX + empirical skew model

Reference: Gatheral, J. (2004), "A parsimonious arbitrage-free implied
volatility parameterization with application to the valuation of
volatility derivatives"

All computation is deterministic (AGENTS.md rule 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger(__name__)


@dataclass
class SVIParams:
    """SVI parameterization of the smile at a single expiry.

    Total implied variance:
        w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

    Where k = log(K/F) is log-moneyness.
    """
    a: float     # Level of variance
    b: float     # Slope of wings (b > 0)
    rho: float   # Rotation (-1 < ρ < 1)
    m: float     # Translation (center of smile)
    sigma: float # Smoothing (σ > 0)

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Compute total implied variance w(k) = σ²(k) × T."""
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: float | np.ndarray, T: float) -> float | np.ndarray:
        """Compute implied volatility σ(k, T) from total variance."""
        w = self.total_variance(k)
        w = np.maximum(w, 0)  # Can't have negative variance
        return np.sqrt(w / T) if T > 0 else np.sqrt(w)

    def validate_no_arbitrage(self) -> bool:
        """Check basic no-butterfly-arbitrage conditions.

        For SVI to be arbitrage-free:
          1. a + b·σ·√(1-ρ²) ≥ 0  (minimum variance ≥ 0)
          2. b ≥ 0
          3. |ρ| < 1
          4. σ > 0
        """
        min_var = self.a + self.b * self.sigma * np.sqrt(1 - self.rho ** 2)
        return (
            min_var >= -1e-6
            and self.b >= 0
            and abs(self.rho) < 1
            and self.sigma > 0
        )


@dataclass
class VolSurface:
    """Full implied volatility surface across strikes and expiries."""
    expiries: np.ndarray       # T values (years)
    strikes: np.ndarray        # K values (absolute)
    spot: float                # Current spot price
    iv_matrix: np.ndarray      # Shape (len(expiries), len(strikes))
    svi_params: list[SVIParams] = field(default_factory=list)  # One per expiry


# ── SVI Calibration ─────────────────────────────────────────────────────────────

def calibrate_svi(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    T: float,
) -> SVIParams:
    """
    Fit SVI parameters to observed total implied variance.

    Parameters
    ----------
    log_moneyness : np.ndarray
        k = log(K/F) for each strike.
    total_variance : np.ndarray
        w = σ²(K) × T for each strike.
    T : float
        Time to expiry.

    Returns
    -------
    SVIParams
        Calibrated SVI parameters.
    """
    def objective(x):
        params = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
        model_w = params.total_variance(log_moneyness)
        return float(np.sum((model_w - total_variance) ** 2))

    # Bounds: a, b, rho, m, sigma
    bounds = [
        (-0.5, 0.5),      # a: level
        (0.01, 2.0),       # b: slope (positive)
        (-0.99, 0.99),     # rho: rotation
        (-0.5, 0.5),       # m: translation
        (0.001, 1.0),      # sigma: smoothing
    ]

    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=200, tol=1e-8,
    )

    params = SVIParams(
        a=result.x[0], b=result.x[1], rho=result.x[2],
        m=result.x[3], sigma=result.x[4],
    )

    if not params.validate_no_arbitrage():
        logger.warning("SVI fit violates no-arbitrage condition at T=%.4f", T)

    return params


# ── Surface Construction ────────────────────────────────────────────────────────

def build_iv_surface(
    option_chain: dict,
    spot: float,
    r: float = 0.07,
) -> VolSurface:
    """
    Build IV surface from NSE option chain data.

    Parameters
    ----------
    option_chain : dict
        Keys are expiry dates (str), values are dicts with:
          'strikes': list[float], 'ivs': list[float]
    spot : float
        Current spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    VolSurface
    """
    expiries = []
    all_strikes = set()
    svi_params = []

    for expiry_str, data in sorted(option_chain.items()):
        T = data.get('T', 30 / 365)  # Time to expiry in years
        strikes = np.array(data['strikes'])
        ivs = np.array(data['ivs'])

        if len(strikes) < 3:
            continue

        # Forward price (approximate)
        F = spot * np.exp(r * T)

        # Log-moneyness
        k = np.log(strikes / F)

        # Total variance
        w = ivs ** 2 * T

        # Fit SVI
        svi = calibrate_svi(k, w, T)
        svi_params.append(svi)
        expiries.append(T)
        all_strikes.update(strikes.tolist())

    if not expiries:
        raise ValueError("No valid expiries in option chain data")

    expiries_arr = np.array(sorted(expiries))
    strikes_arr = np.array(sorted(all_strikes))

    # Build IV matrix
    iv_matrix = np.zeros((len(expiries_arr), len(strikes_arr)))
    for i, (T, svi) in enumerate(zip(expiries_arr, svi_params)):
        F = spot * np.exp(r * T)
        k = np.log(strikes_arr / F)
        iv_matrix[i, :] = svi.implied_vol(k, T)

    return VolSurface(
        expiries=expiries_arr,
        strikes=strikes_arr,
        spot=spot,
        iv_matrix=iv_matrix,
        svi_params=svi_params,
    )


def build_synthetic_surface(
    spot: float,
    atm_vol: float,
    expiries: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
    skew_slope: float = -0.10,
    term_slope: float = 0.02,
    smile_curvature: float = 0.05,
) -> VolSurface:
    """
    Build a synthetic IV surface from ATM vol + empirical skew model.

    This is used when we don't have real option chain data but want
    realistic pricing. The model is:

        σ(K, T) = σ_ATM × [1 + skew × k + curvature × k² + term × √T]

    Where k = log(K/S) / σ_ATM√T is standardized moneyness.

    Parameters
    ----------
    spot : float
        Current spot price.
    atm_vol : float
        ATM implied volatility (e.g., from VIX).
    skew_slope : float
        Negative for typical equity skew (OTM puts more expensive).
    term_slope : float
        Positive for typical term structure (longer expiry → higher vol).
    smile_curvature : float
        Positive for convex smile (wings up).
    """
    if expiries is None:
        expiries = np.array([7, 14, 30, 60, 90, 180]) / 365.0

    if strikes is None:
        # ±20% around spot, 41 strikes
        strikes = np.linspace(spot * 0.80, spot * 1.20, 41)

    iv_matrix = np.zeros((len(expiries), len(strikes)))
    svi_params = []

    for i, T in enumerate(expiries):
        sqrt_T = np.sqrt(max(T, 1e-6))
        k = np.log(strikes / spot) / (atm_vol * sqrt_T)

        # Empirical skew model
        iv = atm_vol * (1 + skew_slope * k + smile_curvature * k ** 2
                        + term_slope * sqrt_T)
        iv = np.maximum(iv, 0.01)  # Floor at 1%
        iv_matrix[i, :] = iv

        # Fit SVI to the synthetic smile
        log_m = np.log(strikes / spot)
        w = iv ** 2 * T
        svi = calibrate_svi(log_m, w, T)
        svi_params.append(svi)

    return VolSurface(
        expiries=expiries,
        strikes=strikes,
        spot=spot,
        iv_matrix=iv_matrix,
        svi_params=svi_params,
    )


# ── Surface Interpolation ──────────────────────────────────────────────────────

def interpolate_iv(
    surface: VolSurface,
    K: float,
    T: float,
    r: float = 0.07,
) -> float:
    """
    Interpolate IV at arbitrary (K, T) from the surface.

    Uses SVI parameters for the nearest expiry, or bilinear
    interpolation if between expiries.
    """
    if len(surface.svi_params) == 0:
        # Fallback: bilinear interpolation on the matrix
        try:
            spline = RectBivariateSpline(
                surface.expiries, surface.strikes, surface.iv_matrix, kx=1, ky=1
            )
            return float(np.clip(spline(T, K)[0, 0], 0.01, 3.0))
        except Exception:
            # Return ATM vol estimate
            mid_idx = len(surface.strikes) // 2
            return float(surface.iv_matrix[0, mid_idx])

    # Find bracketing expiries
    exp = surface.expiries
    if T <= exp[0]:
        idx = 0
        svi = surface.svi_params[0]
        F = surface.spot * np.exp(r * T)
        k = np.log(K / F)
        return float(np.clip(svi.implied_vol(k, max(T, exp[0])), 0.01, 3.0))
    elif T >= exp[-1]:
        idx = len(exp) - 1
        svi = surface.svi_params[idx]
        F = surface.spot * np.exp(r * T)
        k = np.log(K / F)
        return float(np.clip(svi.implied_vol(k, T), 0.01, 3.0))

    # Interpolate between two expiries
    idx = np.searchsorted(exp, T) - 1
    T1, T2 = exp[idx], exp[idx + 1]
    svi1, svi2 = surface.svi_params[idx], surface.svi_params[idx + 1]

    weight = (T - T1) / (T2 - T1)

    F1 = surface.spot * np.exp(r * T1)
    F2 = surface.spot * np.exp(r * T2)
    k1 = np.log(K / F1)
    k2 = np.log(K / F2)

    # Linear interpolation in total variance space (variance is additive)
    w1 = svi1.total_variance(k1)
    w2 = svi2.total_variance(k2)
    w_interp = (1 - weight) * w1 + weight * w2

    iv = np.sqrt(max(w_interp, 0) / max(T, 1e-6))
    return float(np.clip(iv, 0.01, 3.0))


# ── Arbitrage Detection ────────────────────────────────────────────────────────

@dataclass
class ArbitrageViolation:
    """A detected arbitrage opportunity in the vol surface."""
    type: str          # "butterfly" | "calendar" | "negative_variance"
    strike: float
    expiry: float
    severity: float    # How far from no-arb (in vol points)
    description: str


def detect_surface_arbitrage(surface: VolSurface) -> list[ArbitrageViolation]:
    """
    Scan the vol surface for butterfly and calendar spread arbitrage.

    Butterfly Arbitrage:
      The second derivative of total variance w.r.t. strike must be ≥ 0
      (i.e., the smile must be convex). Otherwise you can construct
      a butterfly spread with negative premium.

    Calendar Arbitrage:
      Total variance must be non-decreasing in T for fixed K.
      Otherwise you can sell a near-dated option and buy a far-dated
      option for a guaranteed profit.

    Returns
    -------
    list[ArbitrageViolation]
        Detected violations (empty = surface is clean).
    """
    violations = []

    # 1. Calendar arbitrage: w(K, T1) ≤ w(K, T2) for T1 < T2
    for j in range(len(surface.strikes)):
        for i in range(len(surface.expiries) - 1):
            iv1 = surface.iv_matrix[i, j]
            iv2 = surface.iv_matrix[i + 1, j]
            T1 = surface.expiries[i]
            T2 = surface.expiries[i + 1]

            w1 = iv1 ** 2 * T1
            w2 = iv2 ** 2 * T2

            if w2 < w1 - 1e-6:
                violations.append(ArbitrageViolation(
                    type="calendar",
                    strike=surface.strikes[j],
                    expiry=T2,
                    severity=float(w1 - w2),
                    description=(
                        f"Calendar arb: w(K={surface.strikes[j]:.0f}, T={T1:.3f})="
                        f"{w1:.4f} > w(T={T2:.3f})={w2:.4f}"
                    ),
                ))

    # 2. Butterfly arbitrage: convexity check per expiry
    for i, T in enumerate(surface.expiries):
        ivs = surface.iv_matrix[i, :]
        K = surface.strikes
        w = ivs ** 2 * T

        if len(K) < 3:
            continue

        # Numerical second derivative of w w.r.t. K
        for j in range(1, len(K) - 1):
            dK = K[j + 1] - K[j - 1]
            d2w = (w[j + 1] - 2 * w[j] + w[j - 1]) / (dK / 2) ** 2

            if d2w < -1e-4:
                violations.append(ArbitrageViolation(
                    type="butterfly",
                    strike=K[j],
                    expiry=T,
                    severity=float(-d2w),
                    description=(
                        f"Butterfly arb at K={K[j]:.0f}, T={T:.3f}: "
                        f"d²w/dK²={d2w:.6f} < 0"
                    ),
                ))

    # 3. Negative variance
    for i, T in enumerate(surface.expiries):
        for j, K in enumerate(surface.strikes):
            iv = surface.iv_matrix[i, j]
            if iv < 0:
                violations.append(ArbitrageViolation(
                    type="negative_variance",
                    strike=K,
                    expiry=T,
                    severity=float(-iv),
                    description=f"Negative IV={iv:.4f} at K={K:.0f}, T={T:.3f}",
                ))

    return violations


# ── Skew Metrics ────────────────────────────────────────────────────────────────

@dataclass
class SkewMetrics:
    """Standard volatility skew metrics."""
    atm_vol: float          # σ_ATM
    skew_25d: float         # σ(25Δ put) - σ(25Δ call), a.k.a. 25Δ risk reversal
    butterfly_25d: float    # (σ(25Δ put) + σ(25Δ call)) / 2 - σ_ATM
    skew_slope: float       # dσ/dk at ATM (local skew)
    term_structure: float   # σ(long T) - σ(short T) at ATM


def compute_skew_metrics(
    surface: VolSurface,
    T: float = 30 / 365,
    r: float = 0.07,
) -> SkewMetrics:
    """
    Compute standard skew metrics at a given expiry.

    25Δ approximation: K_25Δ ≈ S × exp(±0.675 × σ × √T)
    """
    S = surface.spot

    # ATM (at-the-money forward)
    F = S * np.exp(r * T)
    atm_vol = interpolate_iv(surface, F, T, r)

    sqrt_T = np.sqrt(max(T, 1e-6))

    # 25Δ strikes (approximate)
    K_25d_put = F * np.exp(-0.675 * atm_vol * sqrt_T)
    K_25d_call = F * np.exp(0.675 * atm_vol * sqrt_T)

    iv_25d_put = interpolate_iv(surface, K_25d_put, T, r)
    iv_25d_call = interpolate_iv(surface, K_25d_call, T, r)

    # 25Δ risk reversal (negative = puts more expensive)
    rr_25d = iv_25d_put - iv_25d_call

    # 25Δ butterfly (positive = smile, not just skew)
    bf_25d = (iv_25d_put + iv_25d_call) / 2 - atm_vol

    # Local skew slope: finite difference at ATM
    dk = 0.01  # 1% shift
    K_up = F * np.exp(dk)
    K_down = F * np.exp(-dk)
    iv_up = interpolate_iv(surface, K_up, T, r)
    iv_down = interpolate_iv(surface, K_down, T, r)
    skew_slope = (iv_up - iv_down) / (2 * dk)

    # Term structure at ATM
    if len(surface.expiries) >= 2:
        iv_short = interpolate_iv(surface, F, surface.expiries[0], r)
        iv_long = interpolate_iv(surface, F, surface.expiries[-1], r)
        term_struct = iv_long - iv_short
    else:
        term_struct = 0.0

    return SkewMetrics(
        atm_vol=atm_vol,
        skew_25d=rr_25d,
        butterfly_25d=bf_25d,
        skew_slope=skew_slope,
        term_structure=term_struct,
    )

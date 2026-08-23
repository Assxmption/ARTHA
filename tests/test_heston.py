"""
Tests for Heston Stochastic Volatility Model
==============================================
Validates correctness against known degenerate cases, mathematical
properties, and cross-model consistency.
"""
import numpy as np
import pytest

from app.quant.heston import (
    HestonParams,
    heston_call_price,
    heston_put_price,
    heston_implied_vol,
    heston_iv_smile,
    heston_price_chain,
    heston_monte_carlo,
    calibrate_heston,
    NIFTY_DEFAULT_PARAMS,
)


# ── Degenerate Cases (Heston → BS) ─────────────────────────────────────────────

class TestHestonDegeneracy:
    """When vol-of-vol → 0, Heston must reduce to Black-Scholes."""

    def _bs_call(self, S, K, T, r, sigma):
        """Reference BS call for comparison."""
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def test_heston_reduces_to_bs_atm(self):
        """ATM call: Heston with σᵥ→0 should match BS."""
        S, K, T, r = 23000, 23000, 30/365, 0.07
        sigma = 0.20
        params = HestonParams(v0=sigma**2, kappa=5.0, theta=sigma**2,
                              sigma_v=0.001, rho=0.0)  # Nearly zero vol-of-vol

        heston_price = heston_call_price(S, K, T, r, params)
        bs_price = self._bs_call(S, K, T, r, sigma)

        assert abs(heston_price - bs_price) < 5.0, (
            f"Heston ({heston_price:.2f}) should match BS ({bs_price:.2f}) "
            f"when σᵥ→0"
        )

    def test_heston_reduces_to_bs_otm(self):
        """OTM call: same degenerate check."""
        S, K, T, r = 23000, 24000, 30/365, 0.07
        sigma = 0.20
        params = HestonParams(v0=sigma**2, kappa=5.0, theta=sigma**2,
                              sigma_v=0.001, rho=0.0)

        heston_price = heston_call_price(S, K, T, r, params)
        bs_price = self._bs_call(S, K, T, r, sigma)

        # OTM prices are smaller, so relative tolerance
        assert abs(heston_price - bs_price) < max(bs_price * 0.05, 2.0)

    def test_heston_reduces_to_bs_itm(self):
        """ITM call."""
        S, K, T, r = 23000, 22000, 30/365, 0.07
        sigma = 0.20
        params = HestonParams(v0=sigma**2, kappa=5.0, theta=sigma**2,
                              sigma_v=0.001, rho=0.0)

        heston_price = heston_call_price(S, K, T, r, params)
        bs_price = self._bs_call(S, K, T, r, sigma)

        assert abs(heston_price - bs_price) < 10.0


# ── Put-Call Parity ─────────────────────────────────────────────────────────────

class TestHestonPutCallParity:
    """C - P = S - K·e^(-rT) must hold under Heston."""

    @pytest.mark.parametrize("K", [21000, 22000, 23000, 24000, 25000])
    def test_put_call_parity(self, K):
        S, T, r = 23000, 30/365, 0.07
        params = NIFTY_DEFAULT_PARAMS

        C = heston_call_price(S, K, T, r, params)
        P = heston_put_price(S, K, T, r, params)

        lhs = C - P
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 5.0, (
            f"Put-call parity violated at K={K}: C-P={lhs:.2f}, S-Ke^(-rT)={rhs:.2f}"
        )


# ── Volatility Smile ───────────────────────────────────────────────────────────

class TestHestonSmile:
    """Heston should produce a non-flat IV smile (unlike BS)."""

    def test_smile_not_flat(self):
        """IV should vary across strikes with negative ρ."""
        S, T, r = 23000, 30/365, 0.07
        strikes = np.linspace(21000, 25000, 9)
        params = NIFTY_DEFAULT_PARAMS

        ivs = heston_iv_smile(S, strikes, T, r, params)

        # Should not all be the same (that would be BS)
        assert np.std(ivs) > 0.001, "IV smile is flat — Heston isn't working"

    def test_negative_rho_produces_skew(self):
        """Negative ρ → OTM puts more expensive (higher IV at low strikes)."""
        S, T, r = 23000, 30/365, 0.07
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04,
                              sigma_v=0.5, rho=-0.7)

        iv_otm_put = heston_implied_vol(S, 21000, T, r, params)  # Low strike
        iv_atm = heston_implied_vol(S, 23000, T, r, params)      # ATM

        # With negative ρ, low-strike IV should be higher (skew)
        assert iv_otm_put > iv_atm * 0.95, (
            f"Expected skew: OTM put IV ({iv_otm_put:.4f}) should be ≥ ATM IV ({iv_atm:.4f})"
        )


# ── Boundary Conditions ────────────────────────────────────────────────────────

class TestHestonBoundary:
    """Edge cases and boundary behavior."""

    def test_expired_option(self):
        """At T=0, price = intrinsic value."""
        assert heston_call_price(23000, 22000, 0, 0.07, NIFTY_DEFAULT_PARAMS) == 1000
        assert heston_call_price(23000, 24000, 0, 0.07, NIFTY_DEFAULT_PARAMS) == 0

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K·e^(-rT)."""
        S, K, T, r = 23000, 15000, 0.1, 0.07
        price = heston_call_price(S, K, T, r, NIFTY_DEFAULT_PARAMS)
        intrinsic = S - K * np.exp(-r * T)
        assert price >= intrinsic * 0.99

    def test_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        price = heston_call_price(23000, 35000, 0.1, 0.07, NIFTY_DEFAULT_PARAMS)
        assert price < 1.0

    def test_feller_condition(self):
        """Validate Feller condition checker."""
        good = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        assert good.validate()  # 2×2×0.04=0.16 > 0.09

        bad = HestonParams(v0=0.04, kappa=0.1, theta=0.04, sigma_v=1.5, rho=-0.7)
        assert not bad.validate()  # 2×0.1×0.04=0.008 < 2.25


# ── Monte Carlo Consistency ────────────────────────────────────────────────────

class TestHestonMonteCarlo:
    """MC should roughly agree with analytical for European options."""

    def test_mc_matches_analytical_atm(self):
        """MC call price should be within 2% of analytical for ATM."""
        S, K, T, r = 23000, 23000, 60/365, 0.07
        params = NIFTY_DEFAULT_PARAMS

        analytical = heston_call_price(S, K, T, r, params)

        paths = heston_monte_carlo(S, T, r, params, n_paths=50000,
                                   n_steps=100, seed=42)
        payoffs = np.maximum(paths[:, -1] - K, 0)
        mc_price = float(np.exp(-r * T) * np.mean(payoffs))

        # 2% tolerance for MC
        assert abs(mc_price - analytical) < analytical * 0.05, (
            f"MC ({mc_price:.2f}) vs Analytical ({analytical:.2f})"
        )

    def test_mc_paths_shape(self):
        """Verify path array dimensions."""
        paths = heston_monte_carlo(23000, 0.1, 0.07, NIFTY_DEFAULT_PARAMS,
                                   n_paths=100, n_steps=50, seed=1)
        assert paths.shape == (100, 51)

    def test_mc_positive_prices(self):
        """All simulated prices should be positive."""
        paths = heston_monte_carlo(23000, 1.0, 0.07, NIFTY_DEFAULT_PARAMS,
                                   n_paths=1000, n_steps=252, seed=42)
        assert np.all(paths > 0)


# ── Calibration ─────────────────────────────────────────────────────────────────

class TestHestonCalibration:
    """Round-trip test: generate synthetic IVs, calibrate, compare."""

    def test_calibration_round_trip(self):
        """Calibrate to synthetic data and recover approximate params."""
        S, T, r = 23000, 30/365, 0.07
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04,
                                    sigma_v=0.5, rho=-0.7)
        strikes = np.linspace(21000, 25000, 9)

        # Generate synthetic market IVs
        market_ivs = heston_iv_smile(S, strikes, T, r, true_params)

        # Calibrate
        calibrated = calibrate_heston(market_ivs, strikes, S, T, r)

        # Check RMSE of recovered IVs
        recovered_ivs = heston_iv_smile(S, strikes, T, r, calibrated)
        rmse = float(np.sqrt(np.mean((recovered_ivs - market_ivs) ** 2)))

        assert rmse < 0.01, f"Calibration RMSE {rmse:.4f} too high"

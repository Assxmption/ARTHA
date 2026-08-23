"""
Tests for Merton Jump-Diffusion Model
=======================================
Validates correctness against BS degenerate case, mathematical properties,
and cross-model consistency.
"""
import numpy as np
import pytest
from math import exp, log

from app.quant.merton_jump import (
    MertonParams,
    merton_call_price,
    merton_put_price,
    merton_implied_vol,
    merton_iv_smile,
    merton_price_chain,
    merton_monte_carlo,
    calibrate_merton,
    NIFTY_DEFAULT_PARAMS,
)


# ── BS reference ────────────────────────────────────────────────────────────────

def _bs_call(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * (T ** 0.5))
    d2 = d1 - sigma * (T ** 0.5)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


# ── Degenerate Case (Merton → BS when λ → 0) ───────────────────────────────────

class TestMertonDegeneracy:
    """When jump intensity → 0, Merton must reduce to Black-Scholes."""

    @pytest.mark.parametrize("K", [22000, 23000, 24000])
    def test_no_jumps_matches_bs(self, K):
        S, T, r, sigma = 23000, 30/365, 0.07, 0.20

        params = MertonParams(sigma=sigma, lam=0.001, mu_j=0.0, sigma_j=0.01)
        merton_price = merton_call_price(S, K, T, r, params)
        bs_price = _bs_call(S, K, T, r, sigma)

        assert abs(merton_price - bs_price) < 5.0, (
            f"Merton ({merton_price:.2f}) should match BS ({bs_price:.2f}) when λ→0"
        )


# ── Put-Call Parity ─────────────────────────────────────────────────────────────

class TestMertonPutCallParity:
    """C - P = S - K·e^(-rT) must hold."""

    @pytest.mark.parametrize("K", [21000, 22000, 23000, 24000, 25000])
    def test_put_call_parity(self, K):
        S, T, r = 23000, 30/365, 0.07
        params = NIFTY_DEFAULT_PARAMS

        C = merton_call_price(S, K, T, r, params)
        P = merton_put_price(S, K, T, r, params)

        lhs = C - P
        rhs = S - K * exp(-r * T)

        assert abs(lhs - rhs) < 5.0, (
            f"Parity violated at K={K}: C-P={lhs:.2f}, S-Ke^(-rT)={rhs:.2f}"
        )


# ── Volatility Smile (Skew) ────────────────────────────────────────────────────

class TestMertonSmile:
    """Jump-diffusion produces a steeper skew than BS."""

    def test_smile_not_flat(self):
        """IV should vary with strike."""
        S, T, r = 23000, 30/365, 0.07
        strikes = np.linspace(21000, 25000, 9)

        ivs = merton_iv_smile(S, strikes, T, r, NIFTY_DEFAULT_PARAMS)
        assert np.std(ivs) > 0.001, "IV smile is flat"

    def test_negative_jumps_produce_skew(self):
        """Negative average jump → OTM puts more expensive."""
        S, T, r = 23000, 30/365, 0.07
        params = MertonParams(sigma=0.15, lam=3.0, mu_j=-0.05, sigma_j=0.05)

        iv_low = merton_implied_vol(S, 21000, T, r, params)
        iv_atm = merton_implied_vol(S, 23000, T, r, params)

        # With negative jumps, low-strike IV should be higher
        assert iv_low > iv_atm, (
            f"Expected skew: OTM put IV ({iv_low:.4f}) > ATM IV ({iv_atm:.4f})"
        )

    def test_more_jumps_steepens_skew(self):
        """Higher λ → steeper skew."""
        S, T, r = 23000, 60/365, 0.07
        strikes = np.linspace(21000, 25000, 5)

        few = MertonParams(sigma=0.15, lam=0.5, mu_j=-0.03, sigma_j=0.05)
        many = MertonParams(sigma=0.15, lam=5.0, mu_j=-0.03, sigma_j=0.05)

        ivs_few = merton_iv_smile(S, strikes, T, r, few)
        ivs_many = merton_iv_smile(S, strikes, T, r, many)

        # More jumps → more variation in IV
        assert np.std(ivs_many) > np.std(ivs_few)


# ── Pricing Properties ──────────────────────────────────────────────────────────

class TestMertonProperties:
    """Mathematical properties that must hold."""

    def test_call_positive(self):
        assert merton_call_price(23000, 23000, 0.1, 0.07, NIFTY_DEFAULT_PARAMS) > 0

    def test_call_increases_with_spot(self):
        """Higher spot → higher call price."""
        params = NIFTY_DEFAULT_PARAMS
        c1 = merton_call_price(22000, 23000, 0.1, 0.07, params)
        c2 = merton_call_price(24000, 23000, 0.1, 0.07, params)
        assert c2 > c1

    def test_put_increases_with_strike(self):
        """Higher strike → higher put price."""
        params = NIFTY_DEFAULT_PARAMS
        p1 = merton_put_price(23000, 22000, 0.1, 0.07, params)
        p2 = merton_put_price(23000, 24000, 0.1, 0.07, params)
        assert p2 > p1

    def test_call_increases_with_time(self):
        """More time → higher option value."""
        params = NIFTY_DEFAULT_PARAMS
        c1 = merton_call_price(23000, 23000, 10/365, 0.07, params)
        c2 = merton_call_price(23000, 23000, 60/365, 0.07, params)
        assert c2 > c1

    def test_expired_call(self):
        """T=0 → intrinsic value."""
        assert merton_call_price(23000, 22000, 0, 0.07) == 1000
        assert merton_call_price(23000, 24000, 0, 0.07) == 0

    def test_merton_higher_than_bs_for_otm_puts(self):
        """Merton should price OTM puts higher than BS (jump risk)."""
        S, K, T, r = 23000, 20000, 60/365, 0.07
        sigma = 0.20
        params = MertonParams(sigma=sigma, lam=3.0, mu_j=-0.05, sigma_j=0.08)

        merton_p = merton_put_price(S, K, T, r, params)
        bs_p = _bs_call(S, K, T, r, sigma) - S + K * exp(-r * T)  # BS put

        assert merton_p > bs_p, (
            f"Merton OTM put ({merton_p:.2f}) should exceed BS ({bs_p:.2f})"
        )


# ── Vectorized Pricing ──────────────────────────────────────────────────────────

class TestMertonChain:
    """Test vectorized option chain pricing."""

    def test_chain_shape(self):
        strikes = np.array([21000, 22000, 23000, 24000, 25000])
        prices = merton_price_chain(23000, strikes, 30/365, 0.07,
                                     NIFTY_DEFAULT_PARAMS, "call")
        assert len(prices) == 5
        assert all(p >= 0 for p in prices)

    def test_calls_decrease_with_strike(self):
        strikes = np.array([21000, 22000, 23000, 24000, 25000])
        prices = merton_price_chain(23000, strikes, 30/365, 0.07,
                                     NIFTY_DEFAULT_PARAMS, "call")
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 1  # -1 for numerical tolerance


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

class TestMertonMonteCarlo:
    """MC should roughly match analytical."""

    def test_mc_matches_analytical(self):
        S, K, T, r = 23000, 23000, 60/365, 0.07
        params = NIFTY_DEFAULT_PARAMS

        analytical = merton_call_price(S, K, T, r, params)

        paths = merton_monte_carlo(S, T, r, params, n_paths=50000,
                                    n_steps=100, seed=42)
        payoffs = np.maximum(paths[:, -1] - K, 0)
        mc_price = float(np.exp(-r * T) * np.mean(payoffs))

        assert abs(mc_price - analytical) < analytical * 0.05, (
            f"MC ({mc_price:.2f}) vs Analytical ({analytical:.2f})"
        )

    def test_mc_positive_prices(self):
        paths = merton_monte_carlo(23000, 1.0, 0.07, NIFTY_DEFAULT_PARAMS,
                                    n_paths=500, n_steps=50, seed=1)
        assert np.all(paths > 0)

    def test_mc_jump_effect(self):
        """Paths with jumps should show higher kurtosis than GBM."""
        heavy_jumps = MertonParams(sigma=0.10, lam=10.0, mu_j=0.0, sigma_j=0.10)
        paths = merton_monte_carlo(23000, 1.0, 0.07, heavy_jumps,
                                    n_paths=5000, n_steps=252, seed=42)
        log_rets = np.diff(np.log(paths), axis=1).flatten()
        kurtosis = float(np.mean((log_rets - log_rets.mean()) ** 4) /
                        (np.std(log_rets) ** 4))
        # Excess kurtosis should be > 3 (normal) due to jumps
        assert kurtosis > 3.5, f"Kurtosis {kurtosis:.2f} should show fat tails"


# ── Calibration ─────────────────────────────────────────────────────────────────

class TestMertonCalibration:
    """Round-trip calibration test."""

    def test_calibration_round_trip(self):
        S, T, r = 23000, 30/365, 0.07
        true_params = MertonParams(sigma=0.18, lam=2.0, mu_j=-0.03, sigma_j=0.05)
        strikes = np.linspace(21000, 25000, 9)

        market_ivs = merton_iv_smile(S, strikes, T, r, true_params)
        calibrated = calibrate_merton(market_ivs, strikes, S, T, r)

        recovered_ivs = merton_iv_smile(S, strikes, T, r, calibrated)
        rmse = float(np.sqrt(np.mean((recovered_ivs - market_ivs) ** 2)))

        assert rmse < 0.01, f"Calibration RMSE {rmse:.4f} too high"

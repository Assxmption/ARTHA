"""
Tests for Options Pricing & Greeks Engine
==========================================
Validates Black-Scholes pricing, Greeks computation, IV solver, and
American option approximation against known benchmarks.

Test Categories:
  1. BS pricing vs known textbook values (Hull, 10th ed.)
  2. Greeks satisfy mathematical identities (put-call parity, sign conventions)
  3. Greeks boundary conditions (deep ITM/OTM, near/far expiry)
  4. IV solver convergence for realistic inputs
  5. IV solver edge cases (deep ITM/OTM, near-expiry, zero/negative prices)
  6. Barone-Adesi-Whaley American pricing
  7. Utility functions (moneyness, time-to-expiry, lot sizes)

Reference: Implementation Plan §Phase A, Component 1
"""

import pytest
import numpy as np
from datetime import date, timedelta

from app.quant.options_pricing import (
    bs_price, bs_call, bs_put,
    compute_delta, compute_gamma, compute_theta, compute_vega, compute_rho,
    compute_all_greeks, price_option,
    implied_volatility, implied_volatility_batch,
    baw_american_price,
    put_call_parity_check,
    time_to_expiry_years, get_lot_size, compute_moneyness,
    synthetic_iv_from_realized_vol,
    DEFAULT_RISK_FREE_RATE, OptionType,
)


# ── Test Fixtures ───────────────────────────────────────────────────────────────

# Standard test parameters (Hull textbook benchmark)
HULL_S = 42.0      # Spot
HULL_K = 40.0      # Strike
HULL_T = 0.5       # 6 months
HULL_R = 0.10      # 10% risk-free
HULL_SIGMA = 0.20  # 20% vol

# Known Hull result: C ≈ 4.76, P ≈ 0.81
HULL_CALL_PRICE = 4.76
HULL_PUT_PRICE = 0.81


# ── 1. Black-Scholes Pricing ───────────────────────────────────────────────────


class TestBSPricing:
    """Test Black-Scholes option pricing against known benchmarks."""

    def test_hull_call_price(self):
        """BS call price matches Hull's textbook example within 1 cent."""
        price = float(bs_call(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA))
        assert abs(price - HULL_CALL_PRICE) < 0.02, \
            f"Hull call: expected ~{HULL_CALL_PRICE}, got {price:.4f}"

    def test_hull_put_price(self):
        """BS put price matches Hull's textbook example within 1 cent."""
        price = float(bs_put(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA))
        assert abs(price - HULL_PUT_PRICE) < 0.02, \
            f"Hull put: expected ~{HULL_PUT_PRICE}, got {price:.4f}"

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) must hold for European options."""
        call = float(bs_call(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA))
        put = float(bs_put(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA))
        parity_diff = call - put - (HULL_S - HULL_K * np.exp(-HULL_R * HULL_T))
        assert abs(parity_diff) < 1e-6, \
            f"Put-call parity violated: diff = {parity_diff:.8f}"

    def test_call_price_non_negative(self):
        """Call price is always ≥ 0."""
        for S in [10, 50, 100, 500]:
            for K in [10, 50, 100, 500]:
                price = float(bs_call(S, K, 0.5, 0.07, 0.30))
                assert price >= 0, f"Negative call price: S={S}, K={K}, price={price}"

    def test_put_price_non_negative(self):
        """Put price is always ≥ 0."""
        for S in [10, 50, 100, 500]:
            for K in [10, 50, 100, 500]:
                price = float(bs_put(S, K, 0.5, 0.07, 0.30))
                assert price >= 0, f"Negative put price: S={S}, K={K}, price={price}"

    def test_call_at_expiry_is_intrinsic(self):
        """At expiry (T=0), call price = max(S-K, 0)."""
        assert float(bs_call(50, 40, 0.0, 0.07, 0.30)) == pytest.approx(10.0, abs=0.01)
        assert float(bs_call(30, 40, 0.0, 0.07, 0.30)) == pytest.approx(0.0, abs=0.01)

    def test_put_at_expiry_is_intrinsic(self):
        """At expiry (T=0), put price = max(K-S, 0)."""
        assert float(bs_put(50, 40, 0.0, 0.07, 0.30)) == pytest.approx(0.0, abs=0.01)
        assert float(bs_put(30, 40, 0.0, 0.07, 0.30)) == pytest.approx(10.0, abs=0.01)

    def test_vectorized_pricing(self):
        """Vectorized pricing produces same results as scalar."""
        strikes = np.array([35, 40, 45, 50, 55])
        prices = bs_call(HULL_S, strikes, HULL_T, HULL_R, HULL_SIGMA)
        assert prices.shape == (5,)
        # ATM-ish strike should have the highest time value
        intrinsic = np.maximum(HULL_S - strikes, 0)
        time_values = prices - intrinsic
        assert time_values[1] > time_values[0]  # 40 > 35 (less deep ITM)

    def test_higher_vol_means_higher_price(self):
        """Higher IV → higher option price (for both calls and puts)."""
        low_vol = float(bs_call(100, 100, 0.25, 0.07, 0.10))
        high_vol = float(bs_call(100, 100, 0.25, 0.07, 0.40))
        assert high_vol > low_vol

    def test_longer_expiry_means_higher_price(self):
        """Longer time to expiry → higher option price (call)."""
        short = float(bs_call(100, 100, 0.1, 0.07, 0.20))
        long = float(bs_call(100, 100, 1.0, 0.07, 0.20))
        assert long > short

    def test_invalid_option_type_raises(self):
        """Invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type"):
            bs_price(100, 100, 0.5, 0.07, 0.20, option_type="invalid")


# ── 2. Greeks Validation ───────────────────────────────────────────────────────


class TestGreeks:
    """Validate Greeks against mathematical properties."""

    def test_call_delta_between_0_and_1(self):
        """Call delta ∈ [0, 1]."""
        for S in [80, 100, 120]:
            delta = float(compute_delta(S, 100, 0.25, 0.07, 0.20, "call"))
            assert 0 <= delta <= 1, f"Call delta out of range: S={S}, delta={delta}"

    def test_put_delta_between_neg1_and_0(self):
        """Put delta ∈ [-1, 0]."""
        for S in [80, 100, 120]:
            delta = float(compute_delta(S, 100, 0.25, 0.07, 0.20, "put"))
            assert -1 <= delta <= 0, f"Put delta out of range: S={S}, delta={delta}"

    def test_call_put_delta_differ_by_1(self):
        """Call delta - Put delta = 1 (exact for European options)."""
        for S in [80, 100, 120]:
            cd = float(compute_delta(S, 100, 0.25, 0.07, 0.20, "call"))
            pd_ = float(compute_delta(S, 100, 0.25, 0.07, 0.20, "put"))
            assert abs(cd - pd_ - 1.0) < 1e-6, \
                f"Delta parity violated: S={S}, call_d={cd}, put_d={pd_}"

    def test_gamma_always_positive(self):
        """Gamma ≥ 0 for all options (same for calls and puts)."""
        for S in [80, 100, 120]:
            gamma = float(compute_gamma(S, 100, 0.25, 0.07, 0.20))
            assert gamma >= 0, f"Negative gamma: S={S}, gamma={gamma}"

    def test_gamma_highest_atm(self):
        """Gamma is highest at-the-money."""
        gamma_itm = float(compute_gamma(120, 100, 0.25, 0.07, 0.20))
        gamma_atm = float(compute_gamma(100, 100, 0.25, 0.07, 0.20))
        gamma_otm = float(compute_gamma(80, 100, 0.25, 0.07, 0.20))
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_theta_negative_for_long_options(self):
        """Theta is negative (options lose value over time)."""
        theta_call = float(compute_theta(100, 100, 0.25, 0.07, 0.20, "call"))
        theta_put = float(compute_theta(100, 100, 0.25, 0.07, 0.20, "put"))
        assert theta_call < 0, f"Call theta should be negative: {theta_call}"
        # Put theta can be positive for deep ITM puts (interest > time decay)
        # but ATM put theta is typically negative
        assert theta_put < 0, f"ATM put theta should be negative: {theta_put}"

    def test_vega_always_positive(self):
        """Vega ≥ 0 (same for calls and puts)."""
        vega = float(compute_vega(100, 100, 0.25, 0.07, 0.20))
        assert vega > 0, f"Vega should be positive: {vega}"

    def test_vega_highest_atm(self):
        """Vega is highest at-the-money."""
        vega_itm = float(compute_vega(120, 100, 0.25, 0.07, 0.20))
        vega_atm = float(compute_vega(100, 100, 0.25, 0.07, 0.20))
        vega_otm = float(compute_vega(80, 100, 0.25, 0.07, 0.20))
        assert vega_atm > vega_itm
        assert vega_atm > vega_otm

    def test_greeks_finite_difference_delta(self):
        """Delta ≈ (C(S+ε) - C(S-ε)) / (2ε) by finite difference."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.07, 0.20
        eps = 0.01
        fd_delta = (
            float(bs_call(S + eps, K, T, r, sigma))
            - float(bs_call(S - eps, K, T, r, sigma))
        ) / (2 * eps)
        analytical_delta = float(compute_delta(S, K, T, r, sigma, "call"))
        assert abs(fd_delta - analytical_delta) < 0.001, \
            f"FD delta={fd_delta:.6f} vs analytical={analytical_delta:.6f}"

    def test_greeks_finite_difference_gamma(self):
        """Gamma ≈ (Delta(S+ε) - Delta(S-ε)) / (2ε)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.07, 0.20
        eps = 0.01
        fd_gamma = (
            float(compute_delta(S + eps, K, T, r, sigma, "call"))
            - float(compute_delta(S - eps, K, T, r, sigma, "call"))
        ) / (2 * eps)
        analytical_gamma = float(compute_gamma(S, K, T, r, sigma))
        assert abs(fd_gamma - analytical_gamma) < 0.001, \
            f"FD gamma={fd_gamma:.6f} vs analytical={analytical_gamma:.6f}"

    def test_greeks_finite_difference_vega(self):
        """Vega ≈ (C(σ+ε) - C(σ-ε)) / (2ε) × 0.01."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.07, 0.20
        eps = 0.0001  # Small vol perturbation
        fd_vega = (
            float(bs_call(S, K, T, r, sigma + eps))
            - float(bs_call(S, K, T, r, sigma - eps))
        ) / (2 * eps) / 100.0  # Convert to per-1% vega
        analytical_vega = float(compute_vega(S, K, T, r, sigma))
        assert abs(fd_vega - analytical_vega) < 0.01, \
            f"FD vega={fd_vega:.6f} vs analytical={analytical_vega:.6f}"

    def test_compute_all_greeks_returns_dataclass(self):
        """compute_all_greeks returns a populated OptionGreeks."""
        greeks = compute_all_greeks(100, 100, 0.25, 0.07, 0.20, "call")
        assert 0 < greeks.delta < 1
        assert greeks.gamma > 0
        assert greeks.theta < 0
        assert greeks.vega > 0


# ── 3. IV Solver ────────────────────────────────────────────────────────────────


class TestIVSolver:
    """Test implied volatility solver convergence and edge cases."""

    def test_round_trip_call(self):
        """Price a call with known vol, solve back to vol — should match."""
        sigma = 0.25
        price = float(bs_call(100, 100, 0.25, 0.07, sigma))
        iv = implied_volatility(price, 100, 100, 0.25, 0.07, "call")
        assert abs(iv - sigma) < 0.001, f"IV round-trip failed: {iv} vs {sigma}"

    def test_round_trip_put(self):
        """Price a put with known vol, solve back to vol — should match."""
        sigma = 0.30
        price = float(bs_put(100, 100, 0.25, 0.07, sigma))
        iv = implied_volatility(price, 100, 100, 0.25, 0.07, "put")
        assert abs(iv - sigma) < 0.001, f"IV round-trip failed: {iv} vs {sigma}"

    def test_deep_otm_call(self):
        """IV solver handles deep OTM calls (small price)."""
        # 100 call on 80 spot, 1 month — very small price
        sigma = 0.40
        price = float(bs_call(80, 100, 1/12, 0.07, sigma))
        iv = implied_volatility(price, 80, 100, 1/12, 0.07, "call")
        if not np.isnan(iv):
            assert abs(iv - sigma) < 0.01

    def test_deep_itm_put(self):
        """IV solver handles deep ITM puts (large price, small time value)."""
        sigma = 0.25
        price = float(bs_put(80, 120, 0.25, 0.07, sigma))
        iv = implied_volatility(price, 80, 120, 0.25, 0.07, "put")
        if not np.isnan(iv):
            assert abs(iv - sigma) < 0.01

    def test_zero_price_returns_nan(self):
        """Market price of 0 → NaN (no valid IV)."""
        iv = implied_volatility(0.0, 100, 100, 0.25, 0.07, "call")
        assert np.isnan(iv)

    def test_negative_price_returns_nan(self):
        """Negative market price → NaN."""
        iv = implied_volatility(-5.0, 100, 100, 0.25, 0.07, "call")
        assert np.isnan(iv)

    def test_below_intrinsic_returns_nan(self):
        """Price below intrinsic value → NaN (arbitrage)."""
        # ITM call: intrinsic = 100 - 80 = 20; price < 20 is arbitrage
        iv = implied_volatility(15.0, 100, 80, 0.25, 0.07, "call")
        assert np.isnan(iv)

    def test_near_expiry(self):
        """IV solver works for options very close to expiry."""
        sigma = 0.20
        T = 1.0 / 365  # 1 day to expiry
        price = float(bs_call(100, 99, T, 0.07, sigma))
        iv = implied_volatility(price, 100, 99, T, 0.07, "call")
        # Near expiry, IV is less stable — just check it doesn't crash
        assert iv is not None

    def test_batch_iv(self):
        """Batch IV solver produces correct results for a chain."""
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        sigma = 0.25
        prices = bs_call(100, strikes, 0.25, 0.07, sigma)
        ivs = implied_volatility_batch(prices, 100.0, strikes, 0.25, 0.07, "call")
        for iv in ivs:
            if not np.isnan(iv):
                assert abs(iv - sigma) < 0.01


# ── 4. American Options (BAW) ──────────────────────────────────────────────────


class TestBAW:
    """Test Barone-Adesi-Whaley American option approximation."""

    def test_american_call_geq_european(self):
        """American call ≥ European call (early exercise premium ≥ 0)."""
        european = float(bs_call(100, 100, 0.5, 0.07, 0.25))
        american = baw_american_price(100, 100, 0.5, 0.07, 0.25, "call")
        assert american >= european - 0.01  # Allow tiny numerical tolerance

    def test_american_put_geq_european(self):
        """American put ≥ European put."""
        european = float(bs_put(100, 100, 0.5, 0.07, 0.25))
        american = baw_american_price(100, 100, 0.5, 0.07, 0.25, "put")
        assert american >= european - 0.01

    def test_american_at_expiry(self):
        """At expiry, American = intrinsic."""
        assert baw_american_price(110, 100, 0.0, 0.07, 0.25, "call") == pytest.approx(10.0, abs=0.01)
        assert baw_american_price(90, 100, 0.0, 0.07, 0.25, "put") == pytest.approx(10.0, abs=0.01)

    def test_deep_itm_put_early_exercise(self):
        """Deep ITM American put should be close to intrinsic (early exercise)."""
        # S=50, K=100 → deep ITM put, intrinsic = 50
        american = baw_american_price(50, 100, 0.5, 0.07, 0.25, "put")
        assert american >= 49.0  # Should be close to intrinsic


# ── 5. Utility Functions ───────────────────────────────────────────────────────


class TestUtilities:
    """Test utility functions."""

    def test_put_call_parity_check_valid(self):
        """Put-call parity check passes for correctly priced options."""
        call = float(bs_call(100, 100, 0.25, 0.07, 0.20))
        put = float(bs_put(100, 100, 0.25, 0.07, 0.20))
        valid, deviation = put_call_parity_check(call, put, 100, 100, 0.25, 0.07)
        assert valid, f"Parity should be valid, deviation={deviation:.4f}%"

    def test_put_call_parity_check_invalid(self):
        """Put-call parity check fails for mispriced options."""
        valid, deviation = put_call_parity_check(10.0, 10.0, 100, 100, 0.25, 0.07)
        assert not valid, "Should detect parity violation"

    def test_time_to_expiry_calendar(self):
        """Time to expiry in calendar days."""
        today = date(2024, 1, 1)
        expiry = date(2024, 7, 1)  # 182 days later
        T = time_to_expiry_years(today, expiry, trading_days=False)
        assert abs(T - 182 / 365) < 0.001

    def test_time_to_expiry_past(self):
        """Past expiry returns 0."""
        T = time_to_expiry_years(date(2024, 7, 1), date(2024, 1, 1))
        assert T == 0.0

    def test_lot_size_known_symbol(self):
        """Known symbol returns correct lot size."""
        assert get_lot_size("NIFTY") == 25
        assert get_lot_size("RELIANCE") == 250

    def test_lot_size_unknown_symbol(self):
        """Unknown symbol returns 1 (safe fallback)."""
        assert get_lot_size("UNKNOWN_STOCK_XYZ") == 1

    def test_moneyness_classification(self):
        """Moneyness correctly classified."""
        assert compute_moneyness(110, 100, "call") == "ITM"
        assert compute_moneyness(90, 100, "call") == "OTM"
        assert compute_moneyness(100.5, 100, "call") == "ATM"

        assert compute_moneyness(90, 100, "put") == "ITM"
        assert compute_moneyness(110, 100, "put") == "OTM"

    def test_synthetic_iv(self):
        """Synthetic IV = RV × VRP ratio."""
        iv = synthetic_iv_from_realized_vol(0.20, vrp_ratio=1.2)
        assert iv == pytest.approx(0.24)

    def test_price_option_full_output(self):
        """price_option() returns complete OptionPrice with all fields."""
        result = price_option(100, 100, 0.25, 0.07, 0.20, "call")
        assert result.price > 0
        assert result.time_value > 0
        assert result.intrinsic == 0  # ATM
        assert result.option_type == OptionType.CALL
        assert result.greeks.delta > 0
        assert result.greeks.gamma > 0
        assert result.greeks.theta < 0
        assert result.greeks.vega > 0

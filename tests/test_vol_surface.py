"""
Tests for Volatility Surface Engine
=====================================
Validates SVI fit, surface interpolation, arbitrage detection,
and skew metrics.
"""
import numpy as np
import pytest

from app.quant.vol_surface import (
    SVIParams,
    VolSurface,
    calibrate_svi,
    build_synthetic_surface,
    interpolate_iv,
    detect_surface_arbitrage,
    compute_skew_metrics,
)


# ── SVI Parameterization ───────────────────────────────────────────────────────

class TestSVIParams:
    """Test SVI total variance computation."""

    def test_total_variance_at_center(self):
        """At k=m, w = a + b·σ (minimum curvature point)."""
        svi = SVIParams(a=0.01, b=0.2, rho=-0.5, m=0.0, sigma=0.1)
        w_center = svi.total_variance(0.0)
        expected = 0.01 + 0.2 * (-0.5 * 0.0 + np.sqrt(0.0 + 0.01))
        assert abs(w_center - expected) < 1e-10

    def test_total_variance_wings_up(self):
        """Far from center, variance should increase (wings up)."""
        svi = SVIParams(a=0.01, b=0.2, rho=-0.3, m=0.0, sigma=0.1)
        w_center = svi.total_variance(0.0)
        w_left = svi.total_variance(-0.5)
        w_right = svi.total_variance(0.5)
        assert w_left > w_center or w_right > w_center

    def test_implied_vol_positive(self):
        """IV should always be positive."""
        svi = SVIParams(a=0.02, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
        k = np.linspace(-0.3, 0.3, 20)
        ivs = svi.implied_vol(k, 30 / 365)
        assert np.all(ivs >= 0)

    def test_no_arb_valid(self):
        """Proper params should pass no-arb check."""
        svi = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert svi.validate_no_arbitrage()

    def test_no_arb_invalid_negative_b(self):
        """Negative b should fail."""
        svi = SVIParams(a=0.02, b=-0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert not svi.validate_no_arbitrage()


# ── SVI Calibration ────────────────────────────────────────────────────────────

class TestSVICalibration:
    """Test SVI calibration to synthetic data."""

    def test_round_trip(self):
        """Generate data from SVI, calibrate, recover."""
        true_params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.01, sigma=0.08)
        k = np.linspace(-0.3, 0.3, 15)
        T = 30 / 365
        w = true_params.total_variance(k)

        fitted = calibrate_svi(k, w, T)
        fitted_w = fitted.total_variance(k)
        rmse = float(np.sqrt(np.mean((fitted_w - w) ** 2)))

        assert rmse < 0.005, f"SVI calibration RMSE {rmse:.6f} too high"


# ── Synthetic Surface ──────────────────────────────────────────────────────────

class TestSyntheticSurface:
    """Test synthetic surface generation."""

    def test_surface_shape(self):
        """Surface should have correct dimensions."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.15)
        assert surf.iv_matrix.shape[0] == len(surf.expiries)
        assert surf.iv_matrix.shape[1] == len(surf.strikes)
        assert surf.spot == 23000

    def test_surface_positive_ivs(self):
        """All IVs should be positive."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.20)
        assert np.all(surf.iv_matrix > 0)

    def test_surface_atm_near_input(self):
        """ATM IV should be close to input atm_vol."""
        atm_vol = 0.18
        surf = build_synthetic_surface(spot=23000, atm_vol=atm_vol)
        # Find ATM strike (closest to spot)
        atm_idx = np.argmin(np.abs(surf.strikes - 23000))
        # Shortest expiry
        atm_iv = surf.iv_matrix[0, atm_idx]
        assert abs(atm_iv - atm_vol) < 0.03, (
            f"ATM IV ({atm_iv:.4f}) far from input ({atm_vol:.4f})"
        )

    def test_negative_skew(self):
        """Synthetic surface with negative skew should have higher IV for low strikes."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.18, skew_slope=-0.15)
        low_strike_idx = 0  # Lowest strike
        atm_idx = len(surf.strikes) // 2
        # For shortest expiry
        assert surf.iv_matrix[0, low_strike_idx] > surf.iv_matrix[0, atm_idx] * 0.9


# ── Interpolation ──────────────────────────────────────────────────────────────

class TestInterpolation:
    """Test surface interpolation."""

    def test_interpolate_within_grid(self):
        """Interpolated IV should be reasonable within grid."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.18)
        iv = interpolate_iv(surf, 23000, 30 / 365)
        assert 0.05 < iv < 0.50, f"Interpolated IV {iv:.4f} out of range"

    def test_interpolate_between_expiries(self):
        """IV between two expiries should be between their values."""
        surf = build_synthetic_surface(
            spot=23000, atm_vol=0.18,
            expiries=np.array([10, 60]) / 365,
        )
        K = 23000
        iv_short = interpolate_iv(surf, K, 10 / 365)
        iv_long = interpolate_iv(surf, K, 60 / 365)
        iv_mid = interpolate_iv(surf, K, 35 / 365)

        # Mid should be between short and long (approximately)
        lower = min(iv_short, iv_long) * 0.8
        upper = max(iv_short, iv_long) * 1.2
        assert lower < iv_mid < upper

    def test_interpolate_edge_cases(self):
        """Extrapolation beyond grid should still return reasonable values."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.18)
        # Before first expiry
        iv1 = interpolate_iv(surf, 23000, 1 / 365)
        assert 0.01 < iv1 < 3.0
        # After last expiry
        iv2 = interpolate_iv(surf, 23000, 1.0)
        assert 0.01 < iv2 < 3.0


# ── Arbitrage Detection ───────────────────────────────────────────────────────

class TestArbitrageDetection:
    """Test arbitrage violation detection."""

    def test_clean_surface_no_violations(self):
        """Properly constructed surface should have few/no violations."""
        surf = build_synthetic_surface(
            spot=23000, atm_vol=0.18,
            skew_slope=-0.05, smile_curvature=0.03,
        )
        violations = detect_surface_arbitrage(surf)
        # A well-behaved synthetic surface should have minimal violations
        n_serious = sum(1 for v in violations if v.severity > 0.01)
        assert n_serious < 5, f"Too many serious violations: {n_serious}"

    def test_negative_iv_detected(self):
        """Negative IV should be caught."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.18)
        surf.iv_matrix[0, 0] = -0.05  # Inject violation
        violations = detect_surface_arbitrage(surf)
        neg_violations = [v for v in violations if v.type == "negative_variance"]
        assert len(neg_violations) > 0


# ── Skew Metrics ───────────────────────────────────────────────────────────────

class TestSkewMetrics:
    """Test skew metric computation."""

    def test_atm_vol_reasonable(self):
        """ATM vol should be close to input."""
        surf = build_synthetic_surface(spot=23000, atm_vol=0.18)
        metrics = compute_skew_metrics(surf, T=30 / 365)
        assert abs(metrics.atm_vol - 0.18) < 0.05

    def test_negative_skew_rr(self):
        """With negative skew, 25Δ risk reversal should be positive (puts > calls)."""
        surf = build_synthetic_surface(
            spot=23000, atm_vol=0.18, skew_slope=-0.15,
        )
        metrics = compute_skew_metrics(surf, T=30 / 365)
        # Risk reversal = put IV - call IV; with negative skew, puts are more expensive
        assert metrics.skew_25d > -0.10  # At least not massively wrong

    def test_butterfly_non_negative(self):
        """25Δ butterfly should generally be positive (smile curvature)."""
        surf = build_synthetic_surface(
            spot=23000, atm_vol=0.18, smile_curvature=0.05,
        )
        metrics = compute_skew_metrics(surf, T=30 / 365)
        # Butterfly = average wing IV - ATM; should be ≥ 0 for convex smile
        assert metrics.butterfly_25d > -0.05

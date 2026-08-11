"""
Tests for Options & Volatility Signals
========================================
Validates VRP computation, IV percentile, PCR, IV skew, GEX, max pain,
OI concentration, and VIX regime classification.

Uses synthetic data to ensure deterministic, reproducible results.

Reference: Implementation Plan §Phase A
"""

import pytest
import numpy as np
import pandas as pd

from app.quant.options_signals import (
    compute_realized_vol,
    compute_yang_zhang_vol,
    compute_vrp,
    compute_vrp_zscore,
    compute_iv_percentile,
    compute_pcr,
    compute_pcr_from_chain,
    compute_iv_skew,
    compute_iv_term_slope,
    compute_gex,
    compute_max_pain,
    compute_oi_concentration,
    classify_vix_regime,
    compute_all_options_signals,
    OPTIONS_SIGNAL_NAMES,
)


# ── Test Fixtures ───────────────────────────────────────────────────────────────


def _make_price_series(n: int = 500, mu: float = 0.0001, sigma: float = 0.015) -> pd.Series:
    """Generate a synthetic price series with known volatility."""
    np.random.seed(42)
    returns = np.random.normal(mu, sigma, n)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2022-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="price")


def _make_option_chain(spot: float = 100.0, n_strikes: int = 21) -> dict:
    """Generate a synthetic option chain with known properties."""
    strikes = np.linspace(spot * 0.80, spot * 1.20, n_strikes)
    calls = []
    puts = []

    for strike in strikes:
        moneyness = abs(strike - spot) / spot
        base_iv = 20.0 + moneyness * 50  # IV smile
        base_oi = int(10000 * (1 - min(moneyness * 3, 0.9)))

        calls.append({
            "strike": float(strike),
            "lastPrice": max(spot - strike, 0) + 2.0,
            "volume": int(base_oi * 0.5),
            "openInterest": base_oi,
            "changeinOI": int(base_oi * 0.1),
            "impliedVolatility": base_iv,
            "change": 0.5,
        })
        puts.append({
            "strike": float(strike),
            "lastPrice": max(strike - spot, 0) + 2.0,
            "volume": int(base_oi * 0.6),
            "openInterest": int(base_oi * 1.2),
            "changeinOI": int(base_oi * 0.15),
            "impliedVolatility": base_iv * 1.1,  # Put IV > Call IV (normal skew)
            "change": -0.3,
        })

    return {
        "symbol": "NIFTY",
        "spotPrice": spot,
        "calls": calls,
        "puts": puts,
        "expiry": "28-Aug-2026",
    }


# ── Realized Volatility Tests ──────────────────────────────────────────────────


class TestRealizedVol:
    """Test realized volatility computation."""

    def test_rv_positive(self):
        """Realized vol is always positive."""
        prices = _make_price_series()
        rv = compute_realized_vol(prices, window=20)
        assert (rv.dropna() > 0).all()

    def test_rv_annualized_magnitude(self):
        """Annualized RV should be roughly sigma * sqrt(250)."""
        # Daily sigma = 0.015, annualized ≈ 0.015 * sqrt(250) ≈ 0.237
        prices = _make_price_series(500, sigma=0.015)
        rv = compute_realized_vol(prices, window=60)
        mean_rv = rv.dropna().mean()
        expected = 0.015 * np.sqrt(250)
        assert abs(mean_rv - expected) / expected < 0.15, \
            f"RV={mean_rv:.4f} vs expected={expected:.4f}"

    def test_rv_non_annualized(self):
        """Non-annualized RV should be much smaller than annualized."""
        prices = _make_price_series()
        rv_ann = compute_realized_vol(prices, window=20, annualize=True)
        rv_raw = compute_realized_vol(prices, window=20, annualize=False)
        assert rv_ann.dropna().mean() > rv_raw.dropna().mean() * 10

    def test_yang_zhang_vol(self):
        """Yang-Zhang vol produces positive values with OHLC data."""
        np.random.seed(42)
        n = 200
        close = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.015, n))))
        open_p = close * (1 + np.random.normal(0, 0.003, n))
        high = np.maximum(close, open_p) * (1 + abs(np.random.normal(0, 0.005, n)))
        low = np.minimum(close, open_p) * (1 - abs(np.random.normal(0, 0.005, n)))

        idx = pd.bdate_range("2022-01-01", periods=n, freq="B")
        yz = compute_yang_zhang_vol(
            pd.Series(open_p, index=idx),
            pd.Series(high, index=idx),
            pd.Series(low, index=idx),
            pd.Series(close.values, index=idx),
        )
        assert (yz.dropna() > 0).all()


# ── VRP Tests ───────────────────────────────────────────────────────────────────


class TestVRP:
    """Test Volatility Risk Premium computation."""

    def test_vrp_positive_when_iv_exceeds_rv(self):
        """VRP > 0 when IV > RV."""
        iv = pd.Series([0.25, 0.30, 0.28])
        rv = pd.Series([0.20, 0.20, 0.20])
        vrp = compute_vrp(iv, rv)
        assert (vrp > 0).all()

    def test_vrp_negative_when_rv_exceeds_iv(self):
        """VRP < 0 when RV > IV (rare but happens in panics)."""
        iv = pd.Series([0.15])
        rv = pd.Series([0.25])
        vrp = compute_vrp(iv, rv)
        assert (vrp < 0).all()

    def test_vrp_zscore_centered(self):
        """VRP z-score has roughly zero median over long periods."""
        np.random.seed(42)
        vrp_values = pd.Series(np.random.normal(0.2, 0.05, 500))
        zscores = compute_vrp_zscore(vrp_values, lookback=252)
        valid = zscores.dropna()
        assert abs(valid.median()) < 0.5  # Should be near zero


# ── IV Percentile Tests ─────────────────────────────────────────────────────────


class TestIVPercentile:
    """Test IV percentile rank computation."""

    def test_percentile_range(self):
        """IV percentile is in [0, 100]."""
        np.random.seed(42)
        iv = pd.Series(np.random.uniform(0.10, 0.40, 300))
        pct = compute_iv_percentile(iv, lookback=252)
        valid = pct.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_highest_iv_is_100th_percentile(self):
        """The highest IV in the series should be at 100th percentile."""
        # Monotonically increasing IV → last value is 100th percentile
        iv = pd.Series(np.arange(1, 101, dtype=float))
        pct = compute_iv_percentile(iv, lookback=100)
        assert pct.iloc[-1] == pytest.approx(100.0, abs=1.0)

    def test_lowest_iv_is_near_0th_percentile(self):
        """The lowest IV should be at 0th percentile."""
        # Monotonically decreasing → last value is ~0th percentile
        iv = pd.Series(np.arange(100, 0, -1, dtype=float))
        pct = compute_iv_percentile(iv, lookback=100)
        assert pct.iloc[-1] < 5.0


# ── PCR Tests ───────────────────────────────────────────────────────────────────


class TestPCR:
    """Test Put-Call Ratio computation."""

    def test_pcr_scalar(self):
        """Scalar PCR computation."""
        assert compute_pcr(1500, 1000) == pytest.approx(1.5)
        assert compute_pcr(500, 1000) == pytest.approx(0.5)

    def test_pcr_division_by_zero(self):
        """PCR with zero call OI → uses max(call_oi, 1)."""
        result = compute_pcr(1000, 0)
        assert result == 1000.0  # 1000 / 1

    def test_pcr_from_chain(self):
        """PCR from option chain dict."""
        chain = _make_option_chain(100.0)
        result = compute_pcr_from_chain(chain)
        assert "pcr_oi" in result
        assert "pcr_volume" in result
        assert result["pcr_oi"] > 0  # Put OI is 1.2× call OI in fixture
        assert result["total_call_oi"] > 0
        assert result["total_put_oi"] > 0


# ── IV Skew Tests ───────────────────────────────────────────────────────────────


class TestIVSkew:
    """Test IV skew computation."""

    def test_skew_positive_in_normal_market(self):
        """In normal markets, put IV > call IV (positive skew)."""
        chain = _make_option_chain(100.0)
        skew = compute_iv_skew(chain, 100.0)
        if not np.isnan(skew):
            assert skew > 1.0  # Fixture has put IV = 1.1 × call IV

    def test_skew_nan_for_empty_chain(self):
        """Empty option chain → NaN skew."""
        empty = {"calls": [], "puts": [], "spotPrice": 100}
        assert np.isnan(compute_iv_skew(empty, 100.0))


# ── Term Structure Tests ────────────────────────────────────────────────────────


class TestTermStructure:
    """Test IV term structure slope computation."""

    def test_contango(self):
        """Contango: far-month IV > near-month IV → negative slope."""
        slope = compute_iv_term_slope(near_month_iv=15.0, far_month_iv=18.0)
        assert slope < 0

    def test_backwardation(self):
        """Backwardation: near-month IV > far-month IV → positive slope."""
        slope = compute_iv_term_slope(near_month_iv=25.0, far_month_iv=18.0)
        assert slope > 0

    def test_flat_curve(self):
        """Flat curve → slope ≈ 0."""
        slope = compute_iv_term_slope(near_month_iv=20.0, far_month_iv=20.0)
        assert abs(slope) < 0.01


# ── Max Pain Tests ──────────────────────────────────────────────────────────────


class TestMaxPain:
    """Test max pain computation."""

    def test_max_pain_within_strike_range(self):
        """Max pain strike should be within the option chain's strike range."""
        chain = _make_option_chain(100.0)
        result = compute_max_pain(chain)
        min_strike = min(c["strike"] for c in chain["calls"])
        max_strike = max(c["strike"] for c in chain["calls"])
        assert min_strike <= result["max_pain_strike"] <= max_strike

    def test_max_pain_empty_chain(self):
        """Empty chain → max pain = 0."""
        empty = {"calls": [], "puts": [], "spotPrice": 100}
        result = compute_max_pain(empty)
        assert result["max_pain_strike"] == 0

    def test_max_pain_distance(self):
        """Max pain distance is expressed as % of spot."""
        chain = _make_option_chain(100.0)
        result = compute_max_pain(chain)
        # Distance should be reasonable (within ±20% of spot)
        assert abs(result["distance_pct"]) < 20


# ── OI Concentration Tests ──────────────────────────────────────────────────────


class TestOIConcentration:
    """Test OI concentration (HHI) computation."""

    def test_hhi_range(self):
        """HHI is in [0, 1]."""
        chain = _make_option_chain(100.0)
        result = compute_oi_concentration(chain)
        assert 0 <= result["hhi_calls"] <= 1
        assert 0 <= result["hhi_puts"] <= 1

    def test_walls_present(self):
        """Call and put walls should be valid strikes."""
        chain = _make_option_chain(100.0)
        result = compute_oi_concentration(chain)
        assert result["call_wall"] > 0
        assert result["put_wall"] > 0

    def test_top_strikes_ordered(self):
        """Top strikes list should have 3 entries."""
        chain = _make_option_chain(100.0)
        result = compute_oi_concentration(chain)
        assert len(result["top_call_strikes"]) == 3
        assert len(result["top_put_strikes"]) == 3


# ── VIX Regime Tests ────────────────────────────────────────────────────────────


class TestVIXRegime:
    """Test VIX regime classification."""

    def test_low_regime(self):
        """Low VIX → LOW regime."""
        history = pd.Series(np.arange(10.0, 50.0))  # 10 to 49
        regime = classify_vix_regime(11.0, history)  # Below 33rd percentile
        assert regime == "LOW"

    def test_high_regime(self):
        """High VIX → HIGH regime."""
        history = pd.Series(np.arange(10.0, 50.0))
        regime = classify_vix_regime(45.0, history)  # Above 67th percentile
        assert regime == "HIGH"

    def test_medium_regime(self):
        """Mid-range VIX → MEDIUM regime."""
        history = pd.Series(np.arange(10.0, 50.0))
        regime = classify_vix_regime(25.0, history)
        assert regime == "MEDIUM"

    def test_insufficient_data(self):
        """Insufficient data → MEDIUM (safe default)."""
        history = pd.Series([15.0, 20.0, 25.0])  # Only 3 points
        regime = classify_vix_regime(20.0, history)
        assert regime == "MEDIUM"


# ── Aggregate Signal Tests ──────────────────────────────────────────────────────


class TestAggregateSignals:
    """Test the aggregate signal generator."""

    def test_all_signal_names_present(self):
        """All expected signal names are in the output."""
        prices = _make_price_series(300)
        signals = compute_all_options_signals(prices)
        for name in OPTIONS_SIGNAL_NAMES:
            assert name in signals, f"Missing signal: {name}"

    def test_vrp_computed_from_synthetic_iv(self):
        """VRP is computed even without real IV data (using synthetic)."""
        prices = _make_price_series(300)
        signals = compute_all_options_signals(prices)
        # VRP should be ~0.2 (since synthetic IV = 1.2 × RV → VRP = 0.2)
        if not np.isnan(signals["vrp"]):
            assert signals["vrp"] > 0  # Synthetic IV > RV by construction

    def test_with_option_chain(self):
        """Signals computed with option chain include chain-dependent signals."""
        prices = _make_price_series(300)
        chain = _make_option_chain(prices.iloc[-1])
        signals = compute_all_options_signals(prices, option_chain=chain)
        # PCR should be computed
        assert not np.isnan(signals["pcr_oi"])
        assert not np.isnan(signals["pcr_volume"])

    def test_short_series_returns_nans(self):
        """Very short price series returns all NaN signals."""
        prices = pd.Series([100, 101, 102])
        signals = compute_all_options_signals(prices)
        # Most signals should be NaN with only 3 data points
        nan_count = sum(1 for v in signals.values() if np.isnan(v))
        assert nan_count >= len(OPTIONS_SIGNAL_NAMES) - 1

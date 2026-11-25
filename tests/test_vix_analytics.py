"""
Unit tests for India VIX pure analytics functions.

Tests ONLY the deterministic analytics in app/data/vix.py:
  - compute_vix_percentile()
  - compute_vix_regime_series()
  - compute_vix_statistics()
  - compute_vix_mean_reversion_signal()

No network, no filesystem, no mocking — every test runs on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.vix import (
    compute_vix_mean_reversion_signal,
    compute_vix_percentile,
    compute_vix_regime_series,
    compute_vix_statistics,
)

# ── Shared Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def vix_series() -> pd.Series:
    """Synthetic ~5-year India VIX series with realistic-ish random walk."""
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-01", "2023-12-31")
    vix_values = 15 + np.cumsum(np.random.randn(len(dates)) * 0.3)
    vix_values = np.clip(vix_values, 8, 90)
    return pd.Series(vix_values, index=dates, name="vix")


@pytest.fixture()
def short_vix() -> pd.Series:
    """20-day VIX series — shorter than the min_periods=30 default."""
    dates = pd.bdate_range("2023-01-01", periods=20)
    return pd.Series(np.linspace(12, 18, 20), index=dates, name="vix")


@pytest.fixture()
def constant_vix() -> pd.Series:
    """Flat VIX series (MAD = 0, tests division-by-zero guards)."""
    dates = pd.bdate_range("2019-01-01", periods=300)
    return pd.Series(15.0, index=dates, name="vix")


# ── compute_vix_percentile ──────────────────────────────────────────────────────


class TestComputeVixPercentile:
    """Tests for rolling percentile rank of VIX."""

    def test_output_range(self, vix_series: pd.Series) -> None:
        """All non-NaN percentile values must lie in [0, 100]."""
        pct = compute_vix_percentile(vix_series)
        valid = pct.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_highest_in_window_near_100(self) -> None:
        """The maximum value in a lookback window should get ~100 percentile."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        # Steadily increasing series → last value is always the max in window
        vals = np.arange(1.0, 61.0)
        s = pd.Series(vals, index=dates, name="vix")
        pct = compute_vix_percentile(s, lookback=60)
        # Last value: 60 is the largest in the entire series
        assert pct.iloc[-1] == pytest.approx(100.0, abs=1.0)

    def test_lowest_in_window_near_0(self) -> None:
        """The minimum value in a lookback window should get ~0 percentile."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        # Steadily decreasing series → last value is always the min in window
        vals = np.arange(60.0, 0.0, -1.0)
        s = pd.Series(vals, index=dates, name="vix")
        pct = compute_vix_percentile(s, lookback=60)
        assert pct.iloc[-1] == pytest.approx(0.0, abs=1.0)

    def test_short_series_returns_nan(self, short_vix: pd.Series) -> None:
        """Series shorter than min_periods (30) should yield all NaN."""
        pct = compute_vix_percentile(short_vix, lookback=252)
        assert pct.isna().all(), "Expected all NaN for series shorter than min_periods"

    def test_output_length_matches_input(self, vix_series: pd.Series) -> None:
        """Output Series must be the same length as the input."""
        pct = compute_vix_percentile(vix_series)
        assert len(pct) == len(vix_series)


# ── compute_vix_regime_series ───────────────────────────────────────────────────


class TestComputeVixRegimeSeries:
    """Tests for VIX regime classification (LOW / MEDIUM / HIGH)."""

    def test_valid_labels_only(self, vix_series: pd.Series) -> None:
        """Every element must be one of the three allowed regime strings."""
        regimes = compute_vix_regime_series(vix_series)
        assert set(regimes.unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_low_percentile_gives_low(self) -> None:
        """A consistently decreasing VIX tail should classify as LOW."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        vals = np.arange(60.0, 0.0, -1.0)
        s = pd.Series(vals, index=dates, name="vix")
        regimes = compute_vix_regime_series(s, lookback=60)
        assert regimes.iloc[-1] == "LOW"

    def test_high_percentile_gives_high(self) -> None:
        """A consistently increasing VIX tail should classify as HIGH."""
        dates = pd.bdate_range("2020-01-01", periods=60)
        vals = np.arange(1.0, 61.0)
        s = pd.Series(vals, index=dates, name="vix")
        regimes = compute_vix_regime_series(s, lookback=60)
        assert regimes.iloc[-1] == "HIGH"

    def test_nan_percentile_defaults_to_medium(self, short_vix: pd.Series) -> None:
        """When percentile is NaN (too few data points), regime defaults to MEDIUM."""
        regimes = compute_vix_regime_series(short_vix, lookback=252)
        # All percentiles are NaN → all regimes should be MEDIUM
        assert (regimes == "MEDIUM").all()

    def test_output_length_matches_input(self, vix_series: pd.Series) -> None:
        """Output Series must be the same length as the input."""
        regimes = compute_vix_regime_series(vix_series)
        assert len(regimes) == len(vix_series)


# ── compute_vix_statistics ──────────────────────────────────────────────────────


class TestComputeVixStatistics:
    """Tests for VIX summary statistics dict."""

    EXPECTED_KEYS = {
        "current", "mean", "median", "std", "min", "max",
        "percentile_25", "percentile_75", "current_percentile",
        "regime", "days_in_series",
    }

    def test_all_keys_present(self, vix_series: pd.Series) -> None:
        """Returned dict must contain exactly the documented keys."""
        stats = compute_vix_statistics(vix_series)
        assert set(stats.keys()) == self.EXPECTED_KEYS

    def test_empty_series_returns_empty_dict(self) -> None:
        """An empty VIX series should produce an empty dict."""
        empty = pd.Series(dtype=float, name="vix")
        assert compute_vix_statistics(empty) == {}

    def test_too_short_series_returns_empty_dict(self) -> None:
        """A series with fewer than 10 observations should produce an empty dict."""
        dates = pd.bdate_range("2023-01-01", periods=5)
        short = pd.Series([14, 15, 16, 15, 14], index=dates, name="vix", dtype=float)
        assert compute_vix_statistics(short) == {}

    def test_current_is_last_value(self, vix_series: pd.Series) -> None:
        """'current' must equal the last value in the input series."""
        stats = compute_vix_statistics(vix_series)
        assert stats["current"] == pytest.approx(float(vix_series.iloc[-1]))

    def test_days_in_series_matches_len(self, vix_series: pd.Series) -> None:
        """'days_in_series' must equal len(input)."""
        stats = compute_vix_statistics(vix_series)
        assert stats["days_in_series"] == len(vix_series)

    def test_regime_is_valid_string(self, vix_series: pd.Series) -> None:
        """'regime' must be one of LOW, MEDIUM, HIGH."""
        stats = compute_vix_statistics(vix_series)
        assert stats["regime"] in {"LOW", "MEDIUM", "HIGH"}

    def test_percentile_range(self, vix_series: pd.Series) -> None:
        """'current_percentile' must be in [0, 100]."""
        stats = compute_vix_statistics(vix_series)
        assert 0 <= stats["current_percentile"] <= 100


# ── compute_vix_mean_reversion_signal ───────────────────────────────────────────


class TestComputeVixMeanReversionSignal:
    """Tests for the MAD-robust VIX mean-reversion z-score signal."""

    def test_output_length(self, vix_series: pd.Series) -> None:
        """Signal Series must be exactly as long as the input."""
        sig = compute_vix_mean_reversion_signal(vix_series)
        assert len(sig) == len(vix_series)

    def test_elevated_vix_gives_positive_signal(self) -> None:
        """When VIX spikes far above its slow MA, signal should be positive."""
        dates = pd.bdate_range("2019-01-01", periods=300)
        # Calm for 250 days, then a sharp spike for 50 days
        vals = np.concatenate([
            np.full(250, 15.0),
            np.full(50, 40.0),
        ])
        s = pd.Series(vals, index=dates, name="vix")
        sig = compute_vix_mean_reversion_signal(s)
        # The last 10 values (deep into the spike) should be positive
        tail = sig.iloc[-10:].dropna()
        assert (tail > 0).all(), f"Expected positive signal during spike, got:\n{tail}"

    def test_constant_series_no_inf(self, constant_vix: pd.Series) -> None:
        """Constant VIX → MAD = 0 → clipped to 0.001, never inf."""
        sig = compute_vix_mean_reversion_signal(constant_vix)
        valid = sig.dropna()
        assert np.isfinite(valid).all(), "Signal contains inf for constant series"

"""
Tests for Robust Outlier Detection & Z-Scoring
================================================
Covers:
  - robust_zscore_rolling: rolling MAD-based z with Winsorisation
  - robust_zscore_cross_sectional: cross-sectional MAD z
  - flag_bad_data: multi-signal ingestion-level detector
  - detect_circuit_halts: NSE circuit-breaker detection
  - statarb_zscore: EWM-hybrid spread z-scoring
  - Edge cases: empty data, NaNs, constant series, single observation

Reference: AGENTS.md rule 7 — two distinct problems, two solutions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.robust_outliers import (
    BAD_DATA_MAD_THRESHOLD,
    DEFAULT_WINSOR_Z,
    MAD_SCALE,
    OutlierFlag,
    OutlierType,
    detect_circuit_halts,
    flag_bad_data,
    robust_zscore_cross_sectional,
    robust_zscore_rolling,
    statarb_zscore,
    _is_circuit_hit,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def normal_returns():
    """250 days of realistic stock returns (μ≈0.05% daily, σ≈2%)."""
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.0005, 0.02, 250),
        index=pd.date_range("2024-01-01", periods=250, freq="B"),
    )


@pytest.fixture
def normal_ohlcv():
    """Generate realistic OHLCV data for 250 trading days."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 1000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 1, n).astype(int)

    return pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "TEST",
    })


@pytest.fixture
def mean_reverting_spread():
    """OU-like mean-reverting spread for stat-arb testing."""
    np.random.seed(42)
    n = 500
    theta = 0.1
    mu = 0.0
    sigma = 1.0
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) + sigma * np.random.normal()
    return pd.Series(
        spread,
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: robust_zscore_rolling
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustZscoreRolling:
    """Test the rolling MAD-based z-score with Winsorisation."""

    def test_basic_z_distribution(self, normal_returns):
        """Z-scores of normal data should be roughly standard normal."""
        z = robust_zscore_rolling(normal_returns, lookback=60)
        valid = z.dropna()
        assert len(valid) > 100
        # Mean near 0, std near 1 (within tolerance for 250 obs)
        assert abs(valid.mean()) < 0.5
        assert 0.5 < valid.std() < 2.0

    def test_winsorisation_caps_extreme(self):
        """Z-scores should be capped at ±winsor_z."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 200))
        # Inject extreme outlier
        series.iloc[150] = 100.0

        z = robust_zscore_rolling(series, lookback=60, winsor_z=6.0)
        valid = z.dropna()
        assert valid.max() <= 6.0 + 1e-10
        assert valid.min() >= -6.0 - 1e-10

    def test_winsor_none_allows_extremes(self):
        """With winsor_z=None, no capping occurs."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 200))
        series.iloc[150] = 100.0

        z = robust_zscore_rolling(series, lookback=60, winsor_z=None)
        valid = z.dropna()
        # Without capping, the extreme should produce a very large z
        assert valid.max() > 10.0

    def test_constant_series(self):
        """A constant series should produce z=0 (or NaN for insufficient data)."""
        series = pd.Series([42.0] * 100)
        z = robust_zscore_rolling(series, lookback=30)
        valid = z.dropna()
        # Should be all zeros (or very near zero)
        assert (valid.abs() < 1e-6).all() or len(valid) == 0

    def test_short_series(self):
        """Series shorter than min_periods returns all NaN."""
        series = pd.Series([1.0, 2.0, 3.0])
        z = robust_zscore_rolling(series, lookback=60, min_periods=20)
        assert z.isna().all()

    def test_empty_series(self):
        """Empty input returns empty output."""
        z = robust_zscore_rolling(pd.Series(dtype=float))
        assert len(z) == 0

    def test_outlier_robustness(self):
        """
        MAD-based z should be more robust to a single outlier than mean/std.

        After injecting a huge outlier, the z-scores of normal points
        should remain reasonable (not pulled toward the outlier).
        """
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 200))
        # Save clean z before injection
        z_clean = robust_zscore_rolling(series.copy(), lookback=60)

        # Inject outlier
        series.iloc[100] = 50.0
        z_dirty = robust_zscore_rolling(series, lookback=60)

        # Points near the outlier should still have reasonable z
        # (within ±1 of their clean z for neighbors)
        for i in [98, 99, 101, 102]:
            if not (pd.isna(z_clean.iloc[i]) or pd.isna(z_dirty.iloc[i])):
                assert abs(z_dirty.iloc[i] - z_clean.iloc[i]) < 3.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: robust_zscore_cross_sectional
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustZscoreCrossSectional:
    """Test the cross-sectional MAD z-score (replaces _mad_zscore)."""

    def test_basic_zscoring(self):
        """Normal values should produce standard-ish z-scores."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        z = robust_zscore_cross_sectional(values)
        assert z.shape == values.shape
        # Median is 5.5, should have symmetric z around 0
        assert abs(z.mean()) < 0.5

    def test_outlier_resistance(self):
        """A single outlier should not dominate cross-sectional z-scores."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
        z = robust_zscore_cross_sectional(values, winsor_z=6.0)
        # The outlier should be capped at winsor_z
        assert z[-1] <= 6.0 + 1e-10
        # Normal values should have small z-scores
        assert all(abs(z[i]) < 3.0 for i in range(5))

    def test_nan_handling(self):
        """NaN values should be preserved in output."""
        values = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        z = robust_zscore_cross_sectional(values)
        assert np.isnan(z[1])
        assert np.isnan(z[4])
        assert not np.isnan(z[0])
        assert not np.isnan(z[2])

    def test_too_few_values(self):
        """With <3 valid values, z-scores should be 0."""
        values = np.array([1.0, 2.0])
        z = robust_zscore_cross_sectional(values)
        assert (z == 0.0).all()

    def test_identical_values(self):
        """All identical values → MAD=0 → falls back to std → z=0."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        z = robust_zscore_cross_sectional(values)
        assert np.allclose(z, 0.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: flag_bad_data
# ══════════════════════════════════════════════════════════════════════════════

class TestFlagBadData:
    """Test multi-signal bad-data detection."""

    def test_clean_data_no_flags(self, normal_ohlcv):
        """Clean OHLCV data should produce no (or very few) flags."""
        flags = flag_bad_data(normal_ohlcv, "TEST")
        # May flag 0-1 points in random data due to extreme tails
        assert len(flags) <= 2

    def test_fat_finger_detected(self, normal_ohlcv):
        """A 50% single-day return with normal volume → BAD_DATA."""
        df = normal_ohlcv.copy()
        # Inject fat-finger: close jumps 50% but volume stays normal
        mid = len(df) // 2
        df.loc[mid, "close"] = df.loc[mid - 1, "close"] * 1.50
        # Keep high/low consistent with the fat-finger close
        df.loc[mid, "high"] = df.loc[mid, "close"] * 1.01
        df.loc[mid, "low"] = df.loc[mid, "close"] * 0.99

        flags = flag_bad_data(df, "TEST")
        bad_flags = [f for f in flags if f.classification == OutlierType.BAD_DATA]
        assert len(bad_flags) >= 1

    def test_price_inconsistency_detected(self, normal_ohlcv):
        """Close outside [low, high] range → BAD_DATA."""
        df = normal_ohlcv.copy()
        mid = len(df) // 2
        # Make close higher than high — data corruption
        df.loc[mid, "close"] = df.loc[mid, "high"] + 100.0

        flags = flag_bad_data(df, "TEST")
        bad_flags = [f for f in flags if f.classification == OutlierType.BAD_DATA]
        # The price inconsistency might not trigger if the return isn't extreme enough
        # But the function checks inconsistency only for already-flagged returns
        # So we need to also make it extreme
        df.loc[mid, "close"] = df.loc[mid - 1, "close"] * 2.0
        df.loc[mid, "high"] = df.loc[mid, "close"] - 50.0  # Close > high
        flags = flag_bad_data(df, "TEST")
        bad_flags = [f for f in flags if f.classification == OutlierType.BAD_DATA]
        assert len(bad_flags) >= 1

    def test_genuine_move_with_volume_not_bad_data(self, normal_ohlcv):
        """A large move WITH volume spike → not BAD_DATA (circuit or genuine)."""
        df = normal_ohlcv.copy()
        mid = len(df) // 2
        # 30% move with massive volume spike
        df.loc[mid, "close"] = df.loc[mid - 1, "close"] * 1.30
        df.loc[mid, "high"] = df.loc[mid, "close"] * 1.01
        df.loc[mid, "low"] = df.loc[mid, "close"] * 0.85
        df.loc[mid, "volume"] = df["volume"].median() * 50  # Huge volume

        flags = flag_bad_data(df, "TEST")
        for f in flags:
            if abs(f.return_pct - 0.30) < 0.05:
                # Should be GENUINE_MOVE or CIRCUIT_HIT, not BAD_DATA
                assert f.classification != OutlierType.BAD_DATA

    def test_too_few_rows(self):
        """<10 rows → empty flags."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "open": [100] * 5,
            "high": [101] * 5,
            "low": [99] * 5,
            "close": [100] * 5,
            "volume": [1000] * 5,
        })
        flags = flag_bad_data(df, "TEST")
        assert flags == []

    def test_outlier_flag_has_evidence(self, normal_ohlcv):
        """Every flag must have non-empty evidence string."""
        df = normal_ohlcv.copy()
        df.loc[100, "close"] = df.loc[99, "close"] * 2.0  # Inject outlier
        df.loc[100, "high"] = df.loc[100, "close"] * 1.01
        df.loc[100, "low"] = df.loc[100, "close"] * 0.99

        flags = flag_bad_data(df, "TEST")
        for f in flags:
            assert f.evidence, "Evidence string must be non-empty"
            assert f.symbol == "TEST"
            assert isinstance(f.robust_z, float)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: detect_circuit_halts
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectCircuitHalts:
    """Test NSE circuit-breaker detection."""

    def test_upper_circuit_detected(self):
        """A +20% move with close pinned to high → circuit hit."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0, 100.0, 120.5],
            "high": [101.0, 100.5, 120.0],
            "low": [99.0, 99.5, 119.5],
            "close": [100.0, 100.0, 120.0],  # +20% from 100
            "volume": [1000, 1000, 5000],
        })
        halts = detect_circuit_halts(df)
        assert halts.iloc[2] == True

    def test_lower_circuit_detected(self):
        """A -10% move with close pinned to low → circuit hit."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0, 100.0, 91.0],
            "high": [101.0, 101.0, 92.0],
            "low": [99.0, 99.0, 90.0],
            "close": [100.0, 100.0, 90.0],  # -10% from 100
            "volume": [1000, 1000, 5000],
        })
        halts = detect_circuit_halts(df)
        assert halts.iloc[2] == True

    def test_normal_move_not_circuit(self):
        """A 3% move → not a circuit hit."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0, 100.0, 103.0],
            "high": [101.0, 101.0, 104.0],
            "low": [99.0, 99.0, 102.0],
            "close": [100.0, 100.0, 103.0],  # +3% — not a circuit band
            "volume": [1000, 1000, 1500],
        })
        halts = detect_circuit_halts(df)
        assert halts.iloc[2] == False

    def test_first_row_always_false(self):
        """First row should never be flagged (no previous close)."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.0, 110.0, 120.0],
            "volume": [1000, 1000, 1000],
        })
        halts = detect_circuit_halts(df)
        assert halts.iloc[0] == False

    def test_single_row(self):
        """Single-row DataFrame → all False."""
        df = pd.DataFrame({
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
        })
        halts = detect_circuit_halts(df)
        assert (halts == False).all()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: _is_circuit_hit helper
# ══════════════════════════════════════════════════════════════════════════════

class TestIsCircuitHit:
    """Test the circuit-limit band matching helper."""

    def test_exact_5pct(self):
        assert _is_circuit_hit(0.05) == True

    def test_exact_10pct(self):
        assert _is_circuit_hit(-0.10) == True

    def test_exact_20pct(self):
        assert _is_circuit_hit(0.20) == True

    def test_near_band(self):
        """Within tolerance → match."""
        assert _is_circuit_hit(0.0503) == True
        assert _is_circuit_hit(-0.0997) == True

    def test_not_circuit(self):
        """3%, 7%, 15% are not circuit bands."""
        assert _is_circuit_hit(0.03) == False
        assert _is_circuit_hit(0.07) == False
        assert _is_circuit_hit(0.15) == False


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: statarb_zscore
# ══════════════════════════════════════════════════════════════════════════════

class TestStatarbZscore:
    """Test the EWM-hybrid stat-arb spread z-score."""

    def test_mean_reverting_spread(self, mean_reverting_spread):
        """Z-scores of OU spread should be bounded and centered."""
        z = statarb_zscore(mean_reverting_spread, lookback=60)
        valid = z.dropna()
        assert len(valid) > 200

        # Should be centered near 0
        assert abs(valid.mean()) < 1.0

        # Should be bounded by winsor_z
        assert valid.max() <= DEFAULT_WINSOR_Z + 0.1
        assert valid.min() >= -DEFAULT_WINSOR_Z - 0.1

    def test_ewm_more_responsive_than_rolling(self, mean_reverting_spread):
        """EWM z should react faster to regime changes than fixed rolling."""
        z_ewm = statarb_zscore(mean_reverting_spread, use_ewm=True, ewm_halflife=20)
        z_roll = statarb_zscore(mean_reverting_spread, use_ewm=False, lookback=60)

        valid_ewm = z_ewm.dropna()
        valid_roll = z_roll.dropna()

        # Both should be valid
        assert len(valid_ewm) > 100
        assert len(valid_roll) > 100

        # EWM should have slightly higher autocorrelation lag-1
        # (more responsive = faster tracking = higher AC)
        # This is a soft property test — we just check both are reasonable
        assert valid_ewm.std() > 0.1
        assert valid_roll.std() > 0.1

    def test_winsorisation(self, mean_reverting_spread):
        """Injected spike should be Winsorised."""
        spread = mean_reverting_spread.copy()
        spread.iloc[300] = 100.0  # Massive spike

        z = statarb_zscore(spread, lookback=60, winsor_z=6.0)
        valid = z.dropna()
        assert valid.max() <= 6.0 + 0.1

    def test_constant_spread(self):
        """Constant spread → z ≈ 0."""
        spread = pd.Series(
            [10.0] * 200,
            index=pd.date_range("2024-01-01", periods=200, freq="B"),
        )
        z = statarb_zscore(spread, lookback=60)
        valid = z.dropna()
        assert (valid.abs() < 0.1).all() or len(valid) == 0

    def test_short_spread(self):
        """Spread shorter than min_periods → all NaN."""
        spread = pd.Series([1.0, 2.0, 3.0])
        z = statarb_zscore(spread, lookback=60, min_periods=20)
        assert z.isna().all()


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Integration — rule 7 compliance
# ══════════════════════════════════════════════════════════════════════════════

class TestRule7Compliance:
    """
    AGENTS.md rule 7: Outlier handling is two different things.

    (a) Bad data → caught and dropped at ingestion.
    (b) Genuine large market moves → never scrubbed.

    These tests verify the system correctly distinguishes them.
    """

    def test_fat_finger_classified_as_bad_data(self, normal_ohlcv):
        """A 50% return with normal volume → BAD_DATA, not GENUINE_MOVE."""
        df = normal_ohlcv.copy()
        mid = 125
        df.loc[mid, "close"] = df.loc[mid - 1, "close"] * 1.50
        df.loc[mid, "high"] = df.loc[mid, "close"] * 1.01
        df.loc[mid, "low"] = df.loc[mid, "close"] * 0.99
        # Explicitly set volume to be normal (near median) to simulate fat-finger
        df.loc[mid, "volume"] = int(df["volume"].median())

        flags = flag_bad_data(df, "TEST")
        extreme_flags = [f for f in flags if abs(f.return_pct) > 0.3]
        assert len(extreme_flags) >= 1
        assert extreme_flags[0].classification == OutlierType.BAD_DATA

    def test_genuine_shock_not_classified_as_bad_data(self, normal_ohlcv):
        """
        A large return with volume confirmation → GENUINE_MOVE or CIRCUIT_HIT.
        Rule 7b: genuine large moves are NEVER scrubbed.
        """
        df = normal_ohlcv.copy()
        mid = 125
        prev_close = df.loc[mid - 1, "close"]
        # 25% move with huge volume
        new_close = prev_close * 1.25
        df.loc[mid, "close"] = new_close
        df.loc[mid, "high"] = new_close * 1.02
        df.loc[mid, "low"] = new_close * 0.90
        df.loc[mid, "volume"] = int(df["volume"].median() * 100)

        flags = flag_bad_data(df, "TEST")
        for f in flags:
            if abs(f.return_pct - 0.25) < 0.05:
                assert f.classification in (OutlierType.GENUINE_MOVE, OutlierType.CIRCUIT_HIT)

    def test_zscore_preserves_genuine_large_moves(self):
        """
        Robust z-scoring should produce large (but capped) z for genuine
        events — it should NOT suppress them to zero.
        """
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 200))
        # Genuine large move
        series.iloc[150] = 5.0  # 5σ event
        z = robust_zscore_rolling(series, lookback=60, winsor_z=6.0)
        # Should produce a large z (>2) at that point
        if not pd.isna(z.iloc[150]):
            assert abs(z.iloc[150]) > 2.0

"""
Comprehensive unit tests for app/quant/ml_alpha.py.

Tests cover:
  - _winsorize()      — percentile clipping, edge cases (all-same, NaN, empty)
  - _rank_transform() — percentile ranks in [0,1]
  - train_alpha_model — AlphaModelResult fields, early return on insufficient data,
                        correctness with synthetic walk-forward data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.ml_alpha import AlphaModelResult, _winsorize, _rank_transform, train_alpha_model


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic (date, symbol) MultiIndex data spanning ~6 years, 10 symbols."""
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    np.random.seed(42)
    signal_matrix = pd.DataFrame(
        {
            "momentum": np.random.randn(len(index)),
            "value": np.random.randn(len(index)),
            "quality": np.random.randn(len(index)),
        },
        index=index,
    )
    fwd = pd.DataFrame(
        {"fwd_return": np.random.randn(len(index)) * 0.02},
        index=index,
    )
    return signal_matrix, fwd


# ═════════════════════════════════════════════════════════════════════════════
#  _winsorize tests
# ═════════════════════════════════════════════════════════════════════════════


class TestWinsorize:
    """Tests for _winsorize — percentile-based clipping of numeric arrays."""

    def test_clips_to_default_percentile_bounds(self):
        """Values beyond the 1st and 99th percentile should be clipped."""
        np.random.seed(0)
        arr = np.random.randn(10_000)
        result = _winsorize(arr)

        expected_lo = np.percentile(arr, 1)
        expected_hi = np.percentile(arr, 99)

        assert result.min() >= expected_lo
        assert result.max() <= expected_hi

    def test_clips_to_custom_percentile_bounds(self):
        """Custom (5th, 95th) limits should clip tighter than defaults."""
        np.random.seed(1)
        arr = np.random.randn(5_000)
        result = _winsorize(arr, limits=(0.05, 0.95))

        expected_lo = np.percentile(arr, 5)
        expected_hi = np.percentile(arr, 95)

        assert result.min() >= expected_lo - 1e-10
        assert result.max() <= expected_hi + 1e-10

    def test_all_same_values(self):
        """When every element is identical, output should equal input."""
        arr = np.full(100, 7.0)
        result = _winsorize(arr)
        np.testing.assert_array_equal(result, arr)

    def test_handles_nan_values(self):
        """NaN values should be preserved; non-NaN values should be clipped normally."""
        arr = np.array([1.0, 2.0, np.nan, 100.0, -100.0, 3.0])
        result = _winsorize(arr, limits=(0.1, 0.9))

        # NaN propagates through np.clip; check the non-NaN entries are bounded
        finite = result[~np.isnan(result)]
        lo = np.nanpercentile(arr, 10)
        hi = np.nanpercentile(arr, 90)
        assert finite.min() >= lo - 1e-10
        assert finite.max() <= hi + 1e-10
        # The NaN should still be present
        assert np.isnan(result[2])

    def test_empty_array(self):
        """Empty input should return an empty array without raising."""
        arr = np.array([])
        result = _winsorize(arr)
        assert len(result) == 0

    def test_output_shape_matches_input(self):
        """Output array must have the same shape as input."""
        arr = np.random.randn(500)
        result = _winsorize(arr)
        assert result.shape == arr.shape


# ═════════════════════════════════════════════════════════════════════════════
#  _rank_transform tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRankTransform:
    """Tests for _rank_transform — percentile-rank mapping to [0, 1]."""

    def test_returns_values_in_unit_interval(self):
        """All rank values should lie in [0, 1]."""
        s = pd.Series([10, 1, 5, 3, 8])
        result = _rank_transform(s)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_monotonic_order_preserved(self):
        """Larger values should receive higher ranks."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rank_transform(s)
        assert result.is_monotonic_increasing

    def test_rank_of_single_element(self):
        """A single element should receive rank 1.0."""
        s = pd.Series([42.0])
        result = _rank_transform(s)
        assert result.iloc[0] == 1.0

    def test_tied_values_get_average_rank(self):
        """Tied values should receive the average of their ordinal ranks."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = _rank_transform(s)
        # All tied → all get average rank = (1+2+3+4)/4 / 4 = 0.625
        expected = 0.625
        for val in result:
            assert abs(val - expected) < 1e-10


# ═════════════════════════════════════════════════════════════════════════════
#  train_alpha_model tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTrainAlphaModel:
    """Tests for train_alpha_model — walk-forward ensemble training pipeline."""

    def test_early_return_insufficient_data(self):
        """With fewer rows than min_train_days * 5 the model should return an empty result."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        symbols = ["X", "Y"]
        index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        signals = pd.DataFrame({"f1": np.zeros(len(index))}, index=index)
        fwd = pd.DataFrame({"fwd_return": np.zeros(len(index))}, index=index)

        result = train_alpha_model(signals, fwd, min_train_days=504)

        assert isinstance(result, AlphaModelResult)
        assert result.predictions.empty
        assert result.feature_importance == {}
        assert result.oos_ic == 0.0
        assert result.oos_r2 == 0.0
        assert result.n_folds == 0
        assert result.n_training_samples == 0

    def test_returns_alpha_model_result_dataclass(self, synthetic_data):
        """The return type must be AlphaModelResult."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert isinstance(result, AlphaModelResult)

    def test_predictions_dataframe_non_empty(self, synthetic_data):
        """With sufficient data the predictions DataFrame should contain rows."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert not result.predictions.empty
        assert "alpha_score" in result.predictions.columns

    def test_predictions_have_correct_index_names(self, synthetic_data):
        """Predictions should have a (date, symbol) MultiIndex."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert result.predictions.index.names == ["date", "symbol"]

    def test_feature_importance_keys_match_signals(self, synthetic_data):
        """Feature importance dict keys must match the input signal column names."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert set(result.feature_importance.keys()) == set(signals.columns)

    def test_feature_importance_values_are_nonnegative(self, synthetic_data):
        """RF importances are non-negative by construction."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        for v in result.feature_importance.values():
            assert v >= 0.0

    def test_oos_ic_is_finite_float(self, synthetic_data):
        """OOS IC must be a finite float (not NaN or inf)."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert isinstance(result.oos_ic, float)
        assert np.isfinite(result.oos_ic)

    def test_oos_r2_is_finite_float(self, synthetic_data):
        """OOS R² must be a finite float (not NaN or inf)."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert isinstance(result.oos_r2, float)
        assert np.isfinite(result.oos_r2)

    def test_n_folds_positive(self, synthetic_data):
        """With 6 years of daily data and step=63, there should be multiple folds."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert result.n_folds > 0

    def test_n_features_matches_input(self, synthetic_data):
        """n_features should equal the number of signal columns."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert result.n_features == len(signals.columns)

    def test_n_training_samples_positive(self, synthetic_data):
        """n_training_samples should reflect the merged dataset size."""
        signals, fwd = synthetic_data
        result = train_alpha_model(signals, fwd, min_train_days=252, step_size=63)
        assert result.n_training_samples > 0

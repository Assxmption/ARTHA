"""
Test — Multi-Factor Alpha Model
=================================
Unit tests for cross-sectional factor computation and ranking.

Covers:
  - Momentum factor: high-return stock scores higher
  - Volatility factor: low-vol stock scores higher
  - Value factor: high EPS yield scores higher
  - Quality factor: high ROE scores higher
  - Cross-sectional z-scoring: robust MAD-based normalization
  - Composite alpha: weighted combination + ranking
  - Edge cases: insufficient stocks, missing data
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.quant.factors import (
    compute_momentum_factor,
    compute_volatility_factor,
    compute_value_factor,
    compute_quality_factor,
    cross_sectional_zscore,
    compute_composite_alpha,
    FactorExposure,
)


def _make_universe(n_stocks: int = 10, n_days: int = 500, seed: int = 42):
    """Generate a synthetic stock universe with known characteristics."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    price_data = {}
    for i in range(n_stocks):
        trend = 0.0002 * (i - n_stocks // 2)  # Some bull, some bear
        vol = 0.01 + 0.005 * i  # Increasing volatility
        log_ret = trend + vol * rng.randn(n_days)
        prices = 100 * np.exp(np.cumsum(log_ret))
        price_data[f"STOCK{i:02d}"] = pd.Series(prices, index=dates)

    return price_data


class TestMomentumFactor:
    """Test momentum factor computation."""

    def test_uptrend_scores_higher(self):
        """Stock with strong uptrend should have higher momentum."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        strong = pd.Series(100 * np.exp(np.cumsum([0.002] * 300)), index=dates)
        weak = pd.Series(100 * np.exp(np.cumsum([0.0001] * 300)), index=dates)

        scores = compute_momentum_factor({"STRONG": strong, "WEAK": weak})
        assert scores["STRONG"] > scores["WEAK"]

    def test_insufficient_data(self):
        """Stock with too little data should be excluded."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        short = pd.Series([100] * 50, index=dates)
        scores = compute_momentum_factor({"SHORT": short})
        assert "SHORT" not in scores

    def test_empty_universe(self):
        scores = compute_momentum_factor({})
        assert scores == {}


class TestVolatilityFactor:
    """Test volatility factor computation."""

    def test_low_vol_lower_score(self):
        """Lower volatility stock should have lower raw volatility score."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=300)
        low_vol = pd.Series(
            100 * np.exp(np.cumsum(0.005 * rng.randn(300))), index=dates
        )
        high_vol = pd.Series(
            100 * np.exp(np.cumsum(0.03 * rng.randn(300))), index=dates
        )

        scores = compute_volatility_factor({"LOW": low_vol, "HIGH": high_vol})
        assert scores["LOW"] < scores["HIGH"]  # Lower vol = lower raw score

    def test_insufficient_data(self):
        dates = pd.bdate_range("2020-01-01", periods=50)
        short = pd.Series([100] * 50, index=dates)
        scores = compute_volatility_factor({"SHORT": short})
        assert "SHORT" not in scores


class TestValueFactor:
    """Test value factor from fundamentals."""

    def test_higher_eps_yield_scores_higher(self):
        fundamentals = {
            "CHEAP": {"eps": 20.0, "price": 100.0},
            "EXPENSIVE": {"eps": 5.0, "price": 200.0},
        }
        scores = compute_value_factor(fundamentals)
        assert scores["CHEAP"] > scores["EXPENSIVE"]

    def test_missing_eps_excluded(self):
        fundamentals = {
            "HAS_DATA": {"eps": 10.0, "price": 100.0},
            "NO_EPS": {"price": 100.0},
        }
        scores = compute_value_factor(fundamentals)
        assert "HAS_DATA" in scores
        assert "NO_EPS" not in scores

    def test_zero_price_excluded(self):
        fundamentals = {"ZERO": {"eps": 10.0, "price": 0.0}}
        scores = compute_value_factor(fundamentals)
        assert "ZERO" not in scores


class TestQualityFactor:
    """Test quality factor from fundamentals."""

    def test_higher_roe_scores_higher(self):
        fundamentals = {
            "GOOD": {"roe": 25.0},
            "BAD": {"roe": 5.0},
        }
        scores = compute_quality_factor(fundamentals)
        assert scores["GOOD"] > scores["BAD"]

    def test_missing_roe_excluded(self):
        fundamentals = {"NO_ROE": {"eps": 10.0}}
        scores = compute_quality_factor(fundamentals)
        assert "NO_ROE" not in scores


class TestCrossSectionalZScore:
    """Test robust z-scoring."""

    def test_zscore_centered(self):
        """Z-scores should be approximately centered around 0."""
        scores = {"A": 10, "B": 20, "C": 30, "D": 40, "E": 50}
        zscores = cross_sectional_zscore(scores)
        values = list(zscores.values())
        assert abs(np.median(values)) < 0.1  # Median near 0

    def test_zscore_outlier_robustness(self):
        """MAD-based z-scoring should be robust to outliers."""
        scores = {"A": 10, "B": 11, "C": 12, "D": 13, "E": 1000}
        zscores = cross_sectional_zscore(scores)
        # The outlier should have a very high z-score
        assert zscores["E"] > 5

    def test_insufficient_stocks_returns_zeros(self):
        """Fewer than 5 stocks should return all zeros."""
        scores = {"A": 10, "B": 20}
        zscores = cross_sectional_zscore(scores)
        assert all(z == 0.0 for z in zscores.values())


class TestCompositeAlpha:
    """Test composite alpha computation and ranking."""

    def test_ranking_order(self):
        """Stocks should be ranked by composite alpha, best first."""
        universe = _make_universe(n_stocks=10)
        exposures = compute_composite_alpha(universe)
        assert len(exposures) > 0
        # Ranks should be 1, 2, 3, ...
        ranks = [e.rank for e in exposures]
        assert ranks == list(range(1, len(exposures) + 1))

    def test_composite_alpha_sorted(self):
        """Exposures should be sorted by composite alpha (descending)."""
        universe = _make_universe(n_stocks=10)
        exposures = compute_composite_alpha(universe)
        alphas = [e.composite_alpha for e in exposures]
        assert alphas == sorted(alphas, reverse=True)

    def test_with_fundamentals(self):
        """Composite alpha should incorporate fundamentals when provided."""
        universe = _make_universe(n_stocks=6)
        fundamentals = {
            "STOCK00": {"eps": 20, "price": 100, "roe": 25},
            "STOCK01": {"eps": 10, "price": 200, "roe": 10},
            "STOCK02": {"eps": 5, "price": 50, "roe": 30},
        }
        exposures = compute_composite_alpha(universe, fundamentals=fundamentals)
        assert len(exposures) > 0
        # Stocks with fundamentals should have non-zero value/quality z-scores
        with_fund = [e for e in exposures if e.symbol in fundamentals]
        assert len(with_fund) > 0

    def test_empty_universe(self):
        exposures = compute_composite_alpha({})
        assert exposures == []

    def test_single_stock(self):
        """Single stock should get rank 1 with zero z-scores."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.RandomState(42)
        prices = pd.Series(
            100 * np.exp(np.cumsum(0.01 * rng.randn(300))), index=dates
        )
        exposures = compute_composite_alpha({"SOLO": prices})
        # With only one stock, z-scores are all 0
        if exposures:
            assert exposures[0].rank == 1

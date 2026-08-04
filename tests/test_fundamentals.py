"""
Test — Fundamentals Data Ingestion
====================================
Unit tests for the fundamentals data layer.

Covers:
  - Market-derived fundamentals computation.
  - Multi-year fundamentals.
  - Edge cases: missing data, single-observation series, NaN handling.
  - MAD outlier detection (from nse.py).
  - Indian fiscal year convention (April–March).

Note: Tests that require live data fetching are marked with @pytest.mark.slow
and only run when RUN_SLOW_TESTS=true.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

from app.data.nse import flag_outliers_mad, normalize_nse_symbol
from app.factstore.schemas import FundamentalRow


class TestSymbolNormalization:
    """Test NSE symbol normalization."""

    def test_basic_symbol(self):
        assert normalize_nse_symbol("HDFCBANK") == "HDFCBANK"

    def test_lowercase(self):
        assert normalize_nse_symbol("hdfcbank") == "HDFCBANK"

    def test_strip_ns_suffix(self):
        assert normalize_nse_symbol("HDFCBANK.NS") == "HDFCBANK"

    def test_strip_whitespace(self):
        assert normalize_nse_symbol("  HDFCBANK  ") == "HDFCBANK"

    def test_alias(self):
        assert normalize_nse_symbol("HDFC BANK") == "HDFCBANK"


class TestMADOutlierDetection:
    """
    Test MAD-based outlier detection.

    AGENTS.md rule 7:
      (a) Bad data → flagged at ingestion using MAD.
      (b) Genuine large moves → never scrubbed.
    """

    def test_no_outliers_in_normal_data(self):
        """Normal returns should produce no flags."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        flags = flag_outliers_mad(returns)
        # With 250 observations and threshold=6, at most 1 or 2 might be flagged
        # but usually none for truly Gaussian data
        assert flags.sum() <= 3

    def test_extreme_outlier_flagged(self):
        """A genuine fat-finger print should be flagged."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        # Inject a 50% single-day return (clearly a data error)
        returns.iloc[50] = 0.50
        flags = flag_outliers_mad(returns)
        assert flags.iloc[50] is True or flags.iloc[50] == True

    def test_moderate_move_not_flagged(self):
        """
        A 5% daily move should NOT be flagged — that's a genuine large
        move (rule 7b), not a data error.  The threshold is set high
        enough (6× MAD) to avoid this.
        """
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        returns.iloc[50] = 0.05  # 5% move — real, not a glitch
        flags = flag_outliers_mad(returns)
        # This should NOT be flagged with our conservative threshold
        assert flags.iloc[50] == False

    def test_too_few_observations(self):
        """With < 3 observations, no outliers should be flagged."""
        returns = pd.Series([0.01, -0.02])
        flags = flag_outliers_mad(returns)
        assert flags.sum() == 0

    def test_constant_returns(self):
        """All identical returns → MAD=0 → no outliers flagged."""
        returns = pd.Series([0.01] * 50)
        flags = flag_outliers_mad(returns)
        assert flags.sum() == 0

    def test_empty_series(self):
        """Empty series → empty flags."""
        returns = pd.Series(dtype=float)
        flags = flag_outliers_mad(returns)
        assert len(flags) == 0


class TestFundamentalRowSchema:
    """Test the FundamentalRow Pydantic model."""

    def test_valid_fundamental_row(self):
        row = FundamentalRow(
            job_id="test",
            symbol="TCS",
            fiscal_year=2025,
            metric="revenue",
            value=150000.0,
            unit="INR_crore",
            source="NSE_ANNUAL_REPORT_2025",
        )
        assert row.symbol == "TCS"
        assert row.value == 150000.0
        assert row.source == "NSE_ANNUAL_REPORT_2025"

    def test_none_value_allowed(self):
        """value=None represents an unavailable metric."""
        row = FundamentalRow(
            job_id="test",
            symbol="TCS",
            fiscal_year=2025,
            metric="eps_diluted",
            value=None,
            source="NSE_ANNUAL_REPORT_2025",
        )
        assert row.value is None

    def test_empty_source_rejected(self):
        """source must be non-empty (AGENTS.md rule 10)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FundamentalRow(
                job_id="test",
                symbol="TCS",
                fiscal_year=2025,
                metric="revenue",
                value=100.0,
                source="",
            )

    def test_fact_id_auto_generated(self):
        """fact_id should be auto-generated if not provided."""
        row = FundamentalRow(
            job_id="test",
            symbol="TCS",
            fiscal_year=2025,
            metric="revenue",
            value=100.0,
            source="test_source",
        )
        assert row.fact_id is not None
        assert len(row.fact_id) == 32  # hex UUID without hyphens


class TestFundamentalsComputation:
    """
    Test the fundamentals computation pipeline.

    These tests use synthetic data to avoid network dependencies.
    Live data tests are in the @pytest.mark.slow section below.
    """

    def test_fundamentals_from_synthetic_data(self, tmp_path, monkeypatch):
        """
        Compute fundamentals from synthetic OHLCV data.
        Bypasses actual data fetching by mocking fetch_nse_ohlcv.
        """
        from app.data import fundamentals

        # Create synthetic OHLCV data for FY2025 (April 2024 – March 2025)
        dates = pd.bdate_range("2024-04-01", "2025-03-31")
        np.random.seed(42)
        n = len(dates)
        close = 1000 + np.cumsum(np.random.normal(0.5, 15, n))
        close = np.maximum(close, 100)  # ensure positive

        synthetic_df = pd.DataFrame({
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(100000, 5000000, n),
            "symbol": "TESTCO",
        })

        # Mock the fetch function
        def mock_fetch(symbol, start_date, end_date, **kwargs):
            mask = (
                (synthetic_df["date"] >= pd.Timestamp(start_date))
                & (synthetic_df["date"] <= pd.Timestamp(end_date))
            )
            return synthetic_df.loc[mask].reset_index(drop=True)

        monkeypatch.setattr(fundamentals, "fetch_nse_ohlcv", mock_fetch)

        rows = fundamentals.compute_market_fundamentals("TESTCO", 2025)

        assert len(rows) >= 4  # at least return, vol, avg_vol, high, low, close
        metrics = {r.metric for r in rows}
        assert "annual_return_pct" in metrics
        assert "annual_volatility_pct" in metrics
        assert "avg_daily_volume" in metrics
        assert "price_high_52w" in metrics
        assert "price_low_52w" in metrics

        # All rows must have a non-empty source
        for row in rows:
            assert row.source, f"FundamentalRow for {row.metric} has empty source"
            assert "TESTCO" in row.source

    def test_empty_data_returns_empty(self, monkeypatch):
        """No data available → empty list, no crash."""
        from app.data import fundamentals

        def mock_fetch(symbol, start_date, end_date, **kwargs):
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "symbol"]
            )

        monkeypatch.setattr(fundamentals, "fetch_nse_ohlcv", mock_fetch)

        rows = fundamentals.compute_market_fundamentals("GHOST", 2025)
        assert rows == []

    def test_single_observation(self, monkeypatch):
        """
        Single data point → insufficient for return/vol calculation.
        Should return empty or minimal metrics, not crash.
        """
        from app.data import fundamentals

        def mock_fetch(symbol, start_date, end_date, **kwargs):
            return pd.DataFrame({
                "date": [pd.Timestamp("2025-01-15")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
                "symbol": ["SINGLEROW"],
            })

        monkeypatch.setattr(fundamentals, "fetch_nse_ohlcv", mock_fetch)

        rows = fundamentals.compute_market_fundamentals("SINGLEROW", 2025)
        # Should not crash; may return limited metrics
        assert isinstance(rows, list)


class TestFundamentalsToDataFrame:
    """Test the pivot table builder."""

    def test_pivot_table(self):
        from app.data.fundamentals import fundamentals_to_dataframe

        rows = [
            FundamentalRow(
                job_id="test", symbol="TCS", fiscal_year=2024,
                metric="revenue", value=150000.0, source="s1",
            ),
            FundamentalRow(
                job_id="test", symbol="TCS", fiscal_year=2025,
                metric="revenue", value=170000.0, source="s2",
            ),
            FundamentalRow(
                job_id="test", symbol="TCS", fiscal_year=2024,
                metric="operating_margin", value=25.0, source="s3",
            ),
            FundamentalRow(
                job_id="test", symbol="TCS", fiscal_year=2025,
                metric="operating_margin", value=27.0, source="s4",
            ),
        ]

        df = fundamentals_to_dataframe(rows)
        assert df.shape == (2, 2)  # 2 years × 2 metrics
        assert df.loc[2025, "revenue"] == 170000.0
        assert df.loc[2024, "operating_margin"] == 25.0

    def test_empty_rows(self):
        from app.data.fundamentals import fundamentals_to_dataframe

        df = fundamentals_to_dataframe([])
        assert df.empty

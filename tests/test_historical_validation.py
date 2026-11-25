"""Tests for real-data train/OOS validation boundary."""

from datetime import date

import pandas as pd
import pytest

from app.quant.backtest import BacktestMetrics
from app.data.security_master import PointInTimeSecurityMaster
from app.quant.historical_validation import (
    HistoricalValidationError,
    MomentumConfig,
    _validation_evidence,
    cross_sectional_momentum,
    load_real_nse_universe,
)


def test_momentum_output_is_limited_to_requested_window():
    index = pd.date_range("2010-01-01", periods=800, freq="B")
    prices = {
        "A": pd.Series(range(100, 900), index=index, dtype=float),
        "B": pd.Series(range(200, 1000), index=index, dtype=float),
        "C": pd.Series(range(300, 1100), index=index, dtype=float),
        "D": pd.Series(range(400, 1200), index=index, dtype=float),
        "E": pd.Series(range(500, 1300), index=index, dtype=float),
    }
    output = cross_sectional_momentum(
        prices,
        pd.Timestamp("2012-01-01"),
        pd.Timestamp("2012-12-31"),
        MomentumConfig(126, 21, 21),
    )
    assert not output.empty
    assert output.index.min() >= pd.Timestamp("2012-01-01")
    assert output.index.max() <= pd.Timestamp("2012-12-31")


def test_missing_real_cache_fails_closed(tmp_path):
    with pytest.raises(HistoricalValidationError, match="No real NSE files"):
        load_real_nse_universe(tmp_path, train_years=10)


def test_future_or_empty_series_is_not_implicitly_supported():
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = {symbol: pd.Series(100.0, index=index) for symbol in "ABCDE"}
    result = cross_sectional_momentum(
        prices, pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-31"),
        MomentumConfig(5, 1, 2),
    )
    assert result.empty


def test_short_holdout_is_explicitly_rejected_by_canonical_gate():
    metrics = BacktestMetrics(
        total_return=0.5,
        annualized_return=0.2,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.10,
        calmar_ratio=2.0,
        win_rate=0.55,
        num_trades=265,
        num_trading_days=265,
        volatility=0.10,
        avg_trade_return=0.001,
        profit_factor=1.5,
    )
    evidence = _validation_evidence(
        metrics,
        method="chronological_holdout_oos",
        identity_payload={"fixture": True},
        disqualifying_reasons=("survivorship_biased_universe",),
    )
    assert evidence["metrics_gate_passed"] is False
    assert evidence["validated"] is False
    assert evidence["eligible_for_factstore"] is False
    assert "oos_observations_below_504" in evidence["rejection_reasons"]
    assert evidence["data_quality_gate_passed"] is False
    assert "survivorship_biased_universe" in evidence["rejection_reasons"]


def test_momentum_accepts_point_in_time_membership_without_future_filtering():
    index = pd.date_range("2010-01-01", periods=800, freq="B")
    prices = {
        "A": pd.Series(range(100, 900), index=index, dtype=float),
        "B": pd.Series(range(200, 1000), index=index, dtype=float),
        "C": pd.Series(range(300, 1100), index=index, dtype=float),
        "D": pd.Series(range(400, 1200), index=index, dtype=float),
        "E": pd.Series(range(500, 1300), index=index, dtype=float),
    }
    master = PointInTimeSecurityMaster.from_records([
        {"security_id": symbol, "canonical_symbol": symbol,
         "listed_from": "2010-01-01"}
        for symbol in prices
    ])
    result = cross_sectional_momentum(
        prices, pd.Timestamp("2012-01-01"), pd.Timestamp("2012-12-31"),
        MomentumConfig(126, 21, 21), security_master=master,
        trading_index=index,
    )
    assert not result.empty

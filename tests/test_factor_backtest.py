"""Regression tests for factor backtest accounting."""

import numpy as np
import pandas as pd
import pytest

from app.quant.factor_backtest import backtest_factor_model


def _prices(symbol: str, n: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Different deterministic trends make the cross-sectional rank non-empty.
    trend = 0.0002 * (ord(symbol[-1]) - ord("A"))
    close = 100.0 * np.exp(np.cumsum(np.full(n, trend)))
    return pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": 1_000_000,
    }, index=dates)


def test_factor_result_uses_compounded_wealth_metrics():
    result = backtest_factor_model(
        {f"STOCK{letter}": _prices(letter) for letter in "ABCDEFGHIJ"},
        rebalance_days=21,
        lookback_momentum=60,
        min_stocks=5,
        long_only=True,
    )
    assert len(result.daily_returns) >= 20
    compounded = float((1.0 + result.daily_returns).prod() - 1.0)
    assert result.total_return == round(compounded, 6)
    assert result.max_drawdown >= 0.0


def test_static_fundamentals_are_rejected_in_historical_backtest():
    with pytest.raises(ValueError, match="Static fundamentals"):
        backtest_factor_model(
            {f"STOCK{letter}": _prices(letter) for letter in "ABCDEFGHIJ"},
            fundamentals={"STOCKA": {"roe": 10}},
            min_stocks=5,
        )

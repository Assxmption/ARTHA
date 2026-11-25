"""Regression tests for pair backtest accounting."""

import numpy as np
import pandas as pd

from app.quant.statarb_backtest import backtest_pair


def test_pair_metrics_use_compounded_daily_equity_curve():
    rng = np.random.RandomState(7)
    index = pd.bdate_range("2015-01-01", periods=500)
    base = 100 * np.exp(np.cumsum(0.005 * rng.randn(len(index))))
    a = pd.Series(base, index=index)
    b = pd.Series(base * np.exp(0.02 * rng.randn(len(index))), index=index)
    result = backtest_pair(
        a, b, lookback=30, use_kalman=False, max_holding_days=30,
    )
    compounded = float((1.0 + result.daily_returns).prod() - 1.0)
    assert result.total_return == round(compounded, 6)
    assert np.isfinite(result.sharpe_ratio)

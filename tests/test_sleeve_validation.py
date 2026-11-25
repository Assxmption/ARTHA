"""Tests for the causal sleeve matrix."""

import numpy as np
import pandas as pd
import pytest

from app.quant.sleeve_validation import (
    RankSleeveConfig,
    _portfolio_weights,
    rank_sleeve_returns,
    run_repeated_real_sleeve_validation,
)


def _prices(n: int = 500) -> dict[str, pd.Series]:
    index = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.RandomState(12)
    return {
        f"S{i}": pd.Series(
            100 * np.exp(np.cumsum(0.0002 * (i - 5) + 0.01 * rng.randn(n))),
            index=index,
        )
        for i in range(10)
    }


def test_rank_sleeve_is_causal_and_uses_requested_index():
    prices = _prices()
    index = next(iter(prices.values())).index
    config = RankSleeveConfig("fixture", 63, 5, 21)
    first = rank_sleeve_returns(
        prices, index[300], index[-1], config, trading_index=index,
    )
    mutated = {symbol: series.copy() for symbol, series in prices.items()}
    for series in mutated.values():
        series.iloc[350:] *= 10.0
    second = rank_sleeve_returns(
        mutated, index[300], index[349], config, trading_index=index,
    )
    pd.testing.assert_series_equal(first.loc[:index[349]], second)


def test_long_short_and_reversal_produce_aligned_returns():
    prices = _prices()
    index = next(iter(prices.values())).index
    result = rank_sleeve_returns(
        prices, index[300], index[-1],
        RankSleeveConfig("reversal", 21, 1, 5, long_only=False, reverse_rank=True),
        trading_index=index,
    )
    assert result.index.min() >= index[300]
    assert result.index.max() <= index[-1]
    assert result.name == "reversal"


def test_inverse_vol_weights_are_normalized():
    frame = pd.DataFrame({
        "a": [0.01, -0.01, 0.01],
        "b": [0.02, -0.02, 0.02],
    })
    weights = _portfolio_weights(frame, "inverse_volatility")
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > weights["b"]


def test_repeated_validation_rejects_insufficient_history(tmp_path):
    with pytest.raises(Exception, match="No real NSE files|sessions"):
        run_repeated_real_sleeve_validation(
            cache_dir=tmp_path,
            train_years=10,
            oos_days=504,
            n_folds=2,
        )

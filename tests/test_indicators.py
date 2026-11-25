"""Chart indicators: definitions checked against independent formulas and hand values."""

import numpy as np
import pandas as pd
import pytest

from app.quant import indicators as ind
from tests.test_live_dashboard import SNAPSHOT

CLOSE = SNAPSHOT["TCS"].dropna().iloc[-400:].reset_index(drop=True)


def test_sma_matches_rolling_mean():
    pd.testing.assert_series_equal(ind.sma(CLOSE, 20), CLOSE.rolling(20).mean(), check_names=False)


def test_ema_seed_and_recursion_by_hand():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    e = ind.ema(s, 3)
    assert np.isnan(e.iloc[1]) and e.iloc[2] == pytest.approx(2.0)          # seed = SMA(3)
    assert e.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)                    # α = 2/(3+1)
    assert e.iloc[4] == pytest.approx(0.5 * 5 + 0.5 * 3.0)


def test_rsi_wilder_by_hand_and_bounds():
    s = pd.Series([10, 11, 12, 11, 12, 13, 12], dtype=float)
    r = ind.rsi(s, 3)
    # first avg over deltas 1,1,-1 → gain 2/3, loss 1/3 → RSI = 100 - 100/(1+2) = 66.67
    assert r.iloc[3] == pytest.approx(100 - 100 / 3)
    # Wilder step: gain=(2/3*2+1)/3, loss=(1/3*2+0)/3
    g, l = (2 / 3 * 2 + 1) / 3, (1 / 3 * 2) / 3
    assert r.iloc[4] == pytest.approx(100 - 100 / (1 + g / l))
    full = ind.rsi(CLOSE).dropna()
    assert ((full >= 0) & (full <= 100)).all()


def test_rsi_all_gains_is_100():
    assert ind.rsi(pd.Series(np.arange(30, dtype=float)), 14).iloc[-1] == 100.0


def test_bollinger_uses_population_std():
    bb = ind.bollinger(CLOSE, 20, 2.0)
    std = CLOSE.rolling(20).std(ddof=0)
    pd.testing.assert_series_equal(bb["bb_upper"] - bb["bb_mid"], 2 * std, check_names=False)


def test_macd_components():
    m = ind.macd(CLOSE)
    pd.testing.assert_series_equal(m["macd"], ind.ema(CLOSE, 12) - ind.ema(CLOSE, 26), check_names=False)
    valid = m["macd"].dropna()
    pd.testing.assert_series_equal(m["macd_signal"].dropna(), ind.ema(valid, 9).dropna(), check_names=False)
    assert np.allclose(m["macd_hist"].dropna(), (m["macd"] - m["macd_signal"]).dropna())


def test_atr_positive_and_matches_true_range_seed():
    hi, lo = CLOSE * 1.01, CLOSE * 0.99
    a = ind.atr(hi, lo, CLOSE, 14)
    prev = CLOSE.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = hi.iloc[0] - lo.iloc[0]
    assert a.iloc[13] == pytest.approx(tr.iloc[:14].mean())
    assert (a.dropna() > 0).all()


def test_vwap_resets_each_session():
    f = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-20 09:15", "2026-11-20 09:16", "2026-11-23 09:15"]),
        "high": [11.0, 13.0, 21.0], "low": [9.0, 11.0, 19.0], "close": [10.0, 12.0, 20.0],
        "volume": [100, 300, 50],
    })
    v = ind.vwap(f)
    assert v.iloc[1] == pytest.approx((10 * 100 + 12 * 300) / 400)
    assert v.iloc[2] == pytest.approx(20.0)  # new session: no carry-over


def test_compute_returns_aligned_lists_with_none_warmup():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=len(CLOSE)), "open": CLOSE,
                          "high": CLOSE * 1.01, "low": CLOSE * 0.99, "close": CLOSE, "volume": 1000})
    out = ind.compute(frame, ["sma20", "bb", "rsi", "macd", "vwap"], intraday=False)
    assert "vwap" not in out  # not meaningful on daily bars
    assert all(len(v) == len(CLOSE) for v in out.values())
    assert out["sma20"][18] is None and out["sma20"][19] is not None

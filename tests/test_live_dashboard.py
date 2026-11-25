"""
Live dashboard engine — offline tests on a REAL recorded market snapshot.

tests/fixtures/nse_adj_close_20261123.parquet holds Yahoo adjusted closes
for 12 NSE names + NIFTY 50 downloaded on 2026-11-23 (no synthetic prices).
ZOMATO is in the fixture security master but has no prices, exactly as on
Yahoo after its rename, to exercise the data-gap path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app.quant.live_dashboard as ld
from app.quant.backtest import MAX_OOS_DRAWDOWN, MIN_OOS_SHARPE, MIN_OOS_TRADING_DAYS
from app.quant.multi_strategy import vol_target_returns

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = pd.read_parquet(FIXTURES / "nse_adj_close_20261123.parquet").astype(float)
MASTER = FIXTURES / "security_master_live_fixture.json"
IST = ld._now_ist().tzinfo


def _fetch_from_snapshot(symbols, start, end):
    cols = [c for c in SNAPSHOT.columns if c in symbols or c == ld.BENCHMARK]
    return SNAPSHOT.loc[start:end, cols]


@pytest.fixture(scope="module")
def result():
    mp = pytest.MonkeyPatch()
    mp.setattr(ld, "ist_today", lambda: date(2026, 11, 23))
    mp.setattr(ld, "_session_closed", lambda now=None: True)
    calls = {"discovery_max_date": None}
    import app.quant.statarb as statarb

    real_discover = statarb.discover_pairs

    def recording_discover(price_data, *a, **k):
        calls["discovery_max_date"] = max(s.index.max() for s in price_data.values())
        return real_discover(price_data, *a, **k)

    mp.setattr(statarb, "discover_pairs", recording_discover)
    out = ld.compute_live_dashboard(fetch_prices=_fetch_from_snapshot, master_path=MASTER)
    mp.undo()
    out["_discovery_max_date"] = calls["discovery_max_date"]
    return out


def test_windows_are_chronological_and_oos_is_504_sessions(result):
    w = result["windows"]
    assert w["test_end"] == "2026-11-23" == result["as_of"]
    sessions = SNAPSHOT[ld.BENCHMARK].dropna().index
    oos = sessions[sessions >= pd.Timestamp(w["test_start"])]
    assert len(oos) == MIN_OOS_TRADING_DAYS
    assert pd.Timestamp(w["train_end"]) < pd.Timestamp(w["test_start"])


def test_pair_discovery_never_sees_oos_data(result):
    assert result["_discovery_max_date"] <= pd.Timestamp(result["windows"]["train_end"])


def test_missing_symbols_are_reported_not_invented(result):
    assert result["universe"]["missing_data"] == ["ZOMATO"]
    assert result["universe"]["with_data"] == 12


def test_gate_flags_follow_rule_6_thresholds(result):
    def check(m):
        expected = (
            m["sharpe_ratio"] >= MIN_OOS_SHARPE
            and m["max_drawdown"] <= MAX_OOS_DRAWDOWN
            and m["num_trading_days"] >= MIN_OOS_TRADING_DAYS
        )
        assert m["validated"] is expected

    port = result["portfolio"]
    for key in ("oos", "oos_vol_targeted"):
        check(port[key])
    for pair in result["pairs"]["pairs"]:
        check(pair["oos"])
        assert pair["validated"] is pair["oos"]["validated"]
    assert result["pairs"]["validated"] == sum(p["validated"] for p in result["pairs"]["pairs"])


def test_metrics_are_finite_and_oos_covers_full_window(result):
    port = result["portfolio"]
    for key in ("oos", "oos_vol_targeted", "benchmark_oos", "train"):
        for field in ("sharpe_ratio", "max_drawdown", "annualized_return", "volatility"):
            assert np.isfinite(port[key][field]), (key, field)
    assert port["oos"]["num_trading_days"] == MIN_OOS_TRADING_DAYS
    assert abs(sum(port["weights"].values()) - 1.0) < 1e-6


def test_pair_counts_are_consistent(result):
    pr = result["pairs"]
    n = pr["universe_with_full_training_history"]
    assert pr["total_scanned"] == n * (n - 1) // 2
    assert pr["backtested"] == len(pr["pairs"]) <= min(pr["cointegrated"], ld.MAX_PAIRS_BACKTESTED)
    for p in pr["pairs"]:
        assert p["signal"] in (-1, 0, 1)
        assert p["signal_date"] == "2026-11-23"


def test_current_composition_uses_only_active_priced_names(result):
    priced = set(SNAPSHOT.columns) - {ld.BENCHMARK}
    for members in result["portfolio"]["current_composition"].values():
        assert set(members["long"]) | set(members["short"]) <= priced


def test_benchmark_metrics_match_raw_nifty_returns(result):
    from app.quant.backtest import compute_metrics

    sessions = SNAPSHOT[ld.BENCHMARK].dropna()
    oos = sessions.pct_change().loc[result["windows"]["test_start"]:]
    assert compute_metrics(oos).sharpe_ratio == result["portfolio"]["benchmark_oos"]["sharpe_ratio"]


def test_insufficient_history_raises(monkeypatch):
    monkeypatch.setattr(ld, "ist_today", lambda: date(2026, 11, 23))
    short = lambda s, a, b: _fetch_from_snapshot(s, a, b).iloc[-300:]  # noqa: E731
    with pytest.raises(ld.LiveDataError):
        ld.compute_live_dashboard(fetch_prices=short, master_path=MASTER)


def test_missing_benchmark_raises(monkeypatch):
    monkeypatch.setattr(ld, "ist_today", lambda: date(2026, 11, 23))
    no_bench = lambda s, a, b: _fetch_from_snapshot(s, a, b).drop(columns=[ld.BENCHMARK])  # noqa: E731
    with pytest.raises(ld.LiveDataError):
        ld.compute_live_dashboard(fetch_prices=no_bench, master_path=MASTER)


# ── vol targeting must be causal ───────────────────────────────────────────────

def test_vol_targeting_does_not_use_same_day_return():
    rets = SNAPSHOT[ld.BENCHMARK].dropna().pct_change().dropna().iloc[-300:]
    base = vol_target_returns(rets)
    shocked = rets.copy()
    shocked.iloc[-1] = 0.25  # a huge move on the last day
    after = vol_target_returns(shocked)
    # The scale applied on the last day must not react to that day's return.
    assert after.iloc[-1] / shocked.iloc[-1] == pytest.approx(base.iloc[-1] / rets.iloc[-1])
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])


# ── freshness policy ───────────────────────────────────────────────────────────

def _res(computed: datetime) -> dict:
    return {"engine_version": ld.ENGINE_VERSION, "computed_at": computed.isoformat()}


@pytest.mark.parametrize(
    "computed, now, fresh",
    [
        # after Monday's close, computed after it → fresh until next session
        (datetime(2026, 11, 23, 16, 0, tzinfo=IST), datetime(2026, 11, 23, 22, 0, tzinfo=IST), True),
        (datetime(2026, 11, 23, 16, 0, tzinfo=IST), datetime(2026, 11, 24, 9, 0, tzinfo=IST), True),
        # computed before Monday's close, viewed after it → stale
        (datetime(2026, 11, 23, 14, 0, tzinfo=IST), datetime(2026, 11, 23, 16, 0, tzinfo=IST), False),
        # intraday: 15-minute TTL
        (datetime(2026, 11, 25, 10, 0, tzinfo=IST), datetime(2026, 11, 25, 10, 10, tzinfo=IST), True),
        (datetime(2026, 11, 25, 10, 0, tzinfo=IST), datetime(2026, 11, 25, 10, 20, tzinfo=IST), False),
        # NSE holiday (2026-11-24, Guru Nanak Jayanti): Monday's post-close result stays fresh all day
        (datetime(2026, 11, 23, 16, 0, tzinfo=IST), datetime(2026, 11, 24, 12, 0, tzinfo=IST), True),
        (datetime(2026, 11, 23, 16, 0, tzinfo=IST), datetime(2026, 11, 25, 10, 0, tzinfo=IST), False),
        # weekend: Friday post-close result stays fresh
        (datetime(2026, 11, 27, 16, 0, tzinfo=IST), datetime(2026, 11, 29, 12, 0, tzinfo=IST), True),
    ],
)
def test_freshness_policy(computed, now, fresh):
    assert ld.is_fresh(_res(computed), now=now) is fresh


def test_old_engine_version_is_never_fresh():
    r = _res(datetime.now(IST))
    r["engine_version"] = "old"
    assert ld.is_fresh(r) is False


def test_cache_serves_stale_result_and_refreshes_in_background(tmp_path):
    import threading

    stale = {**_res(datetime.now(IST) - timedelta(days=3)), "as_of": "2026-11-20"}
    done = threading.Event()

    def compute():
        done.set()
        return {**_res(datetime.now(IST)), "as_of": "2026-11-23"}

    cache = ld.LiveDashboardCache(path=tmp_path / "d.json", compute=compute)
    cache._result = stale
    first = cache.get()
    assert first["stale"] is True and first["as_of"] == "2026-11-20"
    assert done.wait(5)
    for _ in range(50):
        if not cache._refreshing.is_set():
            break
        threading.Event().wait(0.05)
    second = cache.get()
    assert second["stale"] is False and second["as_of"] == "2026-11-23"
    assert (tmp_path / "d.json").exists()


def test_cache_computes_synchronously_when_empty(tmp_path):
    cache = ld.LiveDashboardCache(
        path=tmp_path / "d.json", compute=lambda: {**_res(datetime.now(IST)), "as_of": "x"}
    )
    assert cache.get()["as_of"] == "x"

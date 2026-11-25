"""
Session replay — offline tests on the REAL recorded snapshot
(tests/fixtures/nse_adj_close_20261123.parquet, Yahoo adjusted closes).

The key test replays a historical session using daily closes as the
"minute" data and checks the result equals the sleeve backtester's own
return for that session — i.e. the replay simulates exactly the strategy
whose OOS statistics the dashboard reports.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import app.quant.live_dashboard as ld
import app.quant.session_replay as sr
from app.data.security_master import PointInTimeSecurityMaster
from app.quant.sleeve_validation import _portfolio_weights, rank_sleeve_returns
from tests.test_live_dashboard import MASTER, SNAPSHOT, _fetch_from_snapshot

SESSIONS = SNAPSHOT[ld.BENCHMARK].dropna().index


def _daily(symbols, start, end):
    cols = [c for c in SNAPSHOT.columns if c in symbols]
    return SNAPSHOT.loc[pd.Timestamp(start):pd.Timestamp(end), cols], "fixture daily"


def _minutes_from_daily(symbols, session):
    """One bar at 15:29 carrying the official close — so intraday == EOD."""
    row = SNAPSHOT.loc[pd.Timestamp(session), [c for c in SNAPSHOT.columns if c in symbols]]
    return pd.DataFrame([row.values], columns=row.index,
                        index=[pd.Timestamp(f"{session} 15:29")]), "fixture minute"


@pytest.fixture(autouse=True)
def _pin_dates(monkeypatch):
    monkeypatch.setattr(ld, "ist_today", lambda: date(2026, 11, 23))
    monkeypatch.setattr(sr, "ist_today", lambda: date(2026, 11, 23))
    monkeypatch.setattr(ld, "_session_closed", lambda now=None: True)
    monkeypatch.setattr(sr, "_session_closed", lambda now=None: True)
    monkeypatch.setattr(ld, "SECURITY_MASTER_PATH", MASTER)
    real = ld.compute_live_dashboard
    monkeypatch.setattr(sr, "compute_live_dashboard",
                        lambda fetch_prices, as_of: real(fetch_prices=fetch_prices, master_path=MASTER, as_of=as_of))


@pytest.fixture(scope="module")
def target():
    return SESSIONS[-6].date()  # a completed historical session inside the OOS window


def test_replay_matches_backtester_return_for_that_session(target):
    out = sr.replay_session(target, fetch_prices=_fetch_from_snapshot,
                            fetch_minutes=_minutes_from_daily, fetch_daily=_daily, save=False)
    prev = pd.Timestamp(out["previous_session"])

    # Independent recomputation with the backtester itself (costs off, since
    # the replay carries the book and trades nothing during the session).
    master = PointInTimeSecurityMaster.from_file(MASTER)
    state = ld.compute_live_dashboard(fetch_prices=_fetch_from_snapshot, master_path=MASTER, as_of=prev.date())
    w = state["windows"]
    prices = {c: SNAPSHOT[c].dropna() for c in SNAPSHOT.columns if c != ld.BENCHMARK}
    idx = SESSIONS[SESSIONS <= pd.Timestamp(target)]
    train_idx = idx[(idx >= pd.Timestamp(w["train_start"])) & (idx <= pd.Timestamp(w["train_end"]))]
    streams_train, streams_t = {}, {}
    for cfg in ld.SLEEVES:
        streams_train[cfg.name] = rank_sleeve_returns(
            prices, pd.Timestamp(w["train_start"]), pd.Timestamp(w["train_end"]), cfg,
            trading_index=train_idx, security_master=master)
        streams_t[cfg.name] = rank_sleeve_returns(
            prices, pd.Timestamp(w["test_start"]), pd.Timestamp(target), cfg,
            trading_index=idx, security_master=master, cost_bps=0.0)
    weights = _portfolio_weights(pd.DataFrame(streams_train).reindex(train_idx).fillna(0.0), "inverse_volatility")
    expected_raw = sum(streams_t[n].loc[pd.Timestamp(target)] * wt for n, wt in weights.items())

    scale = out["book"]["vol_target_scale"]
    assert out["result"]["book_return_eod"] / scale == pytest.approx(expected_raw, abs=1e-6)


def test_book_uses_only_data_through_previous_close(target, monkeypatch):
    seen = {}

    def spy(symbols, start, end):
        seen["end"] = end
        return _fetch_from_snapshot(symbols, start, end)

    out = sr.replay_session(target, fetch_prices=spy, fetch_minutes=_minutes_from_daily,
                            fetch_daily=_daily, save=False)
    assert seen["end"] == pd.Timestamp(out["previous_session"])
    assert out["book"]["as_of_close"] == out["previous_session"]


def test_non_session_date_is_rejected():
    with pytest.raises(RuntimeError, match="not an NSE session"):
        sr.replay_session(date(2026, 11, 22), fetch_prices=_fetch_from_snapshot,
                          fetch_minutes=_minutes_from_daily, fetch_daily=_daily, save=False)


def test_returns_vs_prev_close_forward_fills_and_starts_flat():
    session = date(2026, 11, 23)
    minutes = pd.DataFrame({"A": [101.0, np.nan, 103.0]},
                           index=pd.to_datetime(["2026-11-23 09:17", "2026-11-23 09:18", "2026-11-23 09:20"]))
    r = sr.returns_vs_prev_close(minutes, pd.Series({"A": 100.0}), session)
    assert r.loc["2026-11-23 09:15", "A"] == 0.0          # before first trade
    assert r.loc["2026-11-23 09:18", "A"] == pytest.approx(0.01)  # forward-filled
    assert r.loc["2026-11-23 09:19", "A"] == pytest.approx(0.01)
    assert r.loc["2026-11-23 15:29", "A"] == pytest.approx(0.03)
    assert len(r) == 375                                   # 09:15 … 15:29


def test_pair_return_matches_backtester_convention():
    rets = pd.DataFrame({"A": [0.02], "B": [0.01]})
    assert sr.pair_return(rets, "A", "B", 1, 0.5).iloc[0] == pytest.approx((0.02 - 0.005) / 1.5)
    assert sr.pair_return(rets, "A", "B", -1, 0.5).iloc[0] == pytest.approx(-(0.02 - 0.005) / 1.5)


def test_sleeve_book_after_matches_backtester_schedule():
    """The replay's book must be the one rank_sleeve_returns actually holds."""
    master = PointInTimeSecurityMaster.from_file(MASTER)
    prices = {c: SNAPSHOT[c].dropna() for c in SNAPSHOT.columns if c != ld.BENCHMARK}
    panel = pd.DataFrame(prices).reindex(SESSIONS)
    cfg = ld.SLEEVES[3]  # reversal_1m, weekly rebalance — the most decisions
    test_start = SESSIONS[-504]
    stream = rank_sleeve_returns(prices, test_start, SESSIONS[-1], cfg,
                                 trading_index=SESSIONS, security_master=master, cost_bps=0.0)
    rets = panel.pct_change()
    for pos in range(len(SESSIONS) - 40, len(SESSIONS)):
        book = ld.sleeve_book_after(panel, cfg, test_start, pos - 1, master)
        if not book["long"]:
            continue
        expected = rets.iloc[pos][book["long"]].mean() * cfg.gross_exposure
        assert stream.iloc[stream.index.get_loc(SESSIONS[pos])] == pytest.approx(expected, abs=1e-12), SESSIONS[pos]


def test_rebalance_day_is_not_left_flat():
    """Regression: the decision session used to earn 0 (book released a day early)."""
    master = PointInTimeSecurityMaster.from_file(MASTER)
    prices = {c: SNAPSHOT[c].dropna() for c in SNAPSHOT.columns if c != ld.BENCHMARK}
    cfg = ld.SLEEVES[3]
    stream = rank_sleeve_returns(prices, SESSIONS[-504], SESSIONS[-1], cfg,
                                 trading_index=SESSIONS, security_master=master, cost_bps=0.0)
    active = stream.iloc[cfg.lookback_days + cfg.skip_days + 2:]
    assert (active == 0.0).mean() < 0.02  # was ~20% for a 5-session rebalance

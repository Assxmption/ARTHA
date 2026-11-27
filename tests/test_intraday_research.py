"""Intraday research harness: causality, execution timing, costs, gate plumbing."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from app.quant import intraday_research as ir


def _cube(D=30, S=20, seed=0):
    rng = np.random.default_rng(seed)
    K = ir.SLOTS
    steps = rng.normal(0, 0.001, size=(D, K, S))
    close = 100 * np.exp(np.cumsum(steps.reshape(D * K, S), axis=0)).reshape(D, K, S)
    open_ = np.concatenate([close[:, :1] * (1 + rng.normal(0, 0.0005, (D, 1, S))), close[:, :-1]], axis=1)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(D)]
    return ir.Cube(days, [f"S{i}" for i in range(S)], open_, close)


def _truncate(cube, d, k):
    """Same cube with everything after (day d, slot k) unknown (NaN), like a live partial day."""
    o, c = cube.open.copy(), cube.close.copy()
    o[d, k + 1:] = np.nan
    c[d, k + 1:] = np.nan
    o[d + 1:] = np.nan
    c[d + 1:] = np.nan
    return ir.Cube(cube.days, cube.symbols, o, c)


@pytest.mark.parametrize("name", ["xs_reversal_30m", "gap_fade", "intraday_momentum"])
def test_strategies_are_causal(name):
    cube = _cube()
    ca = np.zeros((cube.shape[0], cube.shape[2]), dtype=bool)
    full = ir.strategy_weights(name, cube, ca)
    d = 20
    for k in (0, 6, 11, 17, 40, 68, 70):
        part = ir.strategy_weights(name, _truncate(cube, d, k), ca)
        np.testing.assert_allclose(part[d, k], full[d, k], err_msg=f"{name} slot {k} uses future data")


def test_pairs_strategy_is_causal():
    cube = _cube(D=40, S=6, seed=3)
    ca = np.zeros((40, 6), dtype=bool)
    pairs = [ir.PairSpec(0, 1, 0.9, "S0/S1"), ir.PairSpec(2, 3, 1.1, "S2/S3")]
    full = ir.intraday_pairs(cube, ca, pairs, lookback=30)
    for k in (10, 30, 60):
        part = ir.intraday_pairs(_truncate(cube, 35, k), ca, pairs, lookback=30)
        np.testing.assert_allclose(part[35, k], full[35, k])


def test_next_open_fill_and_square_off():
    D, K, S = 2, ir.SLOTS, 1
    o = np.full((D, K, S), 100.0)
    o[0, 10:, 0] = 110.0         # jumps at the open of slot 10 and stays there
    c = o.copy()
    cube = ir.Cube([date(2025, 1, 1), date(2025, 1, 2)], ["X"], o, c)
    W = np.zeros((D, K, S))
    W[0, 9:, 0] = 1.0            # decided at close of slot 9 → filled at open of slot 10: misses the jump
    free = ir.CostModel(slippage_bps=0, stt_sell=0, exchange=0, sebi=0, stamp_buy=0)
    assert ir.simulate(cube, W, free).iloc[0] == pytest.approx(0.0)
    W[:] = 0
    W[0, 8:, 0] = 1.0            # decided at close of slot 8 → filled at open of 9: captures 100→110
    assert ir.simulate(cube, W, free).iloc[0] == pytest.approx(0.10)
    # Positions never survive past the square-off, even if weights say so.
    o2 = o.copy()
    o2[0, ir.EXIT_SLOT + 1:, 0] = 200.0
    cube2 = ir.Cube(cube.days, ["X"], o2, o2.copy())
    W[:] = 0
    W[0, 60:, 0] = 1.0
    assert ir.simulate(cube2, W, free).iloc[0] == pytest.approx(0.0)


def test_costs_are_side_aware_and_round_trip():
    cm = ir.CostModel()
    assert cm.sell > cm.buy                         # STT is charged on the sell side
    assert 8e-4 < cm.buy + cm.sell < 1.1e-3         # ≈ 9.5 bp round trip at 3 bp slippage
    D, K, S = 1, ir.SLOTS, 1
    o = np.full((D, K, S), 100.0)
    cube = ir.Cube([date(2025, 1, 1)], ["X"], o, o.copy())
    W = np.zeros((D, K, S))
    W[0, 5:20, 0] = 1.0                              # one long round trip, flat prices
    assert ir.simulate(cube, W, cm).iloc[0] == pytest.approx(-(cm.buy + cm.sell))
    W[0, 5:20, 0] = -1.0                             # a short pays sell on entry, buy on exit
    assert ir.simulate(cube, W, cm).iloc[0] == pytest.approx(-(cm.buy + cm.sell))


def test_corporate_action_mask_keeps_genuine_shocks():
    cube = _cube(D=10, S=3, seed=1)
    cube.open[4:, :, 0] *= 0.7                               # genuine −30 % gap that persists: kept
    cube.close[4:, :, 0] *= 0.7
    cube.open[6, :, 1] = cube.close[5, -1, 1] * 0.5          # 1:1 bonus at the open: masked
    cube.close[6, :, 1] = cube.close[5, -1, 1] * 0.5
    ca = ir.corporate_action_mask(cube, {"S2": [str(cube.days[8])]})
    assert ca[6, 1] and ca[8, 2]
    assert not ca[4, 0] and not ca[5, 0]
    assert ir.shock_days(cube, ca) >= 1


def test_evaluate_reports_gate_and_hypotheses(monkeypatch):
    cube = _cube(D=200, S=15, seed=2)
    monkeypatch.setattr(ir, "select_pairs", lambda cube, train_days, top=10: [ir.PairSpec(0, 1, 1.0, "S0/S1")])
    rep = ir.evaluate(cube, oos_sessions=60)
    assert rep["hypotheses_tested"] == 4
    assert set(rep["strategies"]) == set(ir.STRATEGIES)
    for r in rep["strategies"].values():
        assert r["oos"]["num_trading_days"] == 60
        assert r["validated"] is False             # 60 sessions can never clear the 504-session gate
        assert "oos_stress_5bp" in r

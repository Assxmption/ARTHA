"""Dated cost schedules (Budget 2026-27 F&O STT change)."""

from datetime import date

import pytest

from app.quant import intraday_research as ir
from app.quant.costs import schedule_for


def test_futures_stt_switches_on_1_april_2026():
    assert schedule_for("index_futures", date(2026, 3, 31)).stt_sell == pytest.approx(0.0002)
    assert schedule_for("index_futures", date(2026, 4, 1)).stt_sell == pytest.approx(0.0005)
    assert schedule_for("options_premium", date(2026, 4, 1)).stt_sell == pytest.approx(0.0015)


def test_equity_intraday_matches_cost_model_defaults():
    sch = schedule_for("equity_intraday", date(2026, 11, 27))
    cm = ir.CostModel()
    assert (cm.stt_sell, cm.exchange, cm.stamp_buy) == (sch.stt_sell, sch.exchange, sch.stamp_buy)
    buy, sell = sch.per_side(3.0)
    assert (cm.buy, cm.sell) == (pytest.approx(buy), pytest.approx(sell))


def test_futures_round_trip_cheaper_than_stocks_but_not_dramatically():
    fut = ir.CostModel.for_segment("index_futures", slippage_bps=0.5, on=date(2026, 11, 27))
    stk = ir.CostModel()
    rt_fut, rt_stk = fut.buy + fut.sell, stk.buy + stk.sell
    assert 5e-4 < rt_fut < 8e-4            # ≈ 6–7 bp after the STT hike
    assert rt_fut < rt_stk < 2 * rt_fut


def test_unknown_segment_raises():
    with pytest.raises(KeyError):
        schedule_for("crypto")

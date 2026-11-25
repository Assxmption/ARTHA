"""Tests for the signal-validation trust boundary."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.factstore.schemas import OptionsSignal, QuantSignal, SignalType
from app.quant.backtest import BacktestMetrics, WalkForwardResult
from app.quant.signal_injector import _validation_kwargs


def _good_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.2,
        annualized_return=0.1,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=0.2,
        calmar_ratio=0.5,
        win_rate=0.55,
        num_trades=100,
        num_trading_days=504,
        volatility=0.12,
        avg_trade_return=0.001,
        profit_factor=1.2,
    )


def test_walk_forward_evidence_is_serialisable_and_complete():
    result = WalkForwardResult(
        strategy_name="fixture_strategy",
        in_sample_metrics=_good_metrics(),
        out_of_sample_metrics=_good_metrics(),
        is_validated=True,
    )
    evidence = result.evidence()
    assert evidence["validation_method"] == "walk_forward_oos"
    assert evidence["validation_oos_days"] == 504
    assert evidence["validation_id"].startswith("wf-")
    assert evidence["validated"] is True


def test_legacy_backtest_report_is_not_authoritative():
    assert _validation_kwargs({"version": "v5.4"}, "raw") == {"validated": False}


def test_validated_quant_signal_requires_evidence():
    with pytest.raises(ValidationError):
        QuantSignal(
            job_id="validation-test",
            source="fixture",
            symbol_or_pair="TCS-INFY",
            signal_type=SignalType.MOMENTUM,
            value=1.0,
            validated=True,
            as_of=date(2026, 11, 3),
        )


def test_validated_options_signal_requires_evidence():
    with pytest.raises(ValidationError):
        OptionsSignal(
            job_id="validation-test",
            source="fixture",
            symbol_or_index="NIFTY",
            strategy_type="IRON_CONDOR",
            net_premium=100.0,
            max_profit=100.0,
            max_loss=900.0,
            regime_label="SIDEWAYS",
            validated=True,
            as_of=date(2026, 11, 3),
        )

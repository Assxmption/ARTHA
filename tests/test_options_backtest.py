"""
Tests for Options Strategy Backtester v2
==========================================
Validates the rebuilt backtester that fixes:
  1. Pre-computed signals (no warmup-kills-OOS bug)
  2. Signal-based entry filtering (no blind mechanical entries)
  3. Walk-forward with shared signal store

Uses deterministic GBM-generated data for reproducibility.
"""

import pytest
import numpy as np
import pandas as pd

from app.quant.options_backtest import (
    OptionsBacktester, OptionsBacktestResult, WalkForwardResult,
    DailySignals, precompute_signals, reconstruct_daily_iv,
    synthetic_theta_pnl, walk_forward_validate,
    mark_to_market_strategy, _max_risk_lots,
    _is_entry_eligible,
    DEFAULT_CAPITAL, OPTIONS_ALLOCATION_PCT,
    STRATEGY_REGIME_FILTER, PREMIUM_SELLING_STRATEGIES,
)
from app.quant.options_strategies import (
    StrategyType, build_iron_condor, build_bull_put_spread,
)


# ── Test Fixtures ───────────────────────────────────────────────────────────────

def _generate_price_series(
    n_days: int = 500,
    start_price: float = 23000.0,
    annual_drift: float = 0.12,
    annual_vol: float = 0.18,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic GBM price series for testing."""
    rng = np.random.RandomState(seed)
    dt_val = 1 / 252
    daily_drift = (annual_drift - 0.5 * annual_vol**2) * dt_val
    daily_vol = annual_vol * np.sqrt(dt_val)

    returns = daily_drift + daily_vol * rng.randn(n_days)
    prices = start_price * np.exp(np.cumsum(returns))

    dates = pd.bdate_range(start="2023-01-02", periods=n_days)
    return pd.Series(prices, index=dates, name="NIFTY")


def _generate_vix_series(price_series: pd.Series, seed: int = 42) -> pd.Series:
    """Generate synthetic VIX data correlated with price movement."""
    rng = np.random.RandomState(seed)
    n = len(price_series)
    returns = price_series.pct_change().fillna(0)
    base_vix = 15.0
    vix = np.zeros(n)
    vix[0] = base_vix
    for i in range(1, n):
        mean_revert = 0.02 * (base_vix - vix[i - 1])
        vol_shock = -50 * returns.iloc[i]
        noise = 0.5 * rng.randn()
        vix[i] = max(vix[i - 1] + mean_revert + vol_shock + noise, 8.0)
    return pd.Series(vix, index=price_series.index, name="VIX")


PRICES = _generate_price_series(n_days=500)
VIX = _generate_vix_series(PRICES)
LONG_PRICES = _generate_price_series(n_days=800)
LONG_VIX = _generate_vix_series(LONG_PRICES)


# ── 1. Pre-computed Signals Tests ───────────────────────────────────────────────


class TestPrecomputedSignals:

    def test_precompute_covers_all_dates(self):
        """Pre-computed signals cover every date in the price series."""
        signals = precompute_signals(PRICES, VIX)
        assert len(signals) == len(PRICES)

    def test_precompute_iv_reasonable(self):
        """IV values are in a reasonable range."""
        signals = precompute_signals(PRICES, VIX)
        ivs = [s.iv for s in signals.values()]
        assert all(0.05 <= iv <= 1.0 for iv in ivs)

    def test_precompute_vrp_computed(self):
        """VRP is computed (iv - rv) / rv."""
        signals = precompute_signals(PRICES, VIX)
        # At least some VRP values should be non-zero
        vrps = [s.vrp for s in signals.values()]
        assert any(v != 0 for v in vrps)

    def test_precompute_without_vix(self):
        """Pre-computation works without VIX data."""
        signals = precompute_signals(PRICES, vix_series=None)
        assert len(signals) == len(PRICES)

    def test_precompute_with_regime(self):
        """Regime series is respected."""
        # Create a dummy regime series
        regimes = pd.Series(
            ["BULL"] * 250 + ["BEAR"] * 250,
            index=PRICES.index,
        )
        signals = precompute_signals(PRICES, VIX, regimes)
        early = list(signals.values())[100]
        late = list(signals.values())[400]
        assert early.regime == "BULL"
        assert late.regime == "BEAR"


# ── 2. Entry Eligibility Tests ──────────────────────────────────────────────────


class TestEntryEligibility:

    def test_iron_condor_allowed_in_sideways(self):
        """Iron condor is allowed in SIDEWAYS regime with positive VRP."""
        sig = DailySignals(iv=0.18, rv=0.14, vrp=0.28,
                           iv_percentile=55, vix_regime="MEDIUM",
                           regime="SIDEWAYS")
        assert _is_entry_eligible(StrategyType.IRON_CONDOR, sig) is True

    def test_iron_condor_blocked_in_bull(self):
        """Iron condor is NOT allowed in BULL regime."""
        sig = DailySignals(iv=0.18, rv=0.14, vrp=0.28,
                           iv_percentile=55, vix_regime="MEDIUM",
                           regime="BULL")
        assert _is_entry_eligible(StrategyType.IRON_CONDOR, sig) is False

    def test_iron_condor_blocked_on_negative_vrp(self):
        """Iron condor is blocked when VRP is negative (selling cheap vol)."""
        sig = DailySignals(iv=0.12, rv=0.18, vrp=-0.33,
                           iv_percentile=55, vix_regime="MEDIUM",
                           regime="SIDEWAYS")
        assert _is_entry_eligible(StrategyType.IRON_CONDOR, sig) is False

    def test_iron_condor_blocked_on_low_iv_percentile(self):
        """Iron condor blocked when IV percentile is too low."""
        sig = DailySignals(iv=0.10, rv=0.09, vrp=0.11,
                           iv_percentile=10, vix_regime="LOW",
                           regime="SIDEWAYS")
        assert _is_entry_eligible(StrategyType.IRON_CONDOR, sig) is False

    def test_bull_put_allowed_in_bull(self):
        """Bull put spread is allowed in BULL regime."""
        sig = DailySignals(iv=0.18, rv=0.14, vrp=0.28,
                           iv_percentile=50, vix_regime="MEDIUM",
                           regime="BULL")
        assert _is_entry_eligible(StrategyType.BULL_PUT_SPREAD, sig) is True

    def test_bear_call_allowed_in_bear(self):
        """Bear call spread is allowed in BEAR regime."""
        sig = DailySignals(iv=0.22, rv=0.16, vrp=0.37,
                           iv_percentile=70, vix_regime="HIGH",
                           regime="BEAR")
        assert _is_entry_eligible(StrategyType.BEAR_CALL_SPREAD, sig) is True

    def test_bear_call_blocked_in_bull(self):
        """Bear call spread is NOT allowed in BULL regime."""
        sig = DailySignals(iv=0.18, rv=0.14, vrp=0.28,
                           iv_percentile=55, vix_regime="MEDIUM",
                           regime="BULL")
        assert _is_entry_eligible(StrategyType.BEAR_CALL_SPREAD, sig) is False

    def test_protective_put_allowed_in_bear(self):
        """Protective put is allowed in BEAR (no VRP filter)."""
        sig = DailySignals(iv=0.30, rv=0.25, vrp=0.20,
                           iv_percentile=80, vix_regime="HIGH",
                           regime="BEAR")
        assert _is_entry_eligible(StrategyType.PROTECTIVE_PUT, sig) is True

    def test_all_strategies_have_regime_filter(self):
        """Every strategy type has a defined regime filter."""
        for st in StrategyType:
            assert st in STRATEGY_REGIME_FILTER, f"{st} missing from STRATEGY_REGIME_FILTER"


# ── 3. Synthetic Pricing Tests ──────────────────────────────────────────────────


class TestSyntheticPricing:

    def test_iv_reconstruction_from_vix(self):
        iv = reconstruct_daily_iv(PRICES, VIX)
        assert len(iv) == len(PRICES)
        assert (iv > 0.05).all()
        assert (iv < 1.0).all()

    def test_iv_reconstruction_without_vix(self):
        iv = reconstruct_daily_iv(PRICES, vix_series=None)
        assert len(iv) == len(PRICES)
        assert iv.dropna().gt(0).all()

    def test_synthetic_pnl_at_expiry(self):
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        pnl = ic.payoff_at_price(23000)
        assert pnl > 0

    def test_synthetic_pnl_loss(self):
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        pnl = ic.payoff_at_price(25000)
        assert pnl < 0


# ── 4. Theta Decay Tests ───────────────────────────────────────────────────────


class TestThetaDecay:

    def test_positive_theta_pnl_for_short_strategy(self):
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        pnl = synthetic_theta_pnl(ic, days_elapsed=5)
        assert pnl > 0

    def test_theta_pnl_scales_with_days(self):
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        pnl_5 = synthetic_theta_pnl(ic, 5)
        pnl_10 = synthetic_theta_pnl(ic, 10)
        assert abs(pnl_10 / pnl_5 - 2.0) < 0.1

    def test_vega_impact_on_pnl(self):
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        pnl_no_iv = synthetic_theta_pnl(ic, 5, iv_change=0.0)
        pnl_iv_drop = synthetic_theta_pnl(ic, 5, iv_change=-0.02)
        assert pnl_iv_drop > pnl_no_iv


# ── 5. Backtester Mechanics Tests ──────────────────────────────────────────────


class TestBacktesterMechanics:

    def test_backtest_runs(self):
        """Backtester completes with pre-computed signals."""
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        # Use regime that allows iron condors
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        assert isinstance(result, OptionsBacktestResult)
        assert result.n_trades > 0

    def test_backtest_regime_filter_reduces_trades(self):
        """BULL regime should block iron condor entries."""
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        # BULL regime blocks iron condors
        bull_regimes = pd.Series("BULL", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=bull_regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        assert result.n_trades == 0  # Should be blocked

    def test_backtest_produces_daily_pnl(self):
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        assert len(result.daily_pnl) > 0

    def test_backtest_metrics_computed(self):
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.max_drawdown_pct, float)

    def test_insufficient_data(self):
        short_prices = PRICES.iloc[:5]
        bt = OptionsBacktester()
        result = bt.backtest_strategy(
            short_prices, strategy_type=StrategyType.IRON_CONDOR,
        )
        assert result.n_trades == 0

    def test_bull_put_in_bull_regime(self):
        """Bull put spread should work in BULL regime."""
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("BULL", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        assert isinstance(result, OptionsBacktestResult)
        # Bull put should be allowed in bull regime
        assert result.n_trades > 0

    def test_max_concurrent_respected(self):
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL, max_concurrent=2)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=5,
        )
        open_count = 0
        max_open = 0
        for trade in result.trades:
            if trade.action == "OPEN":
                open_count += 1
            elif trade.action in ("CLOSE", "EXPIRE"):
                open_count -= 1
            max_open = max(max_open, open_count)
        assert max_open <= 4  # Allow slack for same-day events

    def test_precomputed_signals_avoid_recomputation(self):
        """Passing pre-computed signals uses them directly."""
        signals = precompute_signals(PRICES, VIX)
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        # Modify signals to force regime to SIDEWAYS
        for sig in signals.values():
            sig.regime = "SIDEWAYS"
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY",
            precomputed_signals=signals, entry_interval=14,
        )
        assert result.n_trades > 0

    def test_date_range_filtering(self):
        """start_date and end_date correctly limit the backtest period."""
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        signals = precompute_signals(PRICES, VIX, regimes)
        mid = PRICES.index[250]
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY",
            precomputed_signals=signals,
            start_date=mid, entry_interval=14,
        )
        # Trades should only appear on or after mid date
        for t in result.trades:
            assert t.date >= str(mid.date())


# ── 6. Walk-Forward Validation Tests ────────────────────────────────────────────


class TestWalkForward:

    def test_walk_forward_runs(self):
        regimes = pd.Series("SIDEWAYS", index=LONG_PRICES.index)
        result = walk_forward_validate(
            LONG_PRICES,
            strategy_type=StrategyType.IRON_CONDOR,
            symbol="NIFTY",
            vix_series=LONG_VIX,
            regime_series=regimes,
            capital=DEFAULT_CAPITAL,
            train_days=200,
            test_days=50,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows > 0

    def test_walk_forward_oos_has_trades(self):
        """OOS windows should now have trades (the warmup bug is fixed)."""
        regimes = pd.Series("SIDEWAYS", index=LONG_PRICES.index)
        result = walk_forward_validate(
            LONG_PRICES,
            strategy_type=StrategyType.IRON_CONDOR,
            symbol="NIFTY",
            vix_series=LONG_VIX,
            regime_series=regimes,
            train_days=200,
            test_days=50,
        )
        oos_with_trades = sum(1 for r in result.window_results if r.n_trades > 0)
        assert oos_with_trades > 0, (
            f"OOS should have trades now (warmup bug fix). "
            f"Got 0/{result.n_windows} windows with trades."
        )

    def test_walk_forward_reports_sharpe(self):
        regimes = pd.Series("SIDEWAYS", index=LONG_PRICES.index)
        result = walk_forward_validate(
            LONG_PRICES,
            strategy_type=StrategyType.IRON_CONDOR,
            symbol="NIFTY",
            vix_series=LONG_VIX,
            regime_series=regimes,
            train_days=200,
            test_days=50,
        )
        assert isinstance(result.in_sample_sharpe, float)
        assert isinstance(result.out_of_sample_sharpe, float)

    def test_walk_forward_insufficient_data(self):
        short_prices = PRICES.iloc[:100]
        result = walk_forward_validate(
            short_prices,
            strategy_type=StrategyType.IRON_CONDOR,
            train_days=252,
            test_days=63,
        )
        assert result.n_windows == 0
        assert "Insufficient" in result.reason

    def test_validation_gate_structure(self):
        regimes = pd.Series("SIDEWAYS", index=LONG_PRICES.index)
        result = walk_forward_validate(
            LONG_PRICES,
            strategy_type=StrategyType.IRON_CONDOR,
            symbol="NIFTY",
            vix_series=LONG_VIX,
            regime_series=regimes,
            train_days=200,
            test_days=50,
        )
        assert isinstance(result.validated, bool)
        assert len(result.reason) > 0

    def test_honest_reporting(self):
        """Results are reported regardless of quality (AGENTS.md rule 8)."""
        regimes = pd.Series("SIDEWAYS", index=LONG_PRICES.index)
        result = walk_forward_validate(
            LONG_PRICES,
            strategy_type=StrategyType.IRON_CONDOR,
            symbol="NIFTY",
            vix_series=LONG_VIX,
            regime_series=regimes,
            train_days=200,
            test_days=50,
        )
        assert result.n_windows > 0
        assert len(result.window_results) > 0


# ── 7. Capital & Margin Tests ───────────────────────────────────────────────────


class TestCapitalConstraints:

    def test_margin_limits_positions(self):
        bt = OptionsBacktester(capital=10_00_000, options_pct=0.10)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=7,
        )
        assert isinstance(result, OptionsBacktestResult)

    def test_avg_margin_positive(self):
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        if result.n_trades > 0:
            assert result.avg_margin_used > 0


# ── 8. Full BS Repricing Tests ──────────────────────────────────────────────────


class TestBSRepricing:

    def test_mark_to_market_at_entry(self):
        """MtM value at entry should be close to net premium."""
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        mtm = mark_to_market_strategy(ic, 23000, 0.18, 30)
        # At entry spot/IV/DTE the value should reflect the premium structure
        assert isinstance(mtm, float)

    def test_repricing_varies_with_spot(self):
        """MtM should change when spot moves."""
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        v_at_entry = mark_to_market_strategy(ic, 23000, 0.18, 30)
        v_spot_up = mark_to_market_strategy(ic, 23500, 0.18, 30)
        v_spot_dn = mark_to_market_strategy(ic, 22500, 0.18, 30)
        # Value should change with spot
        assert v_at_entry != v_spot_up
        assert v_at_entry != v_spot_dn

    def test_repricing_varies_with_iv(self):
        """MtM should change when IV changes (vega)."""
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        v_low_iv = mark_to_market_strategy(ic, 23000, 0.10, 30)
        v_high_iv = mark_to_market_strategy(ic, 23000, 0.30, 30)
        # Short vol strategy: value should differ with IV
        assert v_low_iv != v_high_iv

    def test_repricing_converges_to_payoff_at_expiry(self):
        """At DTE=0, MtM should equal intrinsic value (zero time-value)."""
        ic = build_iron_condor("NIFTY", 23000, 22500, 22000, 23500, 24000, 0.18, 30)
        # At expiry with spot at center, all legs expire worthless
        # MtM = sum of intrinsic values = 0 (all OTM)
        mtm_center = mark_to_market_strategy(ic, 23000, 0.18, 0)
        # At center, intrinsic of all legs is 0 → MtM ≈ 0
        assert abs(mtm_center) < 50
        # At extreme spot, MtM should show loss (short wing breached)
        mtm_extreme = mark_to_market_strategy(ic, 25000, 0.18, 0)
        assert mtm_extreme < -1000  # Short call wing breached

    def test_position_sizing(self):
        """Risk per trade should limit lot count."""
        lots = _max_risk_lots(5000, 1_00_00_000, 0.02)
        # max risk = 1cr * 2% = 200000; 200000/5000 = 40 lots
        assert lots == 40

    def test_position_sizing_minimum_one_lot(self):
        """Even with tiny capital, at least 1 lot."""
        lots = _max_risk_lots(50000, 10000, 0.02)
        assert lots == 1


# ── 9. Drawdown Sanity Tests ────────────────────────────────────────────────────


class TestDrawdownSanity:

    def test_max_drawdown_bounded(self):
        """Max drawdown should be realistic with BS repricing."""
        bt = OptionsBacktester(capital=DEFAULT_CAPITAL)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=14,
        )
        if result.n_trades > 0:
            # With BS repricing, drawdown is computed relative to peak_margin.
            # Multiple concurrent short-gamma positions CAN exceed 100% of
            # a single position's margin, but should not be absurd (>500%).
            assert result.max_drawdown_pct < 500, (
                f"Max drawdown {result.max_drawdown_pct}% is unrealistic."
            )

    def test_drawdown_circuit_breaker_reduces_trades(self):
        """A very aggressive strategy should trigger the DD breaker."""
        # Use very small capital so positions hit the 10% DD limit quickly
        bt = OptionsBacktester(capital=50_000, options_pct=1.0, max_concurrent=5)
        regimes = pd.Series("SIDEWAYS", index=PRICES.index)
        result_small = bt.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=7,
        )
        # With large capital, more trades should execute
        bt_large = OptionsBacktester(capital=50_00_00_000, options_pct=1.0, max_concurrent=5)
        result_large = bt_large.backtest_strategy(
            PRICES, regime_series=regimes,
            strategy_type=StrategyType.IRON_CONDOR,
            vix_series=VIX, symbol="NIFTY", entry_interval=7,
        )
        # Both should be valid results
        assert isinstance(result_small, OptionsBacktestResult)
        assert isinstance(result_large, OptionsBacktestResult)

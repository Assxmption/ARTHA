"""
ARTHA Multi-Strategy Portfolio Backtest
========================================
End-to-end script that:
  1. Loads real NSE stock data from cache
  2. Runs each strategy independently
  3. Combines return streams via vol-targeted Kelly allocation
  4. Reports combined portfolio metrics

Strategies:
  A. Stat-Arb (pairs trading with Kalman filter + cointegration)
  B. Factor Model (L/S quintiles — momentum + fundamental composite)
  C. Equity Momentum (regime-conditioned signal weights)
  D. Mean Reversion (z-score based, SIDEWAYS regime)

Advanced Pricing (Heston + Merton):
  - Used for options strategy sleeve (separate from equity strategies)
  - Vol surface construction for more accurate entry/exit pricing

Target: Combined Sharpe > 1.3 via √N diversification.

All computation is deterministic (AGENTS.md rule 1).
Reference: docs/ARTHA_ARCHITECTURE.md §4.3, §6.3
"""

from __future__ import annotations

import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.regime import detect_regime


from app.quant.factor_backtest import backtest_factor_model
from app.quant.multi_strategy import combine_strategies as combine_strategy_returns
from app.quant.options_backtest import (
    OptionsBacktester, precompute_signals as precompute_options_signals,
)
from app.quant.options_strategies import StrategyType
from app.quant.pca_statarb import backtest_pca_statarb
from app.data.vix import get_vix_close

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("artha.multi_strategy_backtest")

CACHE_DIR = Path("data_cache")
REPORT_DIR = Path("docs/backtest_reports")


# ── Data Loading ────────────────────────────────────────────────────────────────

def load_stock_universe(min_days: int = 500) -> dict[str, pd.DataFrame]:
    """Load all cached stock data with sufficient history."""
    stocks = {}
    for f in CACHE_DIR.glob("*_v5.parquet"):
        sym = f.stem.replace("_v5", "")
        if sym.startswith("IDX_"):
            continue
        try:
            df = pd.read_parquet(f)
            if len(df) >= min_days and "Close" in df.columns:
                stocks[sym] = df
        except Exception as e:
            logger.debug("Skipping %s: %s", sym, e)
    logger.info("Loaded %d stocks with ≥%d days of history", len(stocks), min_days)
    return stocks


def load_nifty_index() -> pd.DataFrame:
    """Load NIFTY 50 index data."""
    f = CACHE_DIR / "IDX_NSEI_v5.parquet"
    if f.exists():
        return pd.read_parquet(f)
    raise FileNotFoundError("NIFTY index data not found")


def load_fundamentals() -> dict[str, dict]:
    """Load cached fundamental data."""
    fundamentals = {}
    for f in CACHE_DIR.glob("*_fund_v5.json"):
        sym = f.stem.replace("_fund_v5", "")
        try:
            with open(f) as fh:
                data = json.load(fh)
                fundamentals[sym] = data
        except Exception:
            pass
    logger.info("Loaded fundamentals for %d stocks", len(fundamentals))
    return fundamentals


# ── Strategy Runners ────────────────────────────────────────────────────────────

def run_statarb_strategy(stocks: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Stat-arb strategy — DISABLED.

    After extensive testing (log spreads, half-life filters, PnL clipping),
    cointegration-based pairs trading consistently loses money on Indian
    equities over 12 years. Root cause: structural breaks (HDFC merger,
    Adani regime, delistings, sector rotations) break pair relationships
    faster than the 504-day discovery window can adapt.

    Keeping the function for future work with alternative pair discovery
    (e.g., factor-based matching, sector-relative spread).
    """
    logger.info("═══ Strategy A: Statistical Arbitrage ═══")
    logger.info("  DISABLED — consistently negative Sharpe on Indian equities")

    close_prices = {}
    for sym, df in stocks.items():
        if 'Close' in df.columns:
            close_prices[sym] = df['Close'].squeeze()

    panel = pd.DataFrame(close_prices).dropna(how='all')
    return pd.Series(0.0, index=panel.index, name="statarb")


def run_factor_strategy(
    stocks: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
) -> pd.Series:
    """
    Run factor model — LONG-ONLY top quintile.

    Key fix for 10yr robustness: the L/S version shorts bottom-quintile
    stocks that can include structural winners (Adani, Trent, etc.)
    which go up 30x over a decade. A -2900% short-side loss on one stock
    wipes out all other gains. Long-only avoids this while still
    capturing the quality/value/momentum factor premium.
    """
    logger.info("═══ Strategy B: Factor Model (Long-Only) ═══")

    result = backtest_factor_model(
        price_data=stocks,
        fundamentals=fundamentals,
        rebalance_days=21,
        lookback_momentum=252,
        quintile_pct=0.20,
        long_only=True,  # Avoid catastrophic short-side losses
    )

    logger.info(
        "  Factor model: %d rebalances, Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
        result.n_rebalances, result.sharpe_ratio,
        result.annualized_return * 100, result.max_drawdown * 100,
    )

    if result.n_rebalances == 0:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="factors")

    ret = result.daily_returns
    ret.name = "factors"
    return ret


def run_momentum_strategy(
    stocks: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
) -> pd.Series:
    """
    Regime-conditioned 6-1 cross-sectional momentum.

    Uses 6-month return skipping 1 month (Jegadeesh & Titman).
    BULL: long-only top quintile. SIDEWAYS/BEAR: full L/S.
    Vol filter: exclude top-quartile volatile stocks.

    This version (simple 6-1 only) achieved Sharpe 0.39 with
    the lowest correlation to factors (0.09) of any variant tested.
    Multi-horizon blends (12-1) increased correlation to 0.14 without
    improving signal — net negative for the portfolio.
    """
    logger.info("═══ Strategy C: Cross-Sectional Momentum ═══")

    close_dfs = {s: df['Close'].squeeze() for s, df in stocks.items() if 'Close' in df.columns}
    panel = pd.DataFrame(close_dfs).dropna(how='all')

    if len(panel) < 170:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="momentum")

    regimes = detect_regime(nifty['Close'].squeeze())
    regime_aligned = regimes.reindex(panel.index, method='ffill').fillna('UNKNOWN')

    mom_6_1 = panel.pct_change(126).shift(21)
    vol_20d = panel.pct_change().rolling(20).std()
    daily_returns = panel.pct_change()
    rebalance_interval = 21

    dates = panel.index
    portfolio_returns = pd.Series(0.0, index=dates)

    for t in range(150, len(dates), rebalance_interval):
        regime = regime_aligned.iloc[t] if t < len(regime_aligned) else 'UNKNOWN'

        scores = mom_6_1.iloc[t].dropna()
        vols = vol_20d.iloc[t].dropna()

        if len(scores) < 10:
            continue

        common = scores.index.intersection(vols.index)
        scores = scores[common]
        vols = vols[common]
        vol_threshold = vols.quantile(0.75)
        low_vol_mask = vols <= vol_threshold
        scores = scores[low_vol_mask]

        if len(scores) < 6:
            continue

        n_per_leg = max(1, len(scores) // 5)
        ranked = scores.sort_values(ascending=False)
        long_syms = ranked.index[:n_per_leg].tolist()

        if regime in ('SIDEWAYS', 'BEAR'):
            short_syms = ranked.index[-n_per_leg:].tolist()
        else:
            short_syms = []

        hold_end = min(t + rebalance_interval, len(dates))
        for d in range(t + 1, hold_end):
            day_rets = daily_returns.iloc[d]
            long_r = day_rets[long_syms].dropna().mean() if long_syms else 0
            short_r = day_rets[short_syms].dropna().mean() if short_syms else 0

            if short_syms:
                portfolio_returns.iloc[d] = long_r - short_r
            else:
                portfolio_returns.iloc[d] = long_r * 0.5

        n_legs = 2 if short_syms else 1
        portfolio_returns.iloc[min(t + 1, len(dates) - 1)] -= 0.0015 * n_legs

    portfolio_returns.name = "momentum"

    arr = portfolio_returns.values[150:]
    if len(arr) > 20:
        sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info("  Momentum: Sharpe=%.3f, Ann.Return=%.2f%%", sharpe, ann_ret * 100)

    return portfolio_returns


def run_mean_reversion_strategy(
    stocks: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
) -> pd.Series:
    """
    Run a long-only mean-reversion (dip-buying) strategy.

    Key insight: In a structural bull market (India 2021-2026),
    shorting "overbought" stocks is suicidal — they're trending,
    not reverting. The profitable MR trade is buying oversold dips
    when capitulation volume confirms the move was an overreaction.

    Entry:  z-score < -2.5σ AND volume > 1.5× 20-day average
    Exit:   After 3 trading days (mechanical, no discretion)
    Sizing: Equal-weight across all triggered stocks
    """
    logger.info("═══ Strategy D: Mean Reversion (Dip-Buy) ═══")

    # Build close and volume panels
    close_dfs = {}
    vol_dfs = {}
    for s, df in stocks.items():
        if 'Close' in df.columns and 'Volume' in df.columns:
            close_dfs[s] = df['Close']
            vol_dfs[s] = df['Volume']

    panel = pd.DataFrame(close_dfs).dropna(how='all')
    vol_panel = pd.DataFrame(vol_dfs).reindex(panel.index)

    if len(panel) < 30:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="mean_reversion")

    # Z-scores: deviation from 20-day SMA
    sma20 = panel.rolling(20).mean()
    std20 = panel.rolling(20).std()
    z_scores = (panel - sma20) / std20.replace(0, np.nan)

    # Volume relative to 20-day average
    vol_sma20 = vol_panel.rolling(20).mean()
    vol_ratio = vol_panel / vol_sma20.replace(0, np.nan)

    daily_returns = panel.pct_change()
    portfolio_returns = pd.Series(0.0, index=panel.index)

    HOLD_DAYS = 3
    Z_ENTRY = -2.5       # Only buy deeply oversold
    VOL_CONFIRM = 1.5    # Volume must be 1.5× average (capitulation signal)
    COST_BPS = 20        # 20bps round-trip

    for t in range(25, len(panel) - HOLD_DAYS, 1):  # Daily scan
        z_today = z_scores.iloc[t].dropna()
        vr_today = vol_ratio.iloc[t].dropna()

        # Oversold + volume spike = capitulation → buy
        common = z_today.index.intersection(vr_today.index)
        z_filt = z_today[common]
        vr_filt = vr_today[common]

        signals = z_filt[(z_filt < Z_ENTRY) & (vr_filt > VOL_CONFIRM)]

        if len(signals) == 0:
            continue

        # Hold for HOLD_DAYS, equal weight
        for d in range(1, HOLD_DAYS + 1):
            idx = t + d
            if idx >= len(panel):
                break

            day_ret = daily_returns.iloc[idx]
            rets = []
            for sym in signals.index:
                if sym in day_ret.index:
                    r = day_ret[sym]
                    if not np.isnan(r):
                        rets.append(r)

            if rets:
                avg_ret = np.mean(rets)
                if d == 1:
                    avg_ret -= COST_BPS / 10000
                # Avoid overwriting — add to existing (multiple signals can overlap)
                portfolio_returns.iloc[idx] += avg_ret / max(len(signals), 1)

    portfolio_returns.name = "mean_reversion"

    arr = portfolio_returns.values[25:]
    if len(arr) > 20:
        sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info("  Mean Reversion: Sharpe=%.3f, Ann.Return=%.2f%%", sharpe, ann_ret * 100)

    return portfolio_returns


def run_trend_following_strategy(
    stocks: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
) -> pd.Series:
    """
    Run a time-series momentum / trend-following strategy.

    Reference: Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum"
    (AQR working paper, later in JFE).

    Unlike cross-sectional momentum which ranks stocks against each other,
    time-series momentum trades each stock's own trend:
    - Long if price > 50-day EMA AND 50-day EMA > 200-day EMA (golden cross)
    - Cash (flat) otherwise

    This captures the trending behavior of Indian equities without
    shorting (which is expensive and dangerous in a bull market).

    Position sizing: volatility-targeted at 15% annualized per stock,
    then equal-weighted across all long positions.
    """
    logger.info("═══ Strategy E: Trend Following ═══")

    close_dfs = {s: df['Close'] for s, df in stocks.items() if 'Close' in df.columns}
    panel = pd.DataFrame(close_dfs).dropna(how='all')

    if len(panel) < 210:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="trend_following")

    # EMAs
    ema50 = panel.ewm(span=50, adjust=False).mean()
    ema200 = panel.ewm(span=200, adjust=False).mean()

    # Daily returns
    daily_returns = panel.pct_change()

    # Rolling volatility for position sizing (20-day)
    rolling_vol = daily_returns.rolling(20).std() * np.sqrt(252)

    portfolio_returns = pd.Series(0.0, index=panel.index)
    TARGET_VOL = 0.15
    REBAL_INTERVAL = 5  # Weekly

    for t in range(205, len(panel), REBAL_INTERVAL):
        # Signal: price > EMA50 > EMA200 (golden cross + above trend)
        price_above_ema50 = panel.iloc[t] > ema50.iloc[t]
        ema50_above_ema200 = ema50.iloc[t] > ema200.iloc[t]
        trend_signal = price_above_ema50 & ema50_above_ema200

        long_syms = trend_signal[trend_signal].dropna().index.tolist()

        if not long_syms:
            continue

        # Hold period
        hold_end = min(t + REBAL_INTERVAL, len(panel))
        for d in range(t + 1, hold_end):
            day_rets = daily_returns.iloc[d]
            vols = rolling_vol.iloc[t]

            weighted_ret = 0.0
            total_weight = 0.0

            for sym in long_syms:
                if sym in day_rets.index and sym in vols.index:
                    r = day_rets[sym]
                    v = vols[sym]
                    if np.isnan(r) or np.isnan(v) or v < 0.01:
                        continue
                    # Vol-target: scale each stock to TARGET_VOL
                    weight = min(TARGET_VOL / v, 3.0)  # Cap at 3x leverage
                    weighted_ret += r * weight
                    total_weight += weight

            if total_weight > 0:
                # Normalize by number of stocks (not total weight)
                portfolio_returns.iloc[d] = weighted_ret / len(long_syms)

        # Transaction cost
        portfolio_returns.iloc[min(t + 1, len(panel) - 1)] -= 0.001

    portfolio_returns.name = "trend_following"

    arr = portfolio_returns.values[205:]
    if len(arr) > 20:
        sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info("  Trend Following: Sharpe=%.3f, Ann.Return=%.2f%%", sharpe, ann_ret * 100)

    return portfolio_returns


def run_short_term_reversal_strategy(
    stocks: dict[str, pd.DataFrame],
) -> pd.Series:
    """
    Run a short-term reversal strategy (1-week).

    Reference: Jegadeesh (1990), "Evidence of Predictable Behavior of
    Security Returns"; Lo & MacKinlay (1990).

    The effect: stocks that fell the most over the past week tend to
    bounce back the next week, and vice versa. This is one of the
    STRONGEST anomalies in Indian markets because:
    - High retail participation → overreaction to news
    - Herding behavior → mean-reversion after panic
    - Market microstructure (bid-ask bounce)

    Long bottom-quintile (last week's losers), weekly rebalance.
    Long-only in bull markets (no shorting winners).
    """
    logger.info("═══ Strategy F: Short-Term Reversal ═══")

    close_dfs = {s: df['Close'] for s, df in stocks.items() if 'Close' in df.columns}
    panel = pd.DataFrame(close_dfs).dropna(how='all')

    if len(panel) < 30:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="st_reversal")

    # Weekly returns (5-day)
    weekly_ret = panel.pct_change(5)
    daily_returns = panel.pct_change()

    portfolio_returns = pd.Series(0.0, index=panel.index)
    REBAL = 5  # Weekly

    for t in range(10, len(panel), REBAL):
        # Last week's returns
        past_week = weekly_ret.iloc[t].dropna()

        if len(past_week) < 10:
            continue

        # Bottom quintile = last week's losers → buy (expect bounce)
        n_per_leg = max(1, len(past_week) // 5)
        ranked = past_week.sort_values()
        losers = ranked.index[:n_per_leg].tolist()  # Worst performers

        # Hold for 1 week
        hold_end = min(t + REBAL, len(panel))
        for d in range(t + 1, hold_end):
            day_rets = daily_returns.iloc[d]
            long_r = day_rets[losers].dropna().mean() if losers else 0

            if not np.isnan(long_r):
                portfolio_returns.iloc[d] = long_r

        # Transaction cost
        portfolio_returns.iloc[min(t + 1, len(panel) - 1)] -= 0.002  # 20bps

    portfolio_returns.name = "st_reversal"

    arr = portfolio_returns.values[10:]
    if len(arr) > 20:
        sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info("  ST Reversal: Sharpe=%.3f, Ann.Return=%.2f%%", sharpe, ann_ret * 100)

    return portfolio_returns


def run_ml_alpha_strategy(
    stocks: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
) -> pd.Series:
    """
    Strategy G: ML Alpha Ensemble.

    Trains a 3-model ensemble (HistGBM + Ridge + RF) on 40+ signals
    using walk-forward purged cross-validation. Trades top/bottom
    quintile of alpha predictions.

    This is the most compute-intensive strategy — it builds the full
    signal matrix, trains models, and generates daily predictions.

    Key design decisions:
    - 21-day purge gap between train and test (prevents look-ahead)
    - Subsampled to weekly signal dates (keeps compute tractable)
    - Cross-sectionally ranked predictions (market-neutral)
    - L/S quintile portfolio with monthly rebalance
    """
    from app.quant.signals import build_signal_matrix
    from app.quant.ml_alpha import train_alpha_model

    logger.info("═══ Strategy G: ML Alpha Ensemble ═══")

    # Build inputs for signal matrix
    prices = {}
    volumes = {}
    sector_map = {}

    for sym, df in stocks.items():
        if 'Close' in df.columns:
            prices[sym] = df['Close']
        if 'Volume' in df.columns:
            volumes[sym] = df['Volume']
        # Get sector from fundamentals
        fund = fundamentals.get(sym, {})
        sector_map[sym] = fund.get('sector', 'Other')

    if len(prices) < 20:
        logger.warning("Too few stocks for ML alpha: %d", len(prices))
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="ml_alpha")

    # Get panel dates — use the full date range (signal builder handles per-stock
    # NaN via its min_history check). The old set.intersection approach failed because
    # recently-listed stocks (JIOFIN 2023, NYKAA 2021) restricted dates to <500.
    panel = pd.DataFrame(prices).dropna(how='all')
    all_dates = sorted(panel.index.tolist())

    # Weekly subsample for signal computation (every 5th trading day)
    # Start after 1yr warmup for enough per-stock history
    weekly_dates = pd.DatetimeIndex(all_dates[252::5])
    if len(weekly_dates) < 100:
        logger.warning("Too few weekly dates for ML: %d", len(weekly_dates))
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="ml_alpha")

    logger.info("Building signal matrix: %d weekly dates, %d stocks...", 
                len(weekly_dates), len(prices))

    # Build signal matrix
    signal_matrix = build_signal_matrix(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sector_map=sector_map,
        dates=weekly_dates,
        min_history=252,
    )

    if signal_matrix.empty or len(signal_matrix) < 1000:
        logger.warning("Signal matrix too small: %d rows", len(signal_matrix))
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="ml_alpha")

    logger.info("Signal matrix: %d rows × %d columns", 
                len(signal_matrix), len(signal_matrix.columns))

    # Build forward returns (5-day forward return for weekly rebalance)
    fwd_records = []
    for date in weekly_dates:
        for sym in prices:
            p = prices[sym]
            if date not in p.index:
                continue
            idx = p.index.get_loc(date)
            if not isinstance(idx, int):
                idx = int(idx) if isinstance(idx, np.integer) else 0
            fwd_idx = idx + 5
            if fwd_idx < len(p):
                fwd_ret = (p.iloc[fwd_idx] - p.iloc[idx]) / p.iloc[idx]
                fwd_records.append({
                    "date": date,
                    "symbol": sym,
                    "fwd_return": fwd_ret,
                })

    fwd_df = pd.DataFrame(fwd_records).set_index(["date", "symbol"])

    if len(fwd_df) < 1000:
        logger.warning("Forward returns too small: %d", len(fwd_df))
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="ml_alpha")

    # Train the ensemble
    logger.info("Training ML ensemble (walk-forward, step=21)...")
    result = train_alpha_model(
        signal_matrix=signal_matrix,
        forward_returns=fwd_df,
        min_train_days=100,  # 100 weekly obs = ~2 years
        step_size=4,         # 4 weekly obs = ~1 month
        n_estimators_xgb=150,
        max_depth_xgb=4,
        learning_rate_xgb=0.05,
        ridge_alpha=1.0,
        n_estimators_rf=80,
    )

    if result.predictions.empty or result.n_folds < 3:
        logger.warning("ML training failed: %d folds, IC=%.4f", 
                       result.n_folds, result.oos_ic)
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="ml_alpha")

    logger.info("ML training: %d folds, OOS IC=%.4f, OOS R²=%.4f",
                result.n_folds, result.oos_ic, result.oos_r2)

    # Top 5 features
    top_features = sorted(result.feature_importance.items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top features: %s", 
                [(f, f"{v:.4f}") for f, v in top_features])

    # Convert predictions to daily portfolio returns
    predictions = result.predictions
    pred_dates = predictions.index.get_level_values("date").unique().sort_values()

    # Build a panel of daily returns
    panel = pd.DataFrame({s: df['Close'] for s, df in stocks.items() 
                          if 'Close' in df.columns}).dropna(how='all')
    daily_returns = panel.pct_change()
    portfolio_returns = pd.Series(0.0, index=panel.index)

    N_LONG = 15        # Top 15 stocks (broader quintile)
    REBAL_FREQ = 4     # Only rebalance every 4th prediction (~monthly)
    TURNOVER_THRESH = 0.30  # Only rebalance if >30% of top quintile changes

    prev_long_syms = []
    pred_count = 0

    for i in range(len(pred_dates)):
        pred_date = pred_dates[i]
        if pred_date not in panel.index:
            continue

        pred_count += 1

        # Only rebalance every REBAL_FREQ predictions (~monthly)
        if pred_count % REBAL_FREQ != 1 and prev_long_syms:
            continue

        t = panel.index.get_loc(pred_date)
        if not isinstance(t, int):
            t = int(t) if isinstance(t, np.integer) else 0

        # Get alpha scores for this date
        day_preds = predictions.xs(pred_date, level="date")["alpha_score"]

        # Long-only: top N_LONG stocks by alpha score
        valid = day_preds.dropna().sort_values(ascending=False)
        if len(valid) < N_LONG:
            continue

        new_long_syms = valid.index[:N_LONG].tolist()

        # Turnover filter: skip if portfolio is barely changing
        if prev_long_syms:
            overlap = len(set(new_long_syms) & set(prev_long_syms))
            turnover = 1 - overlap / N_LONG
            if turnover < TURNOVER_THRESH:
                # Not enough change to justify trading costs
                continue

        # Compute TC based on actual turnover
        if prev_long_syms:
            n_changed = len(set(new_long_syms) - set(prev_long_syms))
            tc_bps = 15 * n_changed / N_LONG  # 15bps proportional to turnover
        else:
            tc_bps = 15  # Full TC on initial entry

        prev_long_syms = new_long_syms

        # Hold until next rebalance
        next_pred_idx = i + REBAL_FREQ
        if next_pred_idx < len(pred_dates):
            next_pred_date = pred_dates[next_pred_idx]
            if next_pred_date in panel.index:
                hold_end = panel.index.get_loc(next_pred_date)
                if not isinstance(hold_end, int):
                    hold_end = int(hold_end) if isinstance(hold_end, np.integer) else t + 21
            else:
                hold_end = min(t + 21, len(panel))
        else:
            hold_end = min(t + 21, len(panel))

        for d in range(t + 1, hold_end):
            if d >= len(panel):
                break

            day_ret = daily_returns.iloc[d]
            # Long-only: equal-weight top quintile, scaled to half exposure
            long_r = day_ret[new_long_syms].dropna().mean()

            if not np.isnan(long_r):
                # Scale to 50% exposure (comparable vol to L/S strategies)
                portfolio_returns.iloc[d] = long_r * 0.5

        # Apply TC on rebalance day
        if t + 1 < len(panel):
            portfolio_returns.iloc[t + 1] -= tc_bps / 10000

    portfolio_returns.name = "ml_alpha"

    arr = portfolio_returns.values[252:]
    if len(arr) > 20:
        sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info("  ML Alpha: Sharpe=%.3f, Ann.Return=%.2f%%, IC=%.4f",
                    sharpe, ann_ret * 100, result.oos_ic)

    return portfolio_returns


# ── PCA Statistical Arbitrage Strategy ──────────────────────────────────────────

def run_pca_statarb_strategy(
    stocks: dict[str, pd.DataFrame],
    n_components: int = 5,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
) -> pd.Series:
    """
    PCA-based cross-sectional stat-arb on the stock universe.

    Extracts systematic risk factors via rolling PCA, then trades
    mean reversion in the idiosyncratic residuals.  Complements
    the Kalman-filter pair-wise stat-arb (which is currently DISABLED
    on Indian equities due to structural breaks).

    Zero LLM calls — entirely deterministic (AGENTS.md rule 1).
    """
    logger.info("═══ Strategy I: PCA Statistical Arbitrage ═══")

    # Build close-price panel
    close_prices = {}
    for sym, df in stocks.items():
        if 'Close' in df.columns:
            s = df['Close'].squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            close_prices[sym] = s

    panel = pd.DataFrame(close_prices).dropna(how='all')

    # Need sufficient breadth for cross-sectional PCA
    if panel.shape[1] < 10:
        logger.warning(
            "  PCA stat-arb needs ≥10 stocks; got %d. Skipping.",
            panel.shape[1],
        )
        return pd.Series(0.0, index=panel.index, name="pca_statarb")

    if panel.shape[0] < 400:
        logger.warning(
            "  Insufficient history for PCA stat-arb: %d days. Skipping.",
            panel.shape[0],
        )
        return pd.Series(0.0, index=panel.index, name="pca_statarb")

    # Forward-fill gaps (holidays, suspensions), then drop remaining NaN columns
    panel = panel.ffill().dropna(axis=1, how='any')
    logger.info("  Universe: %d stocks × %d days", panel.shape[1], panel.shape[0])

    # Run the backtest
    try:
        returns = backtest_pca_statarb(
            prices=panel,
            n_components=n_components,
            entry_z=entry_z,
            exit_z=exit_z,
            tc_bps=15.0,  # Same as statarb.py convention
            window=252,
            lookback_z=20,
        )
        returns.name = "pca_statarb"
        return returns
    except Exception as e:
        logger.error("  PCA stat-arb backtest failed: %s", e, exc_info=True)
        return pd.Series(0.0, index=panel.index, name="pca_statarb")


# ── Options Vol-Carry Strategy ──────────────────────────────────────────────────

def run_options_vol_carry_strategy(
    nifty: pd.DataFrame,
    capital: float = 5_00_00_000.0,
) -> pd.Series:
    """
    Options volatility-carry strategy — IRON_CONDOR on NIFTY.

    Harvests the Volatility Risk Premium (VRP) by selling iron condors
    on NIFTY when regime and VRP conditions are favorable.  Uses
    defined-risk IRON_CONDOR (not SHORT_STRANGLE) per design decision:
    bounded max loss, clear margin requirement, no undefined tail risk.

    The daily P&L from OptionsBacktester is normalized to returns using
    peak margin as the denominator (conservative, matches _compute_metrics
    convention).  This allows the tangency allocator to compare the
    options sleeve on equal footing with equity strategy return streams.

    Zero LLM calls — entirely deterministic (AGENTS.md rule 1).

    Trade-off: IRON_CONDOR collects less premium than SHORT_STRANGLE
    (the wings cost ~15-25% of gross credit), but the defined risk
    makes position sizing deterministic and margin requirements
    predictable.  The SHORT_STRANGLE's stop_loss_pct is a soft bound;
    gaps through it are real in Indian markets (circuit-limit opens).
    """
    logger.info("═══ Strategy H: Options Vol-Carry (IRON_CONDOR on NIFTY) ═══")

    # Extract NIFTY close prices
    if "Close" not in nifty.columns:
        logger.error("  NIFTY data missing 'Close' column")
        return pd.Series(0.0, index=nifty.index, name="options_vol_carry")

    nifty_close = nifty["Close"].squeeze()
    if isinstance(nifty_close, pd.DataFrame):
        nifty_close = nifty_close.iloc[:, 0]
    nifty_close = nifty_close.dropna()

    if len(nifty_close) < 400:
        logger.warning("  Insufficient NIFTY data (%d days) for options backtest", len(nifty_close))
        return pd.Series(0.0, index=nifty_close.index, name="options_vol_carry")

    # Load India VIX for IV reconstruction
    try:
        vix_series = get_vix_close()
        if len(vix_series) > 0:
            logger.info("  Loaded India VIX: %d data points", len(vix_series))
        else:
            vix_series = None
            logger.info("  No VIX data available; using RV×1.2 proxy")
    except Exception as e:
        logger.warning("  VIX fetch failed (%s); using RV×1.2 proxy", e)
        vix_series = None

    # Detect market regimes on NIFTY (for signal filtering)
    try:
        regime_series = detect_regime(nifty_close)
        logger.info("  Regimes detected: %d data points", len(regime_series))
    except Exception as e:
        logger.warning("  Regime detection failed (%s); proceeding without", e)
        regime_series = None

    # Pre-compute all signals on the FULL series once (critical v2 fix)
    all_signals = precompute_options_signals(
        nifty_close,
        vix_series=vix_series,
        regime_series=regime_series,
    )
    logger.info("  Pre-computed options signals for %d dates", len(all_signals))

    # Run the backtest with IRON_CONDOR (defined risk)
    backtester = OptionsBacktester(
        capital=capital,
        options_pct=0.30,     # 30% of total capital for options
        max_concurrent=5,
    )

    result = backtester.backtest_strategy(
        close_prices=nifty_close,
        regime_series=regime_series,
        vix_series=vix_series,
        strategy_type=StrategyType.IRON_CONDOR,
        symbol="NIFTY",
        entry_interval=7,       # New trade every 7 trading days
        dte_at_entry=30,        # 30-day options
        early_exit_pct=0.50,    # Take profit at 50% of max profit
        stop_loss_pct=1.0,      # Close at 100% of max loss
        precomputed_signals=all_signals,
    )

    # Log backtest results
    logger.info(
        "  IRON_CONDOR backtest: %d trades, Win=%.1f%%, P&L=₹%.0f, "
        "Sharpe=%.3f, Sortino=%.3f, MaxDD=%.2f%%",
        result.n_trades,
        result.win_rate * 100,
        result.total_pnl,
        result.sharpe_ratio,
        result.sortino_ratio,
        result.max_drawdown_pct,
    )

    if result.n_trades == 0:
        logger.warning("  No trades generated — check regime/VRP filters")
        return pd.Series(0.0, index=nifty_close.index, name="options_vol_carry")

    # Convert daily P&L to daily returns.
    #
    # Bug fix (Sep 2026): The original code used avg_margin_used as the
    # denominator, which can be tiny on sparse windows (few active trades,
    # margin near zero most days). This produced amplified garbage returns
    # (e.g. -3201% total, -258% annualized) that poisoned the portfolio.
    #
    # Correct approach: use the maximum of:
    #   (a) peak_margin_used — the actual concurrent capital at risk during
    #       the backtest, as computed inside OptionsBacktester, and
    #   (b) capital * 0.10 — a floor based on the allocated capital (assumes
    #       at least 10% of the options sleeve capital is always reserved
    #       as margin for this strategy, even on quiet days).
    #
    # This ensures returns are never amplified beyond what the real capital
    # allocation implies, while still reflecting true risk when margin is
    # legitimately high.
    daily_pnl = result.daily_pnl
    options_capital = capital * 0.10  # 10% of total capital = options sleeve
    denominator = max(result.peak_margin_used, options_capital, 1.0)
    logger.info(
        "  Return denominator: peak_margin=%.0f, capital_floor=%.0f → using %.0f",
        result.peak_margin_used, options_capital, denominator,
    )

    # Align daily P&L to the NIFTY index dates
    # The backtester returns daily_pnl as a list aligned to the dates
    # it iterated over — which is the sorted nifty_close.index
    dates = nifty_close.index.sort_values()
    if len(daily_pnl) <= len(dates):
        # Pad or align
        pnl_series = pd.Series(0.0, index=dates, name="options_vol_carry")
        pnl_series.iloc[:len(daily_pnl)] = daily_pnl
    else:
        pnl_series = pd.Series(
            daily_pnl[:len(dates)], index=dates, name="options_vol_carry",
        )

    # Normalize P&L to returns
    returns = pnl_series / denominator
    returns.name = "options_vol_carry"

    # Report performance on the return series
    arr = returns.values
    non_zero = arr[arr != 0]
    if len(non_zero) > 20:
        ret_sharpe = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        ann_ret = np.mean(arr) * 252
        logger.info(
            "  Options return stream: Sharpe=%.3f, Ann.Return=%.2f%%",
            ret_sharpe, ann_ret * 100,
        )
    else:
        logger.warning("  Options return stream has <20 non-zero days")

    return returns


# ── Portfolio Combination ───────────────────────────────────────────────────────

def combine_strategies(
    strategy_returns: dict[str, pd.Series],
    target_vol: float = 0.15,
    weight_lookback: int = 252,
    rebalance_every: int = 63,
) -> pd.DataFrame:
    """
    Combine strategy return streams using vol-targeting + rolling-window
    correlation-aware allocation.

    Bug fix (Sep 2026): The original code computed tangency weights using
    the full sample, which is look-ahead bias — the allocator could see
    future performance. This version uses a rolling window (default 252
    trading days) and rebalances every `rebalance_every` days. Each
    rebalance uses only data available up to that date.

    Args:
        strategy_returns: {name: daily_return_series}
        target_vol: target annualized volatility per strategy
        weight_lookback: days of history to estimate weights
        rebalance_every: days between weight re-estimations
    """
    logger.info("═══ Combining Strategies (Rolling-Window Allocation) ═══")

    # Align all series to common dates
    df = pd.DataFrame(strategy_returns)
    df = df.dropna(how='all').fillna(0)
    n_days = len(df)
    columns = list(df.columns)

    # Per-strategy full-sample metrics (informational only — NOT used for allocation)
    for col in columns:
        arr = df[col].values
        if np.std(arr) > 0:
            sharpe = (np.mean(arr) / np.std(arr)) * np.sqrt(252)
            ann_ret = np.mean(arr) * 252
            ann_vol = np.std(arr) * np.sqrt(252)
        else:
            sharpe = ann_ret = ann_vol = 0
        logger.info(
            "  %s: Full-sample Sharpe=%.3f, Ann.Ret=%.2f%%, Ann.Vol=%.2f%% "
            "(informational — NOT used for allocation weights)",
            col, sharpe, ann_ret * 100, ann_vol * 100,
        )

    # Vol-target each strategy individually (rolling, no look-ahead)
    scaled = pd.DataFrame(index=df.index)
    for col in columns:
        rolling_vol = df[col].rolling(60, min_periods=20).std() * np.sqrt(252)
        scale = target_vol / rolling_vol.replace(0, np.nan).ffill().fillna(target_vol)
        scale = scale.clip(upper=3.0)  # Cap leverage at 3x
        scaled[col] = df[col] * scale

    # ── Rolling-window tangency allocation (no look-ahead) ─────────
    # Recompute allocation weights every `rebalance_every` days using
    # only data up to each rebalance date.  Between rebalances, weights
    # are held constant.
    daily_rf = (1 + 0.065) ** (1/252) - 1  # India 10Y
    n_assets = len(columns)

    # Start with equal weights until we have enough data
    current_weights = {col: 1.0 / n_assets for col in columns}
    weight_history = []  # [(date, weights_dict), ...]
    combined = pd.Series(0.0, index=df.index, name="portfolio")

    warmup = max(weight_lookback, 60)
    last_rebalance = -rebalance_every  # Force rebalance at first eligible day

    for i in range(n_days):
        # Check if we should rebalance
        if i >= warmup and (i - last_rebalance) >= rebalance_every:
            # Compute weights using data [i-weight_lookback : i] only
            window_data = scaled.iloc[max(0, i - weight_lookback):i]

            if len(window_data) >= 60:
                mu = (window_data.mean() - daily_rf) * 252
                sample_cov = window_data.cov() * 252

                # Ledoit-Wolf shrinkage
                target_cov = np.diag(np.diag(sample_cov.values))
                n_obs = len(window_data)
                delta = min(0.3, n_assets / max(n_obs, 1))
                shrunk_cov = (1 - delta) * sample_cov.values + delta * target_cov
                shrunk_cov += np.eye(n_assets) * 1e-6  # Ridge for stability

                try:
                    cov_inv = np.linalg.inv(shrunk_cov)
                    raw_w = cov_inv @ mu.values
                    raw_w = np.maximum(raw_w, 0)  # Long-only

                    if raw_w.sum() > 1e-10:
                        raw_w /= raw_w.sum()
                        current_weights = {
                            col: float(w) for col, w in zip(columns, raw_w)
                        }
                    else:
                        raise ValueError("All tangency weights zero")
                except (np.linalg.LinAlgError, ValueError):
                    # Fallback: Sharpe²-proportional on the window
                    sharpes = {}
                    for col in columns:
                        arr = window_data[col].values
                        s = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
                        sharpes[col] = max(s, 0.01)
                    sharpe_sq = {k: v ** 2 for k, v in sharpes.items()}
                    total_sq = sum(sharpe_sq.values())
                    current_weights = {k: v / total_sq for k, v in sharpe_sq.items()}

                # Apply caps (35%) and floors (5%), then renormalize
                for k in current_weights:
                    current_weights[k] = max(min(current_weights[k], 0.35), 0.05)
                total_w = sum(current_weights.values())
                current_weights = {k: v / total_w for k, v in current_weights.items()}

                last_rebalance = i
                weight_history.append((df.index[i], dict(current_weights)))

        # Apply current weights
        combined.iloc[i] = sum(
            scaled[col].iloc[i] * current_weights[col] for col in columns
        )

    combined.name = "portfolio"

    # Log weight history summary
    if weight_history:
        logger.info("  %d weight rebalances performed", len(weight_history))
        # Log first, middle, and last rebalance weights
        for label, idx in [
            ("First", 0), ("Mid", len(weight_history) // 2), ("Last", -1),
        ]:
            date, w = weight_history[idx]
            logger.info(
                "  %s rebalance (%s): %s",
                label, date.strftime("%Y-%m-%d"),
                {k: f"{v:.1%}" for k, v in w.items()},
            )
    else:
        logger.warning("  No rebalances — insufficient data for rolling allocation")

    result_df = df.copy()
    result_df["portfolio"] = combined

    return result_df


def dynamic_hedge_overlay(
    portfolio_returns: pd.Series,
    index_returns: pd.Series,
    regimes: pd.Series,
    hedge_ratios: dict[str, float] | None = None,
    beta_lookback: int = 120,
    roll_cost_bps_year: int = 60,
) -> pd.Series:
    """
    Apply regime-conditional dynamic beta hedge to portfolio returns.

    This is the technique that transformed v5.3's raw Sharpe 0.37 → 1.16.
    The key insight: our long-only strategies carry significant market beta.
    By shorting NIFTY futures proportional to rolling beta × regime-dependent
    hedge ratio, we extract pure alpha from the beta-contaminated returns.

    Args:
        portfolio_returns: Raw portfolio return series
        index_returns: NIFTY index return series
        regimes: Regime labels (BULL/BEAR/SIDEWAYS) per date
        hedge_ratios: Regime → hedge fraction (0=no hedge, 1=full hedge)
        beta_lookback: Rolling window for beta estimation
        roll_cost_bps_year: Annual cost of rolling the hedge (NIFTY futures)

    Returns:
        Hedged portfolio return series
    """
    if hedge_ratios is None:
        # Conservative priors — NOT optimized on the backtest sample.
        #
        # Bug fix (Sep 2026): The original ratios {BULL: 0.15, BEAR: 0.80,
        # SIDEWAYS: 0.45} were tuned to maximize Sharpe on the full 12-year
        # sample, which is look-ahead bias. These conservative defaults are
        # based on general portfolio construction principles:
        #   - BEAR 0.70: hedge most of beta in drawdowns (well-supported in
        #     literature: Asness et al. 2012, "Value and Momentum Everywhere")
        #   - BULL 0.30: maintain some hedge even in bull markets to dampen
        #     drawdowns from sudden reversals (2020 COVID crash was ~35% in
        #     3 weeks during a "bull" regime)
        #   - SIDEWAYS 0.50: middle-of-road when regime is unclear
        #
        # If you want to optimize these, do it walk-forward: train on
        # [t-504 : t], validate on [t : t+252], sweep over a 3D grid.
        # Never optimize on the full sample.
        hedge_ratios = {"BULL": 0.30, "BEAR": 0.70, "SIDEWAYS": 0.50}

    logger.info("═══ Dynamic Hedge Overlay ═══")
    logger.info("  Hedge ratios: %s", hedge_ratios)

    # Align series
    aligned = pd.DataFrame({
        "port": portfolio_returns,
        "idx": index_returns,
    }).dropna()

    if len(aligned) < beta_lookback + 20:
        logger.warning("Insufficient data for hedge overlay: %d rows", len(aligned))
        return portfolio_returns

    hedged_returns = pd.Series(np.nan, index=aligned.index, name="hedged_portfolio")

    regime_counts = {"BULL": 0, "BEAR": 0, "SIDEWAYS": 0}
    betas = []

    for i in range(beta_lookback, len(aligned)):
        # Rolling beta
        window = aligned.iloc[i - beta_lookback:i]
        cov_matrix = np.cov(window["port"].values, window["idx"].values)
        var_idx = cov_matrix[1, 1]
        beta = np.clip(
            cov_matrix[0, 1] / var_idx if var_idx > 1e-10 else 1.0,
            0.3, 1.5  # Bound beta to reasonable range
        )
        betas.append(beta)

        # Get regime for this date
        date = aligned.index[i]
        regime = "SIDEWAYS"  # Default
        if not regimes.empty:
            prior = regimes.loc[regimes.index <= date]
            if not prior.empty:
                r = prior.iloc[-1]
                regime = r if isinstance(r, str) else (r.value if hasattr(r, 'value') else str(r))

        hedge_ratio = hedge_ratios.get(regime, 0.50)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Daily roll cost of the hedge
        daily_roll = (roll_cost_bps_year * hedge_ratio) / 10000 / 252

        # Hedged return = portfolio - hedge_ratio × beta × index - roll cost
        hedged_r = (
            aligned.iloc[i]["port"]
            - hedge_ratio * beta * aligned.iloc[i]["idx"]
            - daily_roll
        )
        hedged_returns.iloc[i] = hedged_r

    hedged_returns = hedged_returns.dropna()

    # Log summary
    avg_beta = np.mean(betas) if betas else 0
    logger.info("  Avg rolling beta: %.2f", avg_beta)
    for reg, count in regime_counts.items():
        if count > 0:
            logger.info("  %s: %d days (%.0f%%)",
                       reg, count, count / sum(regime_counts.values()) * 100)

    # Metrics comparison
    raw_arr = aligned["port"].values[beta_lookback:]
    hedged_arr = hedged_returns.values
    if len(raw_arr) > 20 and len(hedged_arr) > 20:
        raw_sharpe = (np.mean(raw_arr) / max(np.std(raw_arr), 1e-8)) * np.sqrt(252)
        hedged_sharpe = (np.mean(hedged_arr) / max(np.std(hedged_arr), 1e-8)) * np.sqrt(252)
        raw_dd = np.max(np.maximum.accumulate(np.cumsum(raw_arr)) - np.cumsum(raw_arr))
        hedged_dd = np.max(np.maximum.accumulate(np.cumsum(hedged_arr)) - np.cumsum(hedged_arr))
        logger.info("  Raw → Hedged: Sharpe %.3f → %.3f, MaxDD %.1f%% → %.1f%%",
                    raw_sharpe, hedged_sharpe, raw_dd * 100, hedged_dd * 100)

    return hedged_returns


# ── Reporting ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: np.ndarray) -> dict:
    """Compute comprehensive performance metrics."""
    # Filter NaN values (e.g. hedged_portfolio has NaN for first 120 days)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 20:
        return {"error": "insufficient data"}

    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns))
    n_years = n / 252

    ann_ret = mean_daily * 252
    ann_vol = std_daily * np.sqrt(252)
    sharpe = mean_daily / max(std_daily, 1e-8) * np.sqrt(252)

    down = returns[returns < 0]
    down_std = float(np.std(down)) if len(down) > 1 else std_daily
    sortino = mean_daily / max(down_std, 1e-8) * np.sqrt(252)

    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd))

    # Calmar ratio
    calmar = ann_ret / max(max_dd, 1e-8) if max_dd > 0 else 0

    # Win rate
    win_rate = float(np.sum(returns > 0) / n)

    # Skewness and kurtosis
    skew = float(np.mean(((returns - mean_daily) / max(std_daily, 1e-8)) ** 3))
    kurt = float(np.mean(((returns - mean_daily) / max(std_daily, 1e-8)) ** 4))

    return {
        "total_return": f"{float(np.sum(returns)) * 100:.2f}%",
        "annualized_return": f"{ann_ret * 100:.2f}%",
        "annualized_vol": f"{ann_vol * 100:.2f}%",
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown": f"{max_dd * 100:.2f}%",
        "calmar_ratio": round(float(calmar), 4),
        "win_rate": f"{win_rate * 100:.1f}%",
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n_trading_days": n,
        "n_years": round(n_years, 2),
    }


def save_report(result_df: pd.DataFrame, report_name: str = "multi_strategy"):
    """Save backtest report to docs/backtest_reports/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"{report_name}_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "strategies": {},
        "portfolio": {},
    }

    for col in result_df.columns:
        arr = result_df[col].values
        metrics = compute_metrics(arr)
        if col == "portfolio":
            report["portfolio"] = metrics
        else:
            report["strategies"][col] = metrics

    # Correlation matrix
    corr = result_df.drop(columns=["portfolio"], errors="ignore").corr()
    report["correlation_matrix"] = corr.to_dict()

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("  ARTHA Multi-Strategy Portfolio — Backtest Report")
    print("=" * 70)

    for name, metrics in report["strategies"].items():
        print(f"\n  📊 {name}:")
        for k, v in metrics.items():
            print(f"     {k:25s}: {v}")

    print(f"\n  🏆 COMBINED PORTFOLIO:")
    for k, v in report["portfolio"].items():
        print(f"     {k:25s}: {v}")

    print("\n  📐 Correlation Matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 70)

    return report


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    logger.info("ARTHA Multi-Strategy Backtest — Starting")

    # 1. Load data
    stocks = load_stock_universe(min_days=500)
    if len(stocks) < 10:
        logger.error("Insufficient stock data. Run data fetcher first.")
        return

    nifty = load_nifty_index()
    fundamentals = load_fundamentals()

    # 2. Run each strategy
    strategy_returns = {}

    # A. Stat-Arb
    try:
        statarb_ret = run_statarb_strategy(stocks)
        if statarb_ret.abs().sum() > 0:
            strategy_returns["statarb"] = statarb_ret
    except Exception as e:
        logger.error("Stat-arb failed: %s", e, exc_info=True)

    # B. Factor Model
    try:
        factor_ret = run_factor_strategy(stocks, fundamentals)
        if factor_ret.abs().sum() > 0:
            strategy_returns["factors"] = factor_ret
    except Exception as e:
        logger.error("Factor model failed: %s", e, exc_info=True)

    # C. Momentum
    try:
        momentum_ret = run_momentum_strategy(stocks, nifty)
        if momentum_ret.abs().sum() > 0:
            strategy_returns["momentum"] = momentum_ret
    except Exception as e:
        logger.error("Momentum failed: %s", e, exc_info=True)

    # D. Mean Reversion
    try:
        mr_ret = run_mean_reversion_strategy(stocks, nifty)
        if mr_ret.abs().sum() > 0:
            strategy_returns["mean_reversion"] = mr_ret
    except Exception as e:
        logger.error("Mean reversion failed: %s", e, exc_info=True)

    # E. Trend Following
    try:
        trend_ret = run_trend_following_strategy(stocks, nifty)
        if trend_ret.abs().sum() > 0:
            strategy_returns["trend_following"] = trend_ret
    except Exception as e:
        logger.error("Trend following failed: %s", e, exc_info=True)

    # F. Short-Term Reversal
    try:
        reversal_ret = run_short_term_reversal_strategy(stocks)
        if reversal_ret.abs().sum() > 0:
            strategy_returns["st_reversal"] = reversal_ret
    except Exception as e:
        logger.error("Short-term reversal failed: %s", e, exc_info=True)

    # G. ML Alpha Ensemble
    try:
        ml_ret = run_ml_alpha_strategy(stocks, fundamentals)
        if ml_ret.abs().sum() > 0:
            strategy_returns["ml_alpha"] = ml_ret
    except Exception as e:
        logger.error("ML alpha failed: %s", e, exc_info=True)

    # H. Options Vol-Carry (IRON_CONDOR on NIFTY)
    try:
        options_ret = run_options_vol_carry_strategy(nifty)
        if options_ret.abs().sum() > 0:
            strategy_returns["options_vol_carry"] = options_ret
    except Exception as e:
        logger.error("Options vol-carry failed: %s", e, exc_info=True)

    # I. PCA Statistical Arbitrage
    try:
        pca_ret = run_pca_statarb_strategy(stocks)
        if pca_ret.abs().sum() > 0:
            strategy_returns["pca_statarb"] = pca_ret
    except Exception as e:
        logger.error("PCA stat-arb failed: %s", e, exc_info=True)

    if not strategy_returns:
        logger.error("No strategies produced returns. Check data and modules.")
        return

    logger.info("Active strategies: %s", list(strategy_returns.keys()))

    # 3. Combine
    result_df = combine_strategies(strategy_returns)

    # ── Options sleeve correlation check ──────────────────────────────────
    # Log correlation of options_vol_carry vs each equity strategy.
    # If ρ > 0.4, the tangency optimizer will not treat it as genuinely
    # diversifying — flag this as a finding to report honestly, not a bug.
    if "options_vol_carry" in result_df.columns:
        equity_strategies = [
            col for col in result_df.columns
            if col not in ("options_vol_carry", "portfolio",
                           "hedged_portfolio", "hedged_vt")
        ]
        for eq_name in equity_strategies:
            if eq_name in result_df.columns:
                corr_val = result_df["options_vol_carry"].corr(result_df[eq_name])
                if not np.isnan(corr_val) and abs(corr_val) > 0.4:
                    logger.warning(
                        "⚠️  options_vol_carry ↔ %s correlation = %.3f (> 0.4). "
                        "Tangency allocator will not treat this as genuinely "
                        "diversifying. This is a finding, not a bug.",
                        eq_name, corr_val,
                    )
                else:
                    logger.info(
                        "  options_vol_carry ↔ %s correlation = %.3f ✓",
                        eq_name, corr_val if not np.isnan(corr_val) else 0.0,
                    )

    # 3.5. Dynamic Hedge Overlay (v5.3 technique: Sharpe 0.37 → 1.16)
    if nifty is not None and "Close" in nifty.columns:
        try:
            from app.quant.regime import detect_regime
            index_prices = nifty["Close"].squeeze()
            index_returns = index_prices.pct_change().dropna()
            regimes = detect_regime(index_prices)

            hedged = dynamic_hedge_overlay(
                portfolio_returns=result_df["portfolio"],
                index_returns=index_returns,
                regimes=regimes,
            )
            result_df["hedged_portfolio"] = hedged

            # Vol-targeted hedged portfolio
            hedged_vol = hedged.rolling(60, min_periods=20).std() * np.sqrt(252)
            vt_scale = (0.15 / hedged_vol.replace(0, np.nan).ffill().fillna(0.15)).clip(upper=3.0)
            result_df["hedged_vt"] = hedged * vt_scale

        except Exception as e:
            logger.error("Hedge overlay failed: %s", e, exc_info=True)

    # 4. Report
    report = save_report(result_df)

    # 5. Validation gate (AGENTS.md rule 6)
    # Use hedged portfolio Sharpe if available, else raw
    best_key = "hedged_portfolio" if "hedged_portfolio" in report.get("strategies", {}) else "portfolio"
    if best_key in report.get("strategies", {}):
        portfolio_sharpe = report["strategies"][best_key].get("sharpe_ratio", 0)
    else:
        portfolio_sharpe = report["portfolio"].get("sharpe_ratio", 0)
    logger.info("Portfolio Sharpe: %.4f", portfolio_sharpe)

    if portfolio_sharpe < 0:
        logger.warning("⚠️  Portfolio Sharpe is NEGATIVE. Review strategy signals.")
    elif portfolio_sharpe < 0.5:
        logger.warning("⚠️  Portfolio Sharpe < 0.5. Below minimum viable threshold.")
    elif portfolio_sharpe < 1.0:
        logger.info("📊 Portfolio Sharpe 0.5-1.0. Acceptable but needs more strategies.")
    elif portfolio_sharpe < 1.5:
        logger.info("✅ Portfolio Sharpe 1.0-1.5. Good — diversification working.")
    else:
        logger.info("🏆 Portfolio Sharpe > 1.5. Excellent — target achieved.")


if __name__ == "__main__":
    main()

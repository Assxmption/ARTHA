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
from app.quant.statarb_backtest import backtest_pair_universe
from app.quant.statarb import discover_pairs
from app.quant.factor_backtest import backtest_factor_model
from app.quant.multi_strategy import combine_strategies as combine_strategy_returns

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
    """Run stat-arb strategy on stock universe."""
    logger.info("═══ Strategy A: Statistical Arbitrage ═══")

    # Extract close prices
    close_prices = {}
    for sym, df in stocks.items():
        if 'Close' in df.columns:
            close_prices[sym] = df['Close']

    # Discover cointegrated pairs
    pairs = discover_pairs(close_prices)
    logger.info("Found %d cointegrated pairs", len(pairs))

    if not pairs:
        logger.warning("No pairs found — returning zero returns")
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="statarb")

    # Backtest all pairs
    results = backtest_pair_universe(close_prices, pairs, max_pairs=8)

    if not results:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="statarb")

    # Combine pair return streams (equal weight)
    pair_returns = []
    for label, result in results.items():
        if result.n_trades > 0:
            pair_returns.append(result.daily_returns)
            logger.info(
                "  Pair %s: %d trades, Sharpe=%.3f, Return=%.2f%%",
                label, result.n_trades, result.sharpe_ratio,
                result.total_return * 100,
            )

    if not pair_returns:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="statarb")

    # Average across pairs
    combined = pd.concat(pair_returns, axis=1).mean(axis=1)
    combined.name = "statarb"
    return combined


def run_factor_strategy(
    stocks: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
) -> pd.Series:
    """Run factor model (L/S quintiles) strategy."""
    logger.info("═══ Strategy B: Factor Model (L/S) ═══")

    result = backtest_factor_model(
        price_data=stocks,
        fundamentals=fundamentals,
        rebalance_days=21,
        lookback_momentum=252,
        quintile_pct=0.20,
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
    Run a regime-conditioned cross-sectional momentum strategy.

    Key insight from Citadel/Two Sigma research: pure L/S momentum
    in a strong bull market loses money because you're shorting winners
    that keep winning. Solution: regime-condition the short leg.

    BULL regime:   Long-only top quintile (ride the trend)
    SIDEWAYS/BEAR: Full L/S (shorts work when momentum reverses)

    Uses 6-1 momentum (126 trading days, skip 21) — better for
    Indian markets which move faster than US large-caps.
    """
    logger.info("═══ Strategy C: Cross-Sectional Momentum ═══")

    # Build close price panel
    close_dfs = {s: df['Close'] for s, df in stocks.items() if 'Close' in df.columns}
    panel = pd.DataFrame(close_dfs).dropna(how='all')

    if len(panel) < 170:
        dates = next(iter(stocks.values())).index
        return pd.Series(0.0, index=dates, name="momentum")

    # Detect regime on NIFTY
    regimes = detect_regime(nifty['Close'])
    regime_aligned = regimes.reindex(panel.index, method='ffill').fillna('UNKNOWN')

    # 6-1 momentum (Jegadeesh & Titman adapted for India)
    mom_6_1 = panel.pct_change(126).shift(21)

    # Volatility filter: exclude top-quartile volatile stocks (quality gate)
    vol_20d = panel.pct_change().rolling(20).std()

    daily_returns = panel.pct_change()
    rebalance_interval = 21

    dates = panel.index
    portfolio_returns = pd.Series(0.0, index=dates)

    for t in range(150, len(dates), rebalance_interval):
        regime = regime_aligned.iloc[t] if t < len(regime_aligned) else 'UNKNOWN'

        # Get momentum scores at rebalance date
        scores = mom_6_1.iloc[t].dropna()
        vols = vol_20d.iloc[t].dropna()

        if len(scores) < 10:
            continue

        # Filter out highest-vol quartile (noisy, mean-reverting)
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

        # Regime conditioning: only short in SIDEWAYS/BEAR
        if regime in ('SIDEWAYS', 'BEAR'):
            short_syms = ranked.index[-n_per_leg:].tolist()
        else:
            short_syms = []  # Long-only in BULL/UNKNOWN

        # Hold period
        hold_end = min(t + rebalance_interval, len(dates))
        for d in range(t + 1, hold_end):
            day_rets = daily_returns.iloc[d]

            long_r = day_rets[long_syms].dropna().mean() if long_syms else 0
            short_r = day_rets[short_syms].dropna().mean() if short_syms else 0

            if short_syms:
                portfolio_returns.iloc[d] = long_r - short_r
            else:
                # Long-only: scale down to 50% of long return (for comparable vol)
                portfolio_returns.iloc[d] = long_r * 0.5

        # Transaction cost on rebalance (15bps per leg)
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


# ── Portfolio Combination ───────────────────────────────────────────────────────

def combine_strategies(
    strategy_returns: dict[str, pd.Series],
    target_vol: float = 0.15,
) -> pd.DataFrame:
    """
    Combine strategy return streams using vol-targeting + Sharpe²-proportional allocation.

    This is the multi-strategy portfolio construction step.
    """
    logger.info("═══ Combining Strategies ═══")

    # Align all series to common dates
    df = pd.DataFrame(strategy_returns)
    df = df.dropna(how='all').fillna(0)

    # Per-strategy metrics
    for col in df.columns:
        arr = df[col].values
        if np.std(arr) > 0:
            sharpe = (np.mean(arr) / np.std(arr)) * np.sqrt(252)
            ann_ret = np.mean(arr) * 252
            ann_vol = np.std(arr) * np.sqrt(252)
        else:
            sharpe = ann_ret = ann_vol = 0
        logger.info(
            "  %s: Sharpe=%.3f, Ann.Ret=%.2f%%, Ann.Vol=%.2f%%",
            col, sharpe, ann_ret * 100, ann_vol * 100,
        )

    # Combined portfolio: vol-targeted equal-weight
    # Scale each strategy to target_vol individually, then average
    scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        rolling_vol = df[col].rolling(60, min_periods=20).std() * np.sqrt(252)
        scale = target_vol / rolling_vol.replace(0, np.nan).ffill().fillna(target_vol)
        scale = scale.clip(upper=3.0)  # Cap leverage at 3x
        scaled[col] = df[col] * scale

    # Sharpe²-proportional weights
    sharpes = {}
    for col in df.columns:
        arr = df[col].values
        s = (np.mean(arr) / max(np.std(arr), 1e-8)) * np.sqrt(252)
        sharpes[col] = max(s, 0.01)  # Floor at 0.01 to avoid zero weights

    sharpe_sq = {k: v ** 2 for k, v in sharpes.items()}
    total_sq = sum(sharpe_sq.values())
    weights = {k: v / total_sq for k, v in sharpe_sq.items()}

    logger.info("  Allocation weights: %s",
                {k: f"{v:.1%}" for k, v in weights.items()})

    # Weighted combination
    combined = sum(scaled[col] * weights[col] for col in scaled.columns)
    combined.name = "portfolio"

    result_df = df.copy()
    result_df["portfolio"] = combined

    return result_df


# ── Reporting ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: np.ndarray) -> dict:
    """Compute comprehensive performance metrics."""
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

    if not strategy_returns:
        logger.error("No strategies produced returns. Check data and modules.")
        return

    logger.info("Active strategies: %s", list(strategy_returns.keys()))

    # 3. Combine
    result_df = combine_strategies(strategy_returns)

    # 4. Report
    report = save_report(result_df)

    # 5. Validation gate (AGENTS.md rule 6)
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

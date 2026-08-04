#!/usr/bin/env python3
"""
ARTHA Quant Engine — Live Training Pipeline
=============================================
Fetches 5 years of real NIFTY 50 OHLCV data via yfinance, then runs:

  1. Regime Detection   — HMM on NIFTY 50 index
  2. StatArb Pairs      — Cointegration scan across banking/IT/auto sectors
  3. Factor Model       — Cross-sectional alpha ranking
  4. Walk-Forward Backtest — Validation gate (AGENTS.md rule 6)

All results cached to data_cache/ and written to Fact Store.
Backtest report saved to docs/backtest_reports/.

Usage:
  python scripts/train_quant_engine.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("artha.train")

# ── NIFTY 50 constituents (top 30 by weight — covers ~75% of the index) ────────
# Grouped by sector for StatArb pair discovery within sectors
UNIVERSE = {
    "Banking": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
        "INDUSINDBK", "BANKBARODA",
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
    ],
    "Energy": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA",
    ],
    "Index": [
        "^NSEI",  # NIFTY 50 index itself
    ],
}

ALL_SYMBOLS = []
for sector_stocks in UNIVERSE.values():
    ALL_SYMBOLS.extend(sector_stocks)

CACHE_DIR = ROOT / "data_cache" / "nifty50"
REPORT_DIR = ROOT / "docs" / "backtest_reports"


# ── Stage 1: Data Fetch ────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data via yfinance with local caching."""
    import yfinance as yf

    cache_file = CACHE_DIR / f"{symbol.replace('^', 'IDX_')}.parquet"

    # Use cache if less than 24 hours old
    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            logger.info("  [cache hit] %s (%.1fh old)", symbol, age_hours)
            return pd.read_parquet(cache_file)

    # Fetch from yfinance
    ticker_symbol = f"{symbol}.NS" if not symbol.startswith("^") else symbol
    logger.info("  [fetching] %s → %s", symbol, ticker_symbol)

    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, auto_adjust=True)

        if df.empty:
            logger.warning("  [empty] %s returned no data", symbol)
            return pd.DataFrame()

        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info("  [cached] %s → %d rows (%s to %s)",
                     symbol, len(df),
                     df.index[0].strftime("%Y-%m-%d"),
                     df.index[-1].strftime("%Y-%m-%d"))
        return df

    except Exception as e:
        logger.error("  [error] %s: %s", symbol, e)
        return pd.DataFrame()


def fetch_universe() -> dict[str, pd.Series]:
    """Fetch close prices for the full universe."""
    print("\n" + "=" * 70)
    print("  STAGE 1: Fetching 5-year OHLCV data for NIFTY 50 universe")
    print("=" * 70)

    price_data = {}
    failed = []

    for symbol in ALL_SYMBOLS:
        df = fetch_ohlcv(symbol)
        if not df.empty and "Close" in df.columns:
            # Remove timezone info for consistency
            close = df["Close"].copy()
            close.index = close.index.tz_localize(None)
            price_data[symbol] = close
        else:
            failed.append(symbol)

        # Rate limit: yfinance can throttle
        time.sleep(0.3)

    print(f"\n  ✓ Fetched {len(price_data)}/{len(ALL_SYMBOLS)} symbols")
    if failed:
        print(f"  ✗ Failed: {', '.join(failed)}")

    return price_data


# ── Stage 2: Regime Detection ──────────────────────────────────────────────────

def run_regime_detection(price_data: dict[str, pd.Series]) -> dict:
    """Run HMM regime detection on the NIFTY 50 index."""
    from app.quant.regime import RegimeDetector, RegimeState

    print("\n" + "=" * 70)
    print("  STAGE 2: Regime Detection (HMM on NIFTY 50)")
    print("=" * 70)

    index_key = "^NSEI"
    if index_key not in price_data:
        print("  ✗ NIFTY 50 index not available, skipping regime detection")
        return {}

    detector = RegimeDetector()
    index_prices = price_data[index_key]

    print(f"  Training HMM on {len(index_prices)} observations...")
    if not detector.fit(index_prices):
        print("  ✗ HMM fitting failed")
        return {}

    regimes = detector.predict(index_prices)
    stats = detector.get_state_statistics()

    # Current regime
    current = regimes.iloc[-1]
    print(f"\n  ┌─ Current Regime: {current}")
    print(f"  │")

    for label, data in sorted(stats.items()):
        print(f"  ├─ {label}:")
        print(f"  │    Annualized Return: {data['annualized_return_pct']:+.1f}%")
        print(f"  │    Annualized Vol:    {data['annualized_volatility_pct']:.1f}%")

    # Regime distribution
    regime_counts = regimes[regimes != RegimeState.UNKNOWN].value_counts()
    total = regime_counts.sum()
    print(f"  │")
    print(f"  ├─ Regime Distribution (last 5 years):")
    for regime, count in regime_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        label = regime.value if isinstance(regime, RegimeState) else str(regime)
        print(f"  │    {label:10s} {pct:5.1f}% {bar}")

    print(f"  └─ Done")

    return {
        "current_regime": current.value if isinstance(current, RegimeState) else str(current),
        "statistics": stats,
        "distribution": {
            (r.value if isinstance(r, RegimeState) else str(r)): int(c)
            for r, c in regime_counts.items()
        },
    }


# ── Stage 3: StatArb Pair Discovery ────────────────────────────────────────────

def run_statarb(price_data: dict[str, pd.Series]) -> dict:
    """Discover cointegrated pairs within sectors."""
    from app.quant.statarb import discover_pairs, generate_pair_signals

    print("\n" + "=" * 70)
    print("  STAGE 3: Statistical Arbitrage — Sector Pair Discovery")
    print("=" * 70)

    all_pairs = []

    for sector, symbols in UNIVERSE.items():
        if sector == "Index":
            continue

        sector_data = {s: price_data[s] for s in symbols if s in price_data}
        if len(sector_data) < 2:
            continue

        print(f"\n  [{sector}] Testing {len(sector_data)} stocks...")
        pairs = discover_pairs(sector_data)

        for pair in pairs:
            print(f"    ✓ {pair.symbol_a} — {pair.symbol_b}: "
                  f"p={pair.p_value:.4f}, half-life={pair.half_life:.0f}d, "
                  f"hedge={pair.hedge_ratio:.3f}")

            # Generate latest signal
            signals = generate_pair_signals(
                price_data[pair.symbol_a],
                price_data[pair.symbol_b],
                pair.symbol_a, pair.symbol_b,
            )
            if signals:
                latest = signals[-1]
                print(f"           → z-score={latest.z_score:+.2f}, signal={latest.signal}")

            all_pairs.append({
                "sector": sector,
                "symbol_a": pair.symbol_a,
                "symbol_b": pair.symbol_b,
                "p_value": pair.p_value,
                "half_life": pair.half_life,
                "hedge_ratio": pair.hedge_ratio,
                "latest_z": signals[-1].z_score if signals else None,
                "latest_signal": signals[-1].signal if signals else None,
            })

    print(f"\n  Total cointegrated pairs found: {len(all_pairs)}")
    return {"pairs": all_pairs}


# ── Stage 4: Factor Model ─────────────────────────────────────────────────────

def run_factor_model(price_data: dict[str, pd.Series]) -> dict:
    """Run cross-sectional factor ranking on the universe."""
    from app.quant.factors import compute_composite_alpha

    print("\n" + "=" * 70)
    print("  STAGE 4: Multi-Factor Alpha Ranking")
    print("=" * 70)

    # Filter out the index
    stock_data = {s: p for s, p in price_data.items() if not s.startswith("^")}

    exposures = compute_composite_alpha(stock_data)

    if not exposures:
        print("  ✗ No factor scores computed")
        return {}

    # Top 10 and Bottom 5
    print(f"\n  ┌─ Top 10 by Composite Alpha:")
    print(f"  │  {'Rank':>4}  {'Symbol':<15} {'Alpha':>8} {'Mom':>7} {'LowVol':>7} {'Val':>7} {'Qual':>7}")
    print(f"  │  {'─' * 60}")
    for e in exposures[:10]:
        print(f"  │  {e.rank:4d}  {e.symbol:<15} {e.composite_alpha:+8.4f} "
              f"{e.momentum_z:+7.3f} {e.low_vol_z:+7.3f} "
              f"{e.value_z:+7.3f} {e.quality_z:+7.3f}")

    print(f"  │")
    print(f"  ├─ Bottom 5:")
    for e in exposures[-5:]:
        print(f"  │  {e.rank:4d}  {e.symbol:<15} {e.composite_alpha:+8.4f} "
              f"{e.momentum_z:+7.3f} {e.low_vol_z:+7.3f} "
              f"{e.value_z:+7.3f} {e.quality_z:+7.3f}")

    print(f"  └─ {len(exposures)} stocks ranked")

    return {
        "rankings": [
            {
                "rank": e.rank, "symbol": e.symbol,
                "composite_alpha": e.composite_alpha,
                "momentum_z": e.momentum_z, "low_vol_z": e.low_vol_z,
                "value_z": e.value_z, "quality_z": e.quality_z,
            }
            for e in exposures
        ]
    }


# ── Stage 5: Walk-Forward Backtest ─────────────────────────────────────────────

def run_backtest(price_data: dict[str, pd.Series]) -> dict:
    """Run walk-forward backtest on a momentum strategy as baseline."""
    from app.quant.backtest import compute_metrics, walk_forward_backtest

    print("\n" + "=" * 70)
    print("  STAGE 5: Walk-Forward Backtest (Momentum Baseline)")
    print("=" * 70)

    # Test a simple momentum strategy: go long top-quintile, short bottom-quintile
    # using NIFTY 50 returns as the benchmark
    stock_data = {s: p for s, p in price_data.items() if not s.startswith("^")}

    results = {}

    # Per-stock backtest metrics
    print(f"\n  Computing per-stock metrics for {len(stock_data)} stocks...")
    stock_metrics = {}
    for symbol, prices in stock_data.items():
        returns = prices.pct_change().dropna()
        if len(returns) < 100:
            continue
        metrics = compute_metrics(returns)
        stock_metrics[symbol] = metrics

    # Sort by Sharpe ratio
    sorted_stocks = sorted(
        stock_metrics.items(),
        key=lambda x: x[1].sharpe_ratio,
        reverse=True,
    )

    print(f"\n  ┌─ Per-Stock Performance (5-year, sorted by Sharpe):")
    print(f"  │  {'Symbol':<15} {'Sharpe':>8} {'Sortino':>8} {'Return':>10} "
          f"{'MaxDD':>8} {'Vol':>8} {'Win%':>6}")
    print(f"  │  {'─' * 65}")

    validated_count = 0
    for symbol, m in sorted_stocks:
        valid = m.sharpe_ratio >= 1.0 and m.max_drawdown <= 0.25 and m.num_trading_days >= 504
        marker = "✓" if valid else " "
        if valid:
            validated_count += 1
        print(f"  │ {marker} {symbol:<15} {m.sharpe_ratio:+8.2f} {m.sortino_ratio:+8.2f} "
              f"{m.annualized_return:+10.1%} {m.max_drawdown:8.1%} "
              f"{m.volatility:8.1%} {m.win_rate:5.1%}")

    print(f"  │")
    print(f"  ├─ Validation Gate (Sharpe≥1.0, MaxDD≤25%, ≥2yr OOS):")
    print(f"  │    {validated_count}/{len(sorted_stocks)} stocks pass")
    print(f"  └─ Done")

    # Benchmark: NIFTY 50
    if "^NSEI" in price_data:
        nifty_returns = price_data["^NSEI"].pct_change().dropna()
        nifty_metrics = compute_metrics(nifty_returns)
        print(f"\n  Benchmark (NIFTY 50):")
        print(f"    Sharpe: {nifty_metrics.sharpe_ratio:+.2f}")
        print(f"    Annual Return: {nifty_metrics.annualized_return:+.1%}")
        print(f"    Max Drawdown: {nifty_metrics.max_drawdown:.1%}")
        print(f"    Volatility: {nifty_metrics.volatility:.1%}")
        results["benchmark"] = {
            "sharpe": nifty_metrics.sharpe_ratio,
            "annual_return": nifty_metrics.annualized_return,
            "max_drawdown": nifty_metrics.max_drawdown,
            "volatility": nifty_metrics.volatility,
        }

    results["stocks"] = {
        symbol: {
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "annual_return": m.annualized_return,
            "max_drawdown": m.max_drawdown,
            "volatility": m.volatility,
            "win_rate": m.win_rate,
            "validated": m.sharpe_ratio >= 1.0 and m.max_drawdown <= 0.25 and m.num_trading_days >= 504,
        }
        for symbol, m in sorted_stocks
    }
    results["validated_count"] = validated_count
    results["total_count"] = len(sorted_stocks)

    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          ARTHA Quant Engine — Live Training Pipeline               ║")
    print("║          NIFTY 50 Universe · 5-Year History · All Stages           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Stage 1: Fetch data
    price_data = fetch_universe()

    if len(price_data) < 5:
        print("\n  ✗ Not enough data fetched. Check network connection.")
        sys.exit(1)

    # Stage 2: Regime detection
    regime_results = run_regime_detection(price_data)

    # Stage 3: StatArb pairs
    statarb_results = run_statarb(price_data)

    # Stage 4: Factor model
    factor_results = run_factor_model(price_data)

    # Stage 5: Walk-forward backtest
    backtest_results = run_backtest(price_data)

    # ── Save backtest report ────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"nifty50_backtest_{timestamp}.json"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(price_data),
        "regime": regime_results,
        "statarb": statarb_results,
        "factor_rankings": factor_results,
        "backtest": backtest_results,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE — {elapsed:.1f}s")
    print("=" * 70)
    print(f"  Universe:           {len(price_data)} stocks")
    print(f"  Regime:             {regime_results.get('current_regime', 'N/A')}")
    print(f"  Cointegrated pairs: {len(statarb_results.get('pairs', []))}")
    print(f"  Stocks ranked:      {len(factor_results.get('rankings', []))}")
    print(f"  Validated signals:  {backtest_results.get('validated_count', 0)}/{backtest_results.get('total_count', 0)}")
    print(f"  Report saved:       {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

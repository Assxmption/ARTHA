"""
ARTHA Quant Engine v5 — Medallion-Class Training Pipeline
==========================================================
Full-stack quantitative backtester:
  1. Fetch 5-year daily OHLCV + fundamentals for 45+ stocks
  2. Compute 40+ signals (8 families) via signals.py
  3. Train ML alpha ensemble (XGBoost + Ridge + RF) via ml_alpha.py
  4. Daily portfolio optimization via portfolio_optimizer.py
  5. NIFTY futures beta-neutral hedge via futures_hedge.py
  6. 6-layer risk management via risk_manager.py
  7. Full backtest with realistic transaction costs

Target: Hedged Sharpe ≥ 1.3, MaxDD ≤ 15%
"""

from __future__ import annotations

import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.regime import detect_regime
from app.quant.signals import build_signal_matrix, ALL_SIGNAL_NAMES
from app.quant.ml_alpha import train_alpha_model
from app.quant.portfolio_optimizer import optimize_portfolio
from app.quant.futures_hedge import apply_futures_hedge, compute_hedge_metrics
from app.quant.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("artha.v5")

# ── Configuration ─────────────────────────────────────────────────────────────

REPORT_DIR = Path("docs/backtest_reports")
CACHE_DIR = Path("data_cache")

# Transaction costs
TC_BROKERAGE_BPS = 5     # 0.05% per trade
TC_IMPACT_BPS = 10       # 0.10% market impact
TC_TOTAL_BPS = TC_BROKERAGE_BPS + TC_IMPACT_BPS  # 15bps per trade

# Universe
NIFTY_SYMBOLS = [
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHREECEM", "SUNPHARMA", "TATACOMM", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

INDEX_SYMBOL = "^NSEI"

SECTOR_MAP = {
    "ADANIENT": "Conglomerate", "ADANIGREEN": "Energy", "ADANIPORTS": "Infrastructure",
    "APOLLOHOSP": "Healthcare", "ASIANPAINT": "Consumer",
    "AXISBANK": "Banking", "BAJAJ-AUTO": "Auto", "BAJFINANCE": "NBFC",
    "BAJAJFINSV": "NBFC", "BHARTIARTL": "Telecom",
    "BPCL": "Energy", "BRITANNIA": "FMCG", "CIPLA": "Pharma",
    "COALINDIA": "Mining", "DIVISLAB": "Pharma",
    "DRREDDY": "Pharma", "EICHERMOT": "Auto", "GRASIM": "Materials",
    "HCLTECH": "IT", "HDFCBANK": "Banking",
    "HDFCLIFE": "Insurance", "HEROMOTOCO": "Auto",
    "HINDALCO": "Metals", "HINDUNILVR": "FMCG",
    "ICICIBANK": "Banking", "INDUSINDBK": "Banking",
    "INFY": "IT", "ITC": "FMCG", "JSWSTEEL": "Metals",
    "KOTAKBANK": "Banking", "LT": "Infrastructure",
    "M&M": "Auto", "MARUTI": "Auto", "NESTLEIND": "FMCG",
    "NTPC": "Energy", "ONGC": "Energy", "POWERGRID": "Energy",
    "RELIANCE": "Conglomerate", "SBILIFE": "Insurance", "SBIN": "Banking",
    "SHREECEM": "Materials", "SUNPHARMA": "Pharma",
    "TATACOMM": "IT", "TATASTEEL": "Metals",
    "TCS": "IT", "TECHM": "IT", "TITAN": "Consumer",
    "ULTRACEMCO": "Materials", "WIPRO": "IT",
}


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_stock_data(symbols: list[str], years: int = 5) -> tuple[dict, dict, dict]:
    """Fetch OHLCV data, compute volumes, and get fundamentals."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    prices = {}
    volumes = {}
    fundamentals = {}
    failed = []

    for sym in symbols + [INDEX_SYMBOL]:
        cache_path = CACHE_DIR / f"{sym.replace('^', 'IDX_')}_v5.parquet"
        yf_sym = sym if sym == INDEX_SYMBOL else f"{sym}.NS"

        try:
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                df = yf.download(yf_sym, start=start, end=end, auto_adjust=True, progress=False)
                if df.empty:
                    failed.append(sym)
                    continue
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)

            if "Close" in df.columns and len(df) > 100:
                prices[sym] = df["Close"].squeeze()
                if "Volume" in df.columns:
                    volumes[sym] = df["Volume"].squeeze()
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", sym, e)
            failed.append(sym)

    # Fundamentals (basic snapshot from yfinance)
    for sym in symbols:
        yf_sym = f"{sym}.NS"
        fund_cache = CACHE_DIR / f"{sym}_fund_v5.json"

        try:
            if fund_cache.exists():
                with open(fund_cache) as f:
                    fundamentals[sym] = json.load(f)
            else:
                ticker = yf.Ticker(yf_sym)
                info = ticker.info or {}
                fund = {
                    "eps": info.get("trailingEps"),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                    "pb": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                    "de": info.get("debtToEquity"),
                    "market_cap": info.get("marketCap"),
                    "sector": SECTOR_MAP.get(sym, "Other"),
                }
                fundamentals[sym] = fund
                with open(fund_cache, "w") as f:
                    json.dump(fund, f)
        except Exception as e:
            logger.debug("Fundamentals failed for %s: %s", sym, e)
            fundamentals[sym] = {"sector": SECTOR_MAP.get(sym, "Other")}

    logger.info("Data fetch: %d stocks + index, %d failed: %s",
                len(prices) - 1, len(failed), failed or "none")

    return prices, volumes, fundamentals


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_v5_backtest(
    prices: dict[str, pd.Series],
    volumes: dict[str, pd.Series],
    fundamentals: dict[str, dict],
) -> dict:
    """Run the full v5 pipeline."""

    print("\n" + "═" * 70)
    print("  STAGE 1: Data Preparation")
    print("═" * 70)

    # Separate index
    index_prices = prices.pop(INDEX_SYMBOL, None)
    stock_symbols = sorted([s for s in prices.keys() if s != INDEX_SYMBOL])

    if index_prices is None:
        print("  ✗ No NIFTY index data")
        return {}

    # Compute daily returns
    stock_returns = {sym: prices[sym].pct_change().dropna() for sym in stock_symbols}
    index_returns = index_prices.pct_change().dropna()

    # Find common date range
    all_dates = None
    for sym in stock_symbols:
        idx = stock_returns[sym].index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)

    all_dates = all_dates.intersection(index_returns.index).sort_values()
    print(f"  ✓ {len(stock_symbols)} stocks, {len(all_dates)} common trading days")

    # Regime detection
    regimes = detect_regime(index_prices)
    current_regime = "BULL"
    if not regimes.empty:
        last = regimes.iloc[-1]
        current_regime = last.value if hasattr(last, 'value') else str(last)
    print(f"  ✓ Regime: {current_regime}")

    # Benchmark metrics
    idx_ann_ret = index_returns.mean() * 252
    idx_ann_vol = index_returns.std() * np.sqrt(252)
    idx_sharpe = idx_ann_ret / idx_ann_vol if idx_ann_vol > 0 else 0
    idx_cum = (1 + index_returns).cumprod()
    idx_dd = (idx_cum / idx_cum.cummax() - 1).min()
    print(f"  ✓ Benchmark: Sharpe={idx_sharpe:+.2f}, Ret={idx_ann_ret*100:.1f}%, DD={abs(idx_dd)*100:.1f}%")

    # ── Stage 2: Signal Generation ────────────────────────────
    print(f"\n{'═' * 70}")
    print("  STAGE 2: Signal Generation (40+ signals)")
    print("═" * 70)

    signal_dates = all_dates[252:]  # Need 252 days lookback
    signal_matrix = build_signal_matrix(
        prices, volumes, fundamentals, SECTOR_MAP,
        signal_dates, min_history=252,
    )

    if signal_matrix.empty:
        print("  ✗ Signal matrix empty")
        return {}

    n_rows, n_signals = signal_matrix.shape
    unique_dates = signal_matrix.index.get_level_values("date").unique()
    unique_stocks = signal_matrix.index.get_level_values("symbol").unique()
    print(f"  ✓ Signal matrix: {n_rows:,} rows × {n_signals} signals")
    print(f"  ✓ Coverage: {len(unique_dates)} dates × {len(unique_stocks)} stocks")

    # Signal quality check: Information Coefficient
    print(f"\n  ┌─ Signal Quality (IC vs next-day returns):")
    fwd_returns_data = []
    for date in unique_dates[:-1]:
        next_date_idx = unique_dates.get_loc(date) + 1
        if next_date_idx >= len(unique_dates):
            continue
        next_date = unique_dates[next_date_idx]

        for sym in unique_stocks:
            if (date, sym) in signal_matrix.index and sym in stock_returns:
                if next_date in stock_returns[sym].index:
                    fwd_ret = stock_returns[sym].loc[next_date]
                    fwd_returns_data.append({"date": date, "symbol": sym, "fwd_return": fwd_ret})

    fwd_returns_df = pd.DataFrame(fwd_returns_data).set_index(["date", "symbol"])

    # Compute IC for each signal
    import warnings
    from scipy.stats import spearmanr
    signal_ics = {}
    merged_for_ic = signal_matrix.join(fwd_returns_df, how="inner")
    for sig in ALL_SIGNAL_NAMES:
        valid = merged_for_ic[[sig, "fwd_return"]].dropna()
        if len(valid) > 100:
            # Skip constant signals (seasonal can be constant within a day)
            if valid[sig].std() < 1e-10:
                signal_ics[sig] = 0.0
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = spearmanr(valid[sig], valid["fwd_return"])
            signal_ics[sig] = ic if not np.isnan(ic) else 0

    # Sort and display top signals
    sorted_ics = sorted(signal_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, ic in sorted_ics[:10]:
        bar = "█" * int(abs(ic) * 500)
        sign = "+" if ic > 0 else "-"
        print(f"  │  {name:20s}  IC={sign}{abs(ic):.4f}  {bar}")
    positive_ic = sum(1 for _, ic in signal_ics.items() if abs(ic) > 0.01)
    print(f"  └─ {positive_ic}/{len(signal_ics)} signals with |IC| > 0.01")

    # ── Stage 3: ML Alpha Training ────────────────────────────
    print(f"\n{'═' * 70}")
    print("  STAGE 3: ML Alpha Model (3-model ensemble)")
    print("═" * 70)

    alpha_result = train_alpha_model(
        signal_matrix, fwd_returns_df,
        min_train_days=504, step_size=21,
    )

    if alpha_result.predictions.empty:
        print("  ✗ ML training failed — insufficient data")
        return {}

    print(f"  ✓ Walk-forward folds: {alpha_result.n_folds}")
    print(f"  ✓ OOS Information Coefficient: {alpha_result.oos_ic:+.4f}")
    print(f"  ✓ OOS R²: {alpha_result.oos_r2:+.4f}")
    print(f"  ✓ Training samples: {alpha_result.n_training_samples:,}")

    # Top features
    top_features = sorted(alpha_result.feature_importance.items(),
                          key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  ┌─ Top ML Features (by importance):")
    for name, imp in top_features:
        bar = "█" * int(imp * 200)
        print(f"  │  {name:20s}  {imp:.4f}  {bar}")
    print(f"  └─")

    # ── Stage 4: Daily Portfolio Construction + Backtest ───────
    print(f"\n{'═' * 70}")
    print("  STAGE 4: Daily Portfolio Construction + Risk Management")
    print("═" * 70)

    predictions = alpha_result.predictions
    pred_dates = predictions.index.get_level_values("date").unique().sort_values()

    risk_mgr = RiskManager()
    prev_weights = None
    portfolio_daily_returns = []
    portfolio_nav = [1.0]
    rebalance_log = []
    trade_count = 0

    for i, date in enumerate(pred_dates):
        # Get alpha scores for this date
        day_mask = predictions.index.get_level_values("date") == date
        if day_mask.sum() == 0:
            continue

        day_preds = predictions.loc[day_mask, "alpha_score"]
        alpha_scores = day_preds.to_dict()
        if isinstance(list(alpha_scores.keys())[0], tuple):
            alpha_scores = {k[1] if isinstance(k, tuple) else k: v for k, v in alpha_scores.items()}

        # Optimize portfolio
        portfolio = optimize_portfolio(
            alpha_scores=alpha_scores,
            returns_history=stock_returns,
            sector_map=SECTOR_MAP,
            prev_weights=prev_weights,
            lookback_days=120,
            date=date,
        )

        weights = portfolio.weights
        if not weights:
            continue

        # Get next trading day's return
        date_loc = pred_dates.get_loc(date)
        if date_loc + 1 >= len(pred_dates):
            break
        next_date = pred_dates[date_loc + 1]

        # Portfolio return for this day
        port_ret = 0.0
        for sym, w in weights.items():
            if sym in stock_returns and next_date in stock_returns[sym].index:
                port_ret += w * stock_returns[sym].loc[next_date]

        # Transaction costs (proportional to turnover)
        tc = portfolio.turnover * TC_TOTAL_BPS / 10000
        port_ret -= tc

        # Count trades
        if prev_weights:
            for sym in set(list(weights.keys()) + list(prev_weights.keys())):
                if abs(weights.get(sym, 0) - prev_weights.get(sym, 0)) > 0.001:
                    trade_count += 1

        # Risk management
        idx_ret = index_returns.loc[next_date] if next_date in index_returns.index else 0
        weights = risk_mgr.update(port_ret, weights.copy(), SECTOR_MAP, idx_ret)

        portfolio_daily_returns.append({
            "date": next_date,
            "portfolio_return": port_ret,
            "index_return": idx_ret,
            "n_positions": portfolio.n_positions,
            "turnover": portfolio.turnover,
            "gross_exposure": portfolio.gross_exposure,
            "risk_scale": risk_mgr.state.gross_exposure_scale,
        })

        prev_weights = weights

        # Log every ~21 days
        if i % 21 == 0 and i > 0:
            nav = np.prod([1 + r["portfolio_return"] for r in portfolio_daily_returns])
            top_3 = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            rebalance_log.append({
                "date": str(date.date()),
                "nav": nav,
                "n_positions": portfolio.n_positions,
                "top_picks": [s for s, _ in top_3],
            })

    if not portfolio_daily_returns:
        print("  ✗ No portfolio returns generated")
        return {}

    print(f"  ✓ Portfolio simulated: {len(portfolio_daily_returns)} days")
    print(f"  ✓ Total trades: {trade_count:,}")
    print(f"  ✓ Risk alerts: {risk_mgr.get_summary()['daily_loss_triggers']} daily loss triggers")

    # ── Stage 5: Futures Hedge ────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  STAGE 5: NIFTY Futures Beta-Neutral Hedge")
    print("═" * 70)

    ret_df = pd.DataFrame(portfolio_daily_returns).set_index("date")
    port_series = ret_df["portfolio_return"]
    idx_series = ret_df["index_return"]

    hedged_returns, hedge_log = apply_futures_hedge(port_series, idx_series)

    if not hedge_log.empty:
        hedge_metrics = compute_hedge_metrics(hedge_log)
        print(f"  ✓ Avg portfolio beta: {hedge_metrics.get('avg_beta', 0):.2f}")
        print(f"  ✓ Hedge effectiveness: {hedge_metrics.get('hedge_effectiveness', 0)*100:.1f}%")
        print(f"  ✓ Total roll cost: {hedge_metrics.get('total_roll_cost_pct', 0):.2f}%")

    # ── Stage 6: Results ──────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  RESULTS: ARTHA Quant Engine v5")
    print("═" * 70)

    def calc_metrics(returns: pd.Series, label: str) -> dict:
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        neg_vol = returns[returns < 0].std() * np.sqrt(252) if (returns < 0).any() else ann_vol
        sortino = ann_ret / neg_vol if neg_vol > 0 else 0
        win_rate = (returns > 0).mean()
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        return {
            "label": label,
            "sharpe": sharpe,
            "sortino": sortino,
            "ann_return": ann_ret,
            "total_return": total_ret,
            "ann_vol": ann_vol,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar": calmar,
            "n_days": len(returns),
        }

    # Unhedged
    unhedged = calc_metrics(port_series, "Unhedged Portfolio")
    # Hedged
    hedged = calc_metrics(hedged_returns, "Beta-Neutral (Hedged)")
    # Benchmark
    bench = calc_metrics(idx_series, "NIFTY 50 Benchmark")

    # Display
    print(f"\n  ┌─ Strategy Performance ({unhedged['n_days']} days):")
    print(f"  │")
    print(f"  │  {'Strategy':<28s} {'Sharpe':>8s} {'Sortino':>8s} {'Return':>8s} {'MaxDD':>8s} {'Vol':>8s} {'WinRate':>8s}")
    print(f"  │  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for m in [unhedged, hedged, bench]:
        status = "✓" if m["sharpe"] >= 1.0 else "✗"
        print(f"  │  {m['label']:<28s} {m['sharpe']:>+7.2f} {m['sortino']:>+7.2f} "
              f"{m['ann_return']*100:>+7.1f}% {m['max_dd']*100:>+7.1f}% "
              f"{m['ann_vol']*100:>7.1f}% {m['win_rate']*100:>6.1f}% {status}")
    print(f"  │")
    print(f"  ├─ Trade Statistics:")
    print(f"  │    Total trades:        {trade_count:,}")
    print(f"  │    Trades/day (avg):     {trade_count / max(unhedged['n_days'], 1):.1f}")
    print(f"  │    Calmar (hedged):      {hedged['calmar']:.2f}")
    print(f"  │    Profit factor:        {hedged['profit_factor']:.2f}")
    print(f"  │")

    # Validation
    hedged_sharpe = hedged["sharpe"]
    hedged_dd = abs(hedged["max_dd"])
    gate_pass = hedged_sharpe >= 1.0 and hedged_dd <= 0.15
    stretch_pass = hedged_sharpe >= 1.3

    print(f"  ├─ Validation Gates:")
    print(f"  │    Hedged Sharpe ≥ 1.0:   {'PASS ✓' if hedged_sharpe >= 1.0 else 'FAIL ✗'} ({hedged_sharpe:+.2f})")
    print(f"  │    MaxDD ≤ 15%:           {'PASS ✓' if hedged_dd <= 0.15 else 'FAIL ✗'} ({hedged_dd*100:.1f}%)")
    print(f"  │    Stretch ≥ 1.3:         {'PASS ✓' if stretch_pass else 'FAIL ✗'} ({hedged_sharpe:+.2f})")
    print(f"  │    OOS IC:                {alpha_result.oos_ic:+.4f} ({'✓' if alpha_result.oos_ic > 0.02 else '✗'})")
    print(f"  │    OOS R²:                {alpha_result.oos_r2:+.4f} ({'✓' if alpha_result.oos_r2 > 0 else '✗'})")
    print(f"  └─")

    # ML model feature importance
    print(f"\n  ┌─ ML Alpha Model:")
    print(f"  │    Walk-forward folds:    {alpha_result.n_folds}")
    print(f"  │    Ensemble:              XGBoost + Ridge + RandomForest")
    print(f"  │    Features:              {alpha_result.n_features}")
    print(f"  │    Training samples:      {alpha_result.n_training_samples:,}")
    print(f"  └─")

    # Risk manager summary
    risk_summary = risk_mgr.get_summary()
    print(f"\n  ┌─ Risk Manager:")
    print(f"  │    Final NAV:             {risk_summary['current_nav']:.4f}")
    print(f"  │    Peak NAV:              {risk_summary['peak_nav']:.4f}")
    print(f"  │    Current DD:            {risk_summary['drawdown']*100:.1f}%")
    print(f"  │    Daily loss triggers:   {risk_summary['daily_loss_triggers']}")
    print(f"  │    5d vol:                {risk_summary['vol_5d']*100:.1f}%")
    print(f"  │    60d vol:               {risk_summary['vol_60d']*100:.1f}%")
    print(f"  └─")

    # ── Save Report ───────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"artha_v5_{ts}.json"

    report = {
        "version": "v5",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe_size": len(stock_symbols),
        "benchmark_sharpe": float(bench["sharpe"]),
        "ml_model": {
            "n_folds": alpha_result.n_folds,
            "oos_ic": float(alpha_result.oos_ic),
            "oos_r2": float(alpha_result.oos_r2),
            "n_features": alpha_result.n_features,
            "n_training_samples": alpha_result.n_training_samples,
            "top_features": [
                {"name": n, "importance": float(i)}
                for n, i in top_features
            ],
        },
        "signal_quality": {
            "n_signals": len(signal_ics),
            "n_significant": positive_ic,
            "top_ics": [
                {"signal": n, "ic": float(ic)}
                for n, ic in sorted_ics[:15]
            ],
        },
        "unhedged": {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in unhedged.items()},
        "hedged": {k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in hedged.items()},
        "benchmark": {k: float(v) if isinstance(v, (float, np.floating)) else v
                      for k, v in bench.items()},
        "risk_summary": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in risk_summary.items()},
        "trade_count": trade_count,
        "hedge_metrics": {k: float(v) for k, v in (hedge_metrics if not hedge_log.empty else {}).items()},
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✓ Report saved: {report_file.name}")

    print(f"\n{'═' * 70}")
    print(f"  TRAINING v5 COMPLETE — {time.time():.0f}")
    print(f"═" * 70)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   ARTHA Quant Engine v5 — Medallion-Class Architecture             ║")
    print("║   40+ Signals · 3-Model ML · Beta-Neutral · Risk-Managed           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Fetch data
    print(f"\n{'═' * 70}")
    print("  DATA ACQUISITION")
    print("═" * 70)

    prices, volumes, fundamentals = fetch_stock_data(NIFTY_SYMBOLS)

    # Run backtest
    report = run_v5_backtest(prices, volumes, fundamentals)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # v4 → v5 comparison
    if report:
        print(f"\n  ┌─ v4 → v5 Improvement:")
        print(f"  │  Architecture:   Factor-tilt → ML Alpha + Futures Hedge")
        print(f"  │  Signals:        7 static → {report.get('ml_model', {}).get('n_features', 40)} ML features")
        print(f"  │  Rebalance:      Weekly → Daily")
        print(f"  │  Risk:           None → 6-layer defense")
        print(f"  │  Hedge:          None → NIFTY futures beta-neutral")
        v4_sharpe = 0.78
        v5_sharpe = report.get("hedged", {}).get("sharpe", 0)
        print(f"  │  VT Sharpe:      +{v4_sharpe:.2f} → {v5_sharpe:+.2f}")
        print(f"  └─")

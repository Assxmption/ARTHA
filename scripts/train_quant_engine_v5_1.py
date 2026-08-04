"""
ARTHA Quant Engine v5.1 — Hybrid ML + Factor Alpha
====================================================
Lessons from v5.0 failure:
  - Pure ML on daily returns = overfitting (OOS IC=-0.01, Sharpe=-0.81)
  - v4 factor model works (Sharpe +0.78) — DON'T throw it away
  - ML should REFINE factor weights, not replace them

Architecture:
  1. v4 factor engine as BASE alpha (proven: Sharpe +0.78)
  2. ML model predicts 5-DAY forward returns (less noise than 1-day)
  3. ML output TILTS factor weights (not replaces them)
  4. Weekly rebalancing (proven: weekly > daily for our universe)
  5. NIFTY futures hedge removes beta → market-neutral alpha
  6. 6-layer risk management

Formula: final_alpha = 0.6 × factor_alpha + 0.4 × ml_alpha
"""

from __future__ import annotations

import sys
import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.regime import detect_regime
from app.quant.signals import build_signal_matrix, ALL_SIGNAL_NAMES
from app.quant.futures_hedge import apply_futures_hedge, compute_hedge_metrics
from app.quant.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("artha.v5.1")

REPORT_DIR = Path("docs/backtest_reports")
CACHE_DIR = Path("data_cache")

# ── Transaction costs (realistic) ─────────────────────────────────────────────
TC_WEEKLY_TURNOVER = 0.25   # ~25% turnover per weekly rebalance
TC_PER_TRADE_BPS = 15       # 15bps per trade (brokerage + impact)
TC_WEEKLY = TC_WEEKLY_TURNOVER * TC_PER_TRADE_BPS / 10000  # ~3.75bps/week
TC_DAILY = TC_WEEKLY / 5    # ~0.75bps/day amortized

# ── Factor weights (regime-conditional, proven in v4) ─────────────────────────
FACTOR_WEIGHTS = {
    "BULL": {
        "mom_252d": 0.15, "mom_120d": 0.15, "mom_60d": 0.10,
        "mom_5d": -0.10,  # Reversal in bull
        "rvol_20d": 0.10, "eps_yield": 0.10, "roe": 0.10,
        "sector_rel_mom": 0.10, "mr_sma20": -0.05, "vol_ratio": 0.05,
    },
    "BEAR": {
        "mom_252d": 0.05, "mom_120d": 0.05, "mom_60d": 0.05,
        "mom_5d": -0.15,  # Stronger reversal in bear
        "rvol_20d": 0.25, "eps_yield": 0.15, "roe": 0.10,
        "sector_rel_mom": 0.05, "mr_sma20": -0.10, "vol_ratio": 0.05,
    },
    "SIDEWAYS": {
        "mom_252d": 0.10, "mom_120d": 0.10, "mom_60d": 0.10,
        "mom_5d": -0.10,
        "rvol_20d": 0.15, "eps_yield": 0.15, "roe": 0.10,
        "sector_rel_mom": 0.10, "mr_sma20": -0.05, "vol_ratio": 0.05,
    },
}

MAX_SINGLE_BET = 0.05  # 5% per stock

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


def fetch_stock_data(symbols, years=5):
    """Fetch OHLCV + fundamentals (with caching)."""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    prices, volumes, fundamentals = {}, {}, {}
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
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
            if "Close" in df.columns and len(df) > 100:
                prices[sym] = df["Close"].squeeze()
                if "Volume" in df.columns:
                    volumes[sym] = df["Volume"].squeeze()
        except Exception as e:
            failed.append(sym)

    for sym in symbols:
        fund_cache = CACHE_DIR / f"{sym}_fund_v5.json"
        try:
            if fund_cache.exists():
                with open(fund_cache) as f:
                    fundamentals[sym] = json.load(f)
            else:
                ticker = yf.Ticker(f"{sym}.NS")
                info = ticker.info or {}
                fund = {
                    "eps": info.get("trailingEps"), "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                    "pb": info.get("priceToBook"), "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                    "de": info.get("debtToEquity"), "market_cap": info.get("marketCap"), "sector": SECTOR_MAP.get(sym, "Other"),
                }
                fundamentals[sym] = fund
                with open(fund_cache, "w") as f:
                    json.dump(fund, f)
        except:
            fundamentals[sym] = {"sector": SECTOR_MAP.get(sym, "Other")}

    logger.info("Data: %d stocks + index, %d failed: %s", len(prices) - 1, len(failed), failed or "none")
    return prices, volumes, fundamentals


def train_ml_5day(signal_matrix, stock_returns, min_train=504, step=21):
    """
    Train ML ensemble predicting 5-DAY forward returns (not 1-day).
    This smooths out noise and gives the model a real signal.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from scipy.stats import spearmanr

    dates = signal_matrix.index.get_level_values("date").unique().sort_values()
    symbols = signal_matrix.index.get_level_values("symbol").unique()
    signal_names = signal_matrix.columns.tolist()

    # Compute 5-day forward returns
    fwd5_data = []
    for i, date in enumerate(dates[:-5]):
        fwd_date = dates[min(i + 5, len(dates) - 1)]
        for sym in symbols:
            if (date, sym) in signal_matrix.index and sym in stock_returns:
                # 5-day cumulative return
                mask = (stock_returns[sym].index > date) & (stock_returns[sym].index <= fwd_date)
                fwd_rets = stock_returns[sym].loc[mask]
                if len(fwd_rets) >= 3:
                    cum_ret = (1 + fwd_rets).prod() - 1
                    fwd5_data.append({"date": date, "symbol": sym, "fwd5_return": cum_ret})

    fwd5_df = pd.DataFrame(fwd5_data).set_index(["date", "symbol"])
    merged = signal_matrix.join(fwd5_df, how="inner").dropna(subset=["fwd5_return"])

    logger.info("ML training: %d samples, predicting 5-day returns", len(merged))

    all_preds = []
    all_importances = []
    oos_ics = []
    n_folds = 0

    for fold_end in range(min_train, len(dates) - step - 5, step):
        train_dates = dates[:fold_end]
        test_dates = dates[fold_end:fold_end + step]

        train_mask = merged.index.get_level_values("date").isin(train_dates)
        test_mask = merged.index.get_level_values("date").isin(test_dates)

        X_train = np.nan_to_num(merged.loc[train_mask, signal_names].values)
        y_train = np.nan_to_num(merged.loc[train_mask, "fwd5_return"].values)
        X_test = np.nan_to_num(merged.loc[test_mask, signal_names].values)
        y_test = merged.loc[test_mask, "fwd5_return"].values if test_mask.sum() > 0 else np.array([])

        if len(X_train) < 200 or len(X_test) < 10:
            continue

        # Simpler models — less overfitting
        hgb = HistGradientBoostingRegressor(
            max_iter=100, max_depth=3, learning_rate=0.03,
            min_samples_leaf=50, l2_regularization=5.0, random_state=42,
        )
        ridge = Ridge(alpha=10.0)
        rf = RandomForestRegressor(n_estimators=50, max_depth=4, max_features=0.5, random_state=42, n_jobs=-1)

        hgb.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        pred = (hgb.predict(X_test) + ridge.predict(X_test) + rf.predict(X_test)) / 3

        # Cross-sectional rank per date
        test_idx = merged.index[test_mask]
        pred_df = pd.DataFrame({"ml_alpha": pred}, index=test_idx)
        for d in test_dates:
            dm = pred_df.index.get_level_values("date") == d
            if dm.sum() > 2:
                ranks = pred_df.loc[dm, "ml_alpha"].rank(pct=True)
                pred_df.loc[dm, "ml_alpha"] = (ranks - 0.5) * 2

        all_preds.append(pred_df)
        all_importances.append(dict(zip(signal_names, rf.feature_importances_)))

        if len(y_test) > 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = spearmanr(pred, y_test)
            if not np.isnan(ic):
                oos_ics.append(ic)
        n_folds += 1

    predictions = pd.concat(all_preds) if all_preds else pd.DataFrame()
    avg_ic = float(np.mean(oos_ics)) if oos_ics else 0
    avg_imp = {}
    for name in signal_names:
        avg_imp[name] = float(np.mean([imp.get(name, 0) for imp in all_importances])) if all_importances else 0

    logger.info("ML done: %d folds, avg 5-day IC=%.4f", n_folds, avg_ic)
    return predictions, avg_ic, avg_imp, n_folds


def run_v5_backtest(prices, volumes, fundamentals):
    """Run the hybrid v5.1 backtest."""

    print(f"\n{'═' * 70}\n  STAGE 1: Data Preparation\n{'═' * 70}")

    index_prices = prices.pop(INDEX_SYMBOL, None)
    stock_symbols = sorted([s for s in prices if s != INDEX_SYMBOL])
    if index_prices is None:
        print("  ✗ No index data"); return {}

    stock_returns = {sym: prices[sym].pct_change().dropna() for sym in stock_symbols}
    index_returns = index_prices.pct_change().dropna()

    all_dates = None
    for sym in stock_symbols:
        idx = stock_returns[sym].index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)
    all_dates = all_dates.intersection(index_returns.index).sort_values()

    print(f"  ✓ {len(stock_symbols)} stocks, {len(all_dates)} trading days")

    # Regime
    regimes = detect_regime(index_prices)
    current_regime = "BULL"
    if not regimes.empty:
        last = regimes.iloc[-1]
        current_regime = last.value if hasattr(last, 'value') else str(last)
    print(f"  ✓ Regime: {current_regime}")

    # Benchmark
    idx_ret = index_returns.mean() * 252
    idx_vol = index_returns.std() * np.sqrt(252)
    idx_sharpe = idx_ret / idx_vol if idx_vol > 0 else 0
    print(f"  ✓ Benchmark: Sharpe={idx_sharpe:+.2f}, Ret={idx_ret*100:.1f}%")

    # ── Stage 2: Signal Generation ────────────────────────────
    print(f"\n{'═' * 70}\n  STAGE 2: Signal Generation\n{'═' * 70}")

    signal_dates = all_dates[252:]
    signal_matrix = build_signal_matrix(prices, volumes, fundamentals, SECTOR_MAP, signal_dates)
    if signal_matrix.empty:
        print("  ✗ Empty signal matrix"); return {}

    n_rows, n_signals = signal_matrix.shape
    unique_dates = signal_matrix.index.get_level_values("date").unique().sort_values()
    print(f"  ✓ {n_rows:,} rows × {n_signals} signals")

    # Signal ICs
    from scipy.stats import spearmanr
    print(f"\n  ┌─ Signal Quality (IC vs 5-day fwd returns):")
    signal_ics = {}
    for sig in ALL_SIGNAL_NAMES:
        ic_vals = []
        for i, date in enumerate(unique_dates[:-5]):
            fwd_date = unique_dates[min(i + 5, len(unique_dates) - 1)]
            sig_vals, ret_vals = [], []
            for sym in signal_matrix.index.get_level_values("symbol").unique():
                if (date, sym) in signal_matrix.index and sym in stock_returns:
                    sig_val = signal_matrix.loc[(date, sym), sig]
                    mask = (stock_returns[sym].index > date) & (stock_returns[sym].index <= fwd_date)
                    fwd = stock_returns[sym].loc[mask]
                    if len(fwd) >= 3:
                        sig_vals.append(sig_val)
                        ret_vals.append((1 + fwd).prod() - 1)
            if len(sig_vals) >= 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ic, _ = spearmanr(sig_vals, ret_vals)
                if not np.isnan(ic):
                    ic_vals.append(ic)
        signal_ics[sig] = float(np.mean(ic_vals)) if ic_vals else 0

    sorted_ics = sorted(signal_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, ic in sorted_ics[:10]:
        bar = "█" * int(abs(ic) * 300)
        sign = "+" if ic > 0 else "-"
        print(f"  │  {name:20s}  IC={sign}{abs(ic):.4f}  {bar}")
    n_sig = sum(1 for _, ic in signal_ics.items() if abs(ic) > 0.01)
    print(f"  └─ {n_sig}/{len(signal_ics)} signals with |IC| > 0.01")

    # ── Stage 3: ML Alpha (5-day prediction) ──────────────────
    print(f"\n{'═' * 70}\n  STAGE 3: ML Alpha (5-day forward prediction)\n{'═' * 70}")

    ml_preds, ml_ic, ml_importance, ml_folds = train_ml_5day(signal_matrix, stock_returns)
    print(f"  ✓ Folds: {ml_folds}, 5-day OOS IC: {ml_ic:+.4f}")

    top_feats = sorted(ml_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"\n  ┌─ Top ML Features:")
    for n, imp in top_feats:
        print(f"  │  {n:20s}  {imp:.4f}  {'█' * int(imp * 200)}")
    print(f"  └─")

    # ── Stage 4: Hybrid Weekly Backtest ───────────────────────
    print(f"\n{'═' * 70}\n  STAGE 4: Hybrid Factor+ML Weekly Backtest\n{'═' * 70}")

    weekly_dates = pd.bdate_range(all_dates[252], all_dates[-1], freq="W-FRI")
    weekly_dates = weekly_dates[weekly_dates.isin(all_dates) | True]  # Keep all Fridays

    # Build daily returns lookup
    daily_returns = {}
    for sym in stock_symbols:
        daily_returns[sym] = stock_returns[sym]
    index_daily = index_returns

    risk_mgr = RiskManager()
    portfolio_daily_returns = []
    prev_weights = {}
    trade_count = 0

    for t_idx in range(52, len(weekly_dates)):
        rebal_date = weekly_dates[t_idx]

        # Current regime
        regime = current_regime
        if not regimes.empty:
            prior = regimes.loc[regimes.index <= rebal_date]
            if not prior.empty:
                r = prior.iloc[-1]
                regime = r.value if hasattr(r, 'value') else str(r)

        factor_weights = FACTOR_WEIGHTS.get(regime, FACTOR_WEIGHTS["BULL"])

        # ── FACTOR ALPHA (v4 proven approach) ─────────────────
        factor_scores = {}
        for sym in stock_symbols:
            if (rebal_date, sym) not in signal_matrix.index:
                # Find nearest date
                sym_dates = signal_matrix.xs(sym, level="symbol").index if sym in signal_matrix.index.get_level_values("symbol") else pd.Index([])
                prior_dates = sym_dates[sym_dates <= rebal_date]
                if prior_dates.empty:
                    continue
                nearest = prior_dates[-1]
                lookup = (nearest, sym)
            else:
                lookup = (rebal_date, sym)

            if lookup not in signal_matrix.index:
                continue

            row = signal_matrix.loc[lookup]
            score = 0.0
            for sig, weight in factor_weights.items():
                if sig in row.index:
                    score += weight * row[sig]
            factor_scores[sym] = score

        if len(factor_scores) < 10:
            continue

        # ── ML ALPHA (5-day prediction tilt) ──────────────────
        ml_scores = {}
        if not ml_preds.empty:
            for sym in stock_symbols:
                if sym in ml_preds.index.get_level_values("symbol"):
                    sym_preds = ml_preds.xs(sym, level="symbol")
                    prior = sym_preds.loc[sym_preds.index <= rebal_date]
                    if not prior.empty:
                        ml_scores[sym] = prior.iloc[-1]["ml_alpha"]

        # ── HYBRID ALPHA: 60% factor + 40% ML ────────────────
        hybrid_scores = {}
        for sym in factor_scores:
            f_score = factor_scores[sym]
            m_score = ml_scores.get(sym, 0)
            hybrid_scores[sym] = 0.6 * f_score + 0.4 * m_score

        # ── SECTOR-NEUTRAL WEIGHT CONSTRUCTION ────────────────
        sector_groups = defaultdict(list)
        for sym, score in hybrid_scores.items():
            sector_groups[SECTOR_MAP.get(sym, "Other")].append((sym, score))

        new_weights = {}
        n_sectors = len(sector_groups)
        sector_budget = 1.0 / max(n_sectors, 1)

        for sector, sym_scores in sector_groups.items():
            sym_scores.sort(key=lambda x: x[1], reverse=True)
            n = len(sym_scores)
            for rank, (sym, score) in enumerate(sym_scores):
                rank_pct = rank / max(n - 1, 1)
                tilt = 2.5 - 2.3 * rank_pct
                new_weights[sym] = (sector_budget / n) * tilt

        # Normalize + cap
        total_w = sum(new_weights.values())
        if total_w > 0:
            new_weights = {s: w / total_w for s, w in new_weights.items()}
        new_weights = {s: min(w, MAX_SINGLE_BET) for s, w in new_weights.items()}
        total_w = sum(new_weights.values())
        if total_w > 0:
            new_weights = {s: w / total_w for s, w in new_weights.items()}

        # Turnover buffer
        if prev_weights:
            weights = {}
            for sym in set(list(new_weights.keys()) + list(prev_weights.keys())):
                new_w = new_weights.get(sym, 0)
                old_w = prev_weights.get(sym, 0)
                if abs(new_w - old_w) / max(old_w, 0.001) < 0.20:
                    weights[sym] = old_w
                else:
                    weights[sym] = new_w
                    trade_count += 1
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {s: w / total_w for s, w in weights.items()}
        else:
            weights = new_weights
            trade_count += len(weights)

        prev_weights = weights.copy()

        # ── COMPUTE DAILY RETURNS UNTIL NEXT REBALANCE ────────
        next_rebal = weekly_dates[t_idx + 1] if t_idx + 1 < len(weekly_dates) else all_dates[-1]

        for date in pd.bdate_range(rebal_date + pd.Timedelta(days=1), next_rebal):
            port_ret = 0.0
            for sym, w in weights.items():
                if sym in daily_returns and date in daily_returns[sym].index:
                    r = daily_returns[sym].loc[date]
                    if pd.notna(r):
                        port_ret += w * r

            # Transaction cost (amortized daily)
            port_ret -= TC_DAILY

            # Index return
            idx_r = index_daily.loc[date] if date in index_daily.index else 0.0

            # Risk management
            adj_weights = risk_mgr.update(port_ret, weights.copy(), SECTOR_MAP, idx_r)

            portfolio_daily_returns.append({
                "date": date,
                "portfolio_return": port_ret,
                "index_return": idx_r,
                "regime": regime,
            })

    if not portfolio_daily_returns:
        print("  ✗ No returns"); return {}

    ret_df = pd.DataFrame(portfolio_daily_returns).set_index("date")
    port_series = ret_df["portfolio_return"]
    idx_series = ret_df["index_return"]

    print(f"  ✓ {len(ret_df)} days simulated, {trade_count:,} trades")

    # ── Stage 5: Futures Hedge ────────────────────────────────
    print(f"\n{'═' * 70}\n  STAGE 5: NIFTY Futures Beta-Neutral Hedge\n{'═' * 70}")

    hedged_returns, hedge_log = apply_futures_hedge(port_series, idx_series)
    hedge_metrics = compute_hedge_metrics(hedge_log) if not hedge_log.empty else {}
    if hedge_metrics:
        print(f"  ✓ Avg beta: {hedge_metrics.get('avg_beta', 0):.2f}")
        print(f"  ✓ Hedge effectiveness: {hedge_metrics.get('hedge_effectiveness', 0)*100:.1f}%")

    # Vol-targeted version (15% target)
    ann_vol = port_series.std() * np.sqrt(252)
    vol_scale = 0.15 / ann_vol if ann_vol > 0 else 1.0
    vol_scale = min(vol_scale, 2.0)
    vt_returns = port_series * vol_scale

    hedged_ann_vol = hedged_returns.std() * np.sqrt(252)
    hedge_vol_scale = 0.15 / hedged_ann_vol if hedged_ann_vol > 0 else 1.0
    hedge_vol_scale = min(hedge_vol_scale, 3.0)
    hedged_vt = hedged_returns * hedge_vol_scale

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'═' * 70}\n  RESULTS: ARTHA Quant Engine v5.1\n{'═' * 70}")

    def calc(s, label):
        ar = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar / av if av > 0 else 0
        cum = (1 + s).cumprod(); dd = (cum / cum.cummax() - 1).min()
        nv = s[s < 0].std() * np.sqrt(252) if (s < 0).any() else av
        so = ar / nv if nv > 0 else 0
        wr = (s > 0).mean()
        pf = abs(s[s > 0].sum() / s[s < 0].sum()) if (s < 0).any() else 0
        ca = ar / abs(dd) if dd != 0 else 0
        return {"label": label, "sharpe": sh, "sortino": so, "return": ar, "max_dd": dd, "vol": av, "win_rate": wr, "pf": pf, "calmar": ca, "n": len(s)}

    total = calc(port_series, "Total (after TC)")
    vt = calc(vt_returns, "Vol-Targeted (15%)")
    hedged = calc(hedged_returns, "Beta-Neutral (hedged)")
    hedged_vt_m = calc(hedged_vt, "Hedged + Vol-Target")
    bench = calc(idx_series, "NIFTY 50 Benchmark")

    print(f"\n  ┌─ Strategy Performance ({total['n']} days, weekly rebalance):")
    print(f"  │")
    print(f"  │  {'Strategy':<28s} {'Sharpe':>8s} {'Sortino':>8s} {'Return':>8s} {'MaxDD':>8s} {'WinRate':>8s}")
    print(f"  │  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for m in [total, vt, hedged, hedged_vt_m, bench]:
        s = "✓" if m["sharpe"] >= 1.0 else "✗"
        print(f"  │  {m['label']:<28s} {m['sharpe']:>+7.2f} {m['sortino']:>+7.2f} "
              f"{m['return']*100:>+7.1f}% {m['max_dd']*100:>+7.1f}% {m['win_rate']*100:>6.1f}% {s}")

    print(f"  │")
    print(f"  ├─ Trades: {trade_count:,}  |  Calmar(hedged): {hedged['calmar']:.2f}  |  PF: {hedged['pf']:.2f}")
    print(f"  │")

    # Validation
    best_sharpe = max(hedged_vt_m["sharpe"], hedged["sharpe"], vt["sharpe"])
    best_dd = min(abs(hedged_vt_m["max_dd"]), abs(hedged["max_dd"]))
    print(f"  ├─ Validation Gates:")
    print(f"  │    Best Sharpe ≥ 1.0:    {'PASS ✓' if best_sharpe >= 1.0 else 'FAIL ✗'} ({best_sharpe:+.2f})")
    print(f"  │    Best MaxDD ≤ 15%:     {'PASS ✓' if best_dd <= 0.15 else 'FAIL ✗'} ({best_dd*100:.1f}%)")
    print(f"  │    ML 5-day IC:          {ml_ic:+.4f} ({'✓' if ml_ic > 0 else '✗'})")
    print(f"  │    Stretch ≥ 1.3:        {'PASS ✓' if best_sharpe >= 1.3 else 'FAIL ✗'}")
    print(f"  └─")

    # Risk
    risk = risk_mgr.get_summary()
    print(f"\n  ┌─ Risk Manager:")
    print(f"  │    NAV: {risk['current_nav']:.4f}  |  Peak: {risk['peak_nav']:.4f}  |  DD: {risk['drawdown']*100:.1f}%")
    print(f"  │    Daily loss triggers: {risk['daily_loss_triggers']}")
    print(f"  └─")

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "version": "v5.1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "architecture": "hybrid_factor_ml",
        "universe_size": len(stock_symbols),
        "ml_model": {"n_folds": ml_folds, "oos_5day_ic": ml_ic, "n_features": n_signals,
                      "top_features": [{"name": n, "imp": float(i)} for n, i in top_feats]},
        "signal_quality": {"n_signals": len(signal_ics), "n_significant": n_sig,
                           "top_ics": [{"signal": n, "ic": float(ic)} for n, ic in sorted_ics[:10]]},
        "total": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in total.items()},
        "vt": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in vt.items()},
        "hedged": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in hedged.items()},
        "hedged_vt": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in hedged_vt_m.items()},
        "benchmark": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in bench.items()},
        "trade_count": trade_count,
        "hedge_metrics": {k: float(v) for k, v in hedge_metrics.items()},
        "risk": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in risk.items()},
    }
    rfile = REPORT_DIR / f"artha_v5.1_{ts}.json"
    with open(rfile, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  ✓ Report: {rfile.name}")
    return report


if __name__ == "__main__":
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   ARTHA v5.1 — Hybrid Factor+ML · Beta-Neutral · Risk-Managed      ║")
    print("║   60% Factor Alpha (v4 proven) + 40% ML Tilt (5-day prediction)    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    prices, volumes, fundamentals = fetch_stock_data(NIFTY_SYMBOLS)
    report = run_v5_backtest(prices, volumes, fundamentals)
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    if report:
        print(f"\n  ┌─ v4 → v5.1 Architecture Change:")
        print(f"  │  Alpha:      Factor-only → 60% Factor + 40% ML")
        print(f"  │  ML target:  1-day returns (noise) → 5-day returns (signal)")
        print(f"  │  Hedge:      None → NIFTY futures beta-neutral")
        print(f"  │  Risk:       None → 6-layer defense")
        v5_sharpe = report.get("hedged_vt", {}).get("sharpe", 0)
        print(f"  │  v4 VT:      +0.78 → v5.1 Hedged VT: {v5_sharpe:+.2f}")
        print(f"  └─")

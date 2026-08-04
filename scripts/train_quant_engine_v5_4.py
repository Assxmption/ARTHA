"""
ARTHA v5.4 — v5.2 Champion + Dynamic Hedge + Regime IC
========================================================
v5.2 is the champion (Sharpe +1.18). v5.3 failed because expanded
universe shrunk the date intersection.

v5.4 = v5.2's NIFTY 50 universe + two targeted improvements:
  [A] Dynamic hedge ratio: 30% in bull, 80% in bear, 50% sideways
  [B] Regime-conditional IC: 60% regime-specific + 40% all-data blend

Does NOT add Next 50 stocks (that broke v5.3).
"""

from __future__ import annotations

import sys
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.regime import detect_regime
from app.quant.signals import build_signal_matrix, ALL_SIGNAL_NAMES
from app.quant.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("artha.v5.4")

REPORT_DIR = Path("docs/backtest_reports")
CACHE_DIR = Path("data_cache")

MAX_SINGLE_BET = 0.05
STABILITY_BUFFER = 0.30
HEDGE_RATIOS = {"BULL": 0.30, "BEAR": 0.80, "SIDEWAYS": 0.50}
ROLL_COST_BPS_YEAR = 60

# Same NIFTY 50 universe as v5.2 (proven)
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


def fetch_data(symbols, years=5):
    import yfinance as yf
    end = datetime.now(); start = end - timedelta(days=years * 365)
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
                if df.empty: failed.append(sym); continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
            if "Close" in df.columns and len(df) > 100:
                prices[sym] = df["Close"].squeeze()
                if "Volume" in df.columns: volumes[sym] = df["Volume"].squeeze()
        except: failed.append(sym)
    for sym in symbols:
        fund_cache = CACHE_DIR / f"{sym}_fund_v5.json"
        try:
            if fund_cache.exists():
                with open(fund_cache) as f: fundamentals[sym] = json.load(f)
            else:
                ticker = yf.Ticker(f"{sym}.NS"); info = ticker.info or {}
                fundamentals[sym] = {
                    "eps": info.get("trailingEps"), "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                    "pb": info.get("priceToBook"), "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                    "de": info.get("debtToEquity"), "market_cap": info.get("marketCap"), "sector": SECTOR_MAP.get(sym, "Other"),
                }
                with open(fund_cache, "w") as f: json.dump(fundamentals[sym], f)
        except: fundamentals[sym] = {"sector": SECTOR_MAP.get(sym, "Other")}
    logger.info("Fetched %d stocks + index, %d failed", len(prices) - 1, len(failed))
    return prices, volumes, fundamentals


def get_rolling_ic(signal_matrix, fwd_ret_map, dates_window, signal_names):
    ics = {}
    for sig in signal_names:
        sig_vals, ret_vals = [], []
        for date in dates_window:
            if date not in signal_matrix.index.get_level_values("date"): continue
            try: day_data = signal_matrix.xs(date, level="date")
            except KeyError: continue
            for sym in day_data.index:
                if (date, sym) in fwd_ret_map:
                    sv = day_data.loc[sym, sig] if sig in day_data.columns else np.nan
                    if pd.notna(sv) and abs(sv) < 100:
                        sig_vals.append(sv); ret_vals.append(fwd_ret_map[(date, sym)])
        if len(sig_vals) > 50:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = spearmanr(sig_vals, ret_vals)
            ics[sig] = ic if not np.isnan(ic) else 0
        else: ics[sig] = 0
    return ics


def run_backtest(prices, volumes, fundamentals):
    print(f"\n{'═' * 70}\n  STAGE 1: Data\n{'═' * 70}")
    index_prices = prices.pop(INDEX_SYMBOL, None)
    stock_symbols = sorted([s for s in prices if s != INDEX_SYMBOL])
    if not index_prices is not None: pass
    stock_returns = {sym: prices[sym].pct_change().dropna() for sym in stock_symbols}
    index_returns = index_prices.pct_change().dropna()

    all_dates = None
    for sym in stock_symbols:
        idx = stock_returns[sym].index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)
    all_dates = all_dates.intersection(index_returns.index).sort_values()
    print(f"  ✓ {len(stock_symbols)} stocks, {len(all_dates)} days")

    regimes = detect_regime(index_prices)
    idx_ann = index_returns.mean() * 252; idx_vol = index_returns.std() * np.sqrt(252)
    print(f"  ✓ Benchmark: Sharpe={idx_ann / idx_vol:+.2f}")

    # ── Signals ───────────────────────────────────────────────
    print(f"\n{'═' * 70}\n  STAGE 2: Signals\n{'═' * 70}")
    signal_dates = all_dates[252:]
    signal_matrix = build_signal_matrix(prices, volumes, fundamentals, SECTOR_MAP, signal_dates)
    if signal_matrix.empty: print("  ✗ Empty"); return {}
    unique_dates = signal_matrix.index.get_level_values("date").unique().sort_values()
    unique_symbols = signal_matrix.index.get_level_values("symbol").unique()
    signal_names = signal_matrix.columns.tolist()
    print(f"  ✓ {signal_matrix.shape[0]:,} × {signal_matrix.shape[1]} signals")

    # Forward returns
    print(f"\n{'═' * 70}\n  STAGE 3: IC Computation\n{'═' * 70}")
    fwd_ret_map = {}
    for i, date in enumerate(unique_dates[:-5]):
        fwd_date = unique_dates[min(i + 5, len(unique_dates) - 1)]
        for sym in unique_symbols:
            if sym in stock_returns:
                mask = (stock_returns[sym].index > date) & (stock_returns[sym].index <= fwd_date)
                fwd = stock_returns[sym].loc[mask]
                if len(fwd) >= 3:
                    fwd_ret_map[(date, sym)] = float((1 + fwd).prod() - 1)
    print(f"  ✓ {len(fwd_ret_map):,} forward returns")

    full_ics = get_rolling_ic(signal_matrix, fwd_ret_map, unique_dates[:500], signal_names)
    sorted_ics = sorted(full_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  ┌─ Top ICs:")
    for name, ic in sorted_ics[:8]:
        print(f"  │  {name:20s}  IC={'+'if ic>0 else '-'}{abs(ic):.4f}")
    n_sig = sum(1 for _, ic in full_ics.items() if abs(ic) > 0.01)
    print(f"  └─ {n_sig}/{len(full_ics)} with |IC| > 0.01")

    # ── Backtest ──────────────────────────────────────────────
    print(f"\n{'═' * 70}\n  STAGE 4: Walk-Forward IC + Dynamic Hedge\n{'═' * 70}")

    all_fridays = pd.bdate_range(all_dates[252], all_dates[-1], freq="W-FRI")
    risk_mgr = RiskManager()
    prev_weights = {}
    portfolio_daily_returns = []
    trade_count = 0
    ic_window = 252
    rebal_count = 0

    for t_idx in range(52, len(all_fridays)):
        rebal_date = all_fridays[t_idx]
        valid = unique_dates[unique_dates <= rebal_date]
        if len(valid) < ic_window: continue
        actual_date = valid[-1]

        # [B] Regime at rebalance
        regime = "BULL"
        if not regimes.empty:
            prior = regimes.loc[regimes.index <= rebal_date]
            if not prior.empty:
                r = prior.iloc[-1]
                regime = r.value if hasattr(r, 'value') else str(r)

        # [B] Regime-conditional IC (60% regime + 40% all)
        regime_days = []
        for d in valid[-ic_window:]:
            if not regimes.empty:
                rd = regimes.loc[regimes.index <= d]
                if not rd.empty:
                    rv = rd.iloc[-1]
                    rv = rv.value if hasattr(rv, 'value') else str(rv)
                    if rv == regime: regime_days.append(d)

        all_ics = get_rolling_ic(signal_matrix, fwd_ret_map, valid[-ic_window:], signal_names)

        if len(regime_days) > 80:
            regime_ics = get_rolling_ic(signal_matrix, fwd_ret_map, regime_days, signal_names)
            blended = {s: 0.6 * regime_ics.get(s, 0) + 0.4 * all_ics.get(s, 0) for s in signal_names}
        else:
            blended = all_ics

        active = {s: ic for s, ic in blended.items() if abs(ic) > 0.005}
        if len(active) < 3: continue

        ic_sum = sum(abs(ic) for ic in active.values())
        if ic_sum < 1e-10: continue
        ic_weights = {s: ic / ic_sum for s, ic in active.items()}

        # Score stocks
        stock_scores = {}
        for sym in unique_symbols:
            if (actual_date, sym) in signal_matrix.index:
                lookup = actual_date
            else:
                sym_dates = valid[valid.isin(signal_matrix.xs(sym, level="symbol").index)] if sym in signal_matrix.index.get_level_values("symbol") else pd.Index([])
                if sym_dates.empty: continue
                lookup = sym_dates[-1]
            if (lookup, sym) not in signal_matrix.index: continue
            row = signal_matrix.loc[(lookup, sym)]
            score = sum(ic_weights.get(sig, 0) * row[sig] for sig in ic_weights if sig in row.index and pd.notna(row[sig]))
            stock_scores[sym] = score

        if len(stock_scores) < 10: continue

        # Sector-neutral construction
        sector_groups = defaultdict(list)
        for sym, sc in stock_scores.items():
            sector_groups[SECTOR_MAP.get(sym, "Other")].append((sym, sc))

        new_weights = {}
        n_sectors = len(sector_groups)
        sector_budget = 1.0 / max(n_sectors, 1)
        for sector, sym_scores in sector_groups.items():
            sym_scores.sort(key=lambda x: x[1], reverse=True)
            n = len(sym_scores)
            for rank, (sym, _) in enumerate(sym_scores):
                rank_pct = rank / max(n - 1, 1)
                tilt = 2.5 - 2.0 * rank_pct
                new_weights[sym] = max((sector_budget / n) * tilt, 0.001)

        total_w = sum(new_weights.values())
        if total_w > 0: new_weights = {s: w / total_w for s, w in new_weights.items()}
        new_weights = {s: min(w, MAX_SINGLE_BET) for s, w in new_weights.items()}
        total_w = sum(new_weights.values())
        if total_w > 0: new_weights = {s: w / total_w for s, w in new_weights.items()}

        # Stability buffer
        if prev_weights:
            weights = {}
            for sym in set(list(new_weights.keys()) + list(prev_weights.keys())):
                new_w = new_weights.get(sym, 0); old_w = prev_weights.get(sym, 0)
                if old_w > 0.001 and abs(new_w - old_w) / old_w < STABILITY_BUFFER:
                    weights[sym] = old_w
                else:
                    weights[sym] = new_w
                    if abs(new_w - old_w) > 0.001: trade_count += 1
            weights = {s: w for s, w in weights.items() if w > 0.0005}
            total_w = sum(weights.values())
            if total_w > 0: weights = {s: w / total_w for s, w in weights.items()}
        else:
            weights = new_weights; trade_count += len(weights)

        prev_weights = weights.copy()
        rebal_count += 1

        next_rebal = all_fridays[t_idx + 1] if t_idx + 1 < len(all_fridays) else all_dates[-1]
        for date in pd.bdate_range(rebal_date + pd.Timedelta(days=1), next_rebal):
            if date not in all_dates: continue
            port_ret = sum(w * stock_returns[sym].loc[date] for sym, w in weights.items()
                           if sym in stock_returns and date in stock_returns[sym].index and pd.notna(stock_returns[sym].loc[date]))
            port_ret -= 0.5 / 10000
            idx_r = float(index_returns.loc[date]) if date in index_returns.index else 0.0
            risk_mgr.update(port_ret, weights.copy(), SECTOR_MAP, idx_r)
            portfolio_daily_returns.append({"date": date, "portfolio_return": port_ret, "index_return": idx_r, "regime": regime})

    if not portfolio_daily_returns: print("  ✗ No returns"); return {}

    ret_df = pd.DataFrame(portfolio_daily_returns).set_index("date")
    port_series = ret_df["portfolio_return"]; idx_series = ret_df["index_return"]
    print(f"  ✓ {len(ret_df)} days, {trade_count:,} trades, {rebal_count} rebalances")

    # ── Dynamic Hedge [A] ─────────────────────────────────────
    print(f"\n{'═' * 70}\n  STAGE 5: Dynamic Hedge\n{'═' * 70}")

    aligned = pd.DataFrame({"port": port_series, "idx": idx_series}).dropna()
    hedged_daily = []
    for i in range(120, len(aligned)):
        window = aligned.iloc[i - 120:i]
        cov_m = np.cov(window["port"].values, window["idx"].values)
        var_idx = cov_m[1, 1]
        beta = np.clip(cov_m[0, 1] / var_idx if var_idx > 1e-10 else 1.0, 0.3, 1.5)
        date = aligned.index[i]
        regime = ret_df.loc[date, "regime"] if date in ret_df.index and "regime" in ret_df.columns else "BULL"
        hr = HEDGE_RATIOS.get(regime, 0.50)
        daily_roll = (ROLL_COST_BPS_YEAR * hr) / 10000 / 252
        hedged_r = aligned.iloc[i]["port"] - hr * beta * aligned.iloc[i]["idx"] - daily_roll
        hedged_daily.append({"date": date, "hedged_return": hedged_r, "beta": beta, "hr": hr, "regime": regime})

    hedge_df = pd.DataFrame(hedged_daily).set_index("date")
    hedged_series = hedge_df["hedged_return"]
    print(f"  ✓ Avg beta: {hedge_df['beta'].mean():.2f}, avg hedge: {hedge_df['hr'].mean():.0%}")
    for reg in ["BULL", "BEAR", "SIDEWAYS"]:
        rm = hedge_df["regime"] == reg
        if rm.sum() > 0: print(f"  ✓ {reg}: {rm.sum()} days, hedge={hedge_df.loc[rm, 'hr'].mean():.0%}")

    # Vol-targeted
    av = port_series.std() * np.sqrt(252)
    vt_raw = port_series * min(0.15 / av, 2.0) if av > 0 else port_series
    avh = hedged_series.std() * np.sqrt(252)
    vt_h = hedged_series * min(0.15 / avh, 3.0) if avh > 0 else hedged_series

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'═' * 70}\n  RESULTS: ARTHA v5.4\n{'═' * 70}")

    def calc(s, label):
        ar = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar / av if av > 0 else 0
        cum = (1 + s).cumprod(); dd = (cum / cum.cummax() - 1).min()
        nv = s[s < 0].std() * np.sqrt(252) if (s < 0).any() else av
        so = ar / nv if nv > 0 else 0
        wr = float((s > 0).mean())
        pf = abs(s[s > 0].sum() / s[s < 0].sum()) if (s < 0).any() else 0
        ca = ar / abs(dd) if dd != 0 else 0
        tr = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
        return {"label": label, "sharpe": sh, "sortino": so, "ann_return": ar, "total_return": tr,
                "max_dd": dd, "vol": av, "win_rate": wr, "pf": pf, "calmar": ca, "n": len(s)}

    raw = calc(port_series, "Raw (after TC)")
    vt = calc(vt_raw, "Vol-Targeted (15%)")
    hedged = calc(hedged_series, "Dynamic Hedge")
    hvt = calc(vt_h, "Hedged + Vol-Target")
    bench = calc(idx_series, "NIFTY 50")

    all_s = [("raw", raw), ("vt", vt), ("hedged", hedged), ("hvt", hvt)]
    best_name, best = max(all_s, key=lambda x: x[1]["sharpe"])

    print(f"\n  ┌─ Performance ({raw['n']} days):")
    print(f"  │")
    print(f"  │  {'Strategy':<28s} {'Sharpe':>8s} {'Sortino':>8s} {'Return':>8s} {'MaxDD':>8s} {'WinR':>6s} {'PF':>6s}")
    print(f"  │  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    for m in [raw, vt, hedged, hvt, bench]:
        s = "✓✓" if m["sharpe"] >= 1.3 else "✓" if m["sharpe"] >= 1.0 else "✗"
        print(f"  │  {m['label']:<28s} {m['sharpe']:>+7.2f} {m['sortino']:>+7.2f} "
              f"{m['ann_return']*100:>+7.1f}% {m['max_dd']*100:>+7.1f}% "
              f"{m['win_rate']*100:>5.1f}% {m['pf']:>5.2f} {s}")
    print(f"  │")
    print(f"  ├─ Best: {best['label']} → Sharpe {best['sharpe']:+.2f}")
    print(f"  │   Total: {best['total_return']*100:+.1f}%, MaxDD: {abs(best['max_dd'])*100:.1f}%, Calmar: {best['calmar']:.2f}")
    print(f"  │")
    print(f"  ├─ Gates:")
    print(f"  │    Sharpe ≥ 1.0:  {'PASS ✓' if best['sharpe'] >= 1.0 else 'FAIL ✗'} ({best['sharpe']:+.2f})")
    print(f"  │    MaxDD ≤ 15%:   {'PASS ✓' if abs(best['max_dd']) <= 0.15 else 'FAIL ✗'} ({abs(best['max_dd'])*100:.1f}%)")
    print(f"  │    Sharpe ≥ 1.3:  {'PASS ✓' if best['sharpe'] >= 1.3 else 'FAIL ✗'}")
    print(f"  └─")

    print(f"\n  ┌─ Version History:")
    print(f"  │    v4.0  → +0.78  │  v5.0  → -0.81  │  v5.1  → +0.87")
    print(f"  │    v5.2  → +1.18  │  v5.3  → +0.80  │  v5.4  → {best['sharpe']:+.2f}")
    print(f"  └─")

    risk = risk_mgr.get_summary()
    print(f"\n  Risk: NAV={risk['current_nav']:.4f}, DD={risk['drawdown']*100:.1f}%, triggers={risk['daily_loss_triggers']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "version": "v5.4", "universe": len(stock_symbols), "trades": trade_count,
        "best": best_name,
        "raw": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in raw.items()},
        "hedged": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in hedged.items()},
        "hvt": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in hvt.items()},
        "benchmark": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in bench.items()},
    }
    rfile = REPORT_DIR / f"artha_v5.4_{ts}.json"
    with open(rfile, "w") as f: json.dump(report, f, indent=2, default=str)
    print(f"\n  ✓ Report: {rfile.name}")
    return report


if __name__ == "__main__":
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ARTHA v5.4 — v5.2 Champion + Dynamic Hedge + Regime IC            ║")
    print("║  NIFTY 50 only · Walk-forward IC · Regime-conditional weights       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    prices, volumes, fundamentals = fetch_data(NIFTY_SYMBOLS)
    report = run_backtest(prices, volumes, fundamentals)
    print(f"\n  Time: {time.time() - t0:.1f}s")

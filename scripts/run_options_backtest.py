import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.options_backtest import walk_forward_validate, DEFAULT_CAPITAL
from app.quant.options_strategies import StrategyType
from app.quant.regime import detect_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    print("Fetching 15 years of NIFTY data for HMM training...")
    end = datetime.now()
    train_start = end - timedelta(days=15 * 365)
    
    nifty_long_df = yf.download("^NSEI", start=train_start, end=end, auto_adjust=True, progress=False)
    if isinstance(nifty_long_df.columns, pd.MultiIndex):
        nifty_long_df.columns = nifty_long_df.columns.get_level_values(0)
    nifty_long = nifty_long_df["Close"].squeeze().dropna()
    
    print("Fetching 5 years of NIFTY and INDIA VIX for backtest...")
    bt_start = end - timedelta(days=5 * 365)
    
    nifty_df = yf.download("^NSEI", start=bt_start, end=end, auto_adjust=True, progress=False)
    vix_df = yf.download("^INDIAVIX", start=bt_start, end=end, auto_adjust=True, progress=False)
    
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
        
    nifty_close = nifty_df["Close"].squeeze().dropna()
    vix_close = vix_df["Close"].squeeze().dropna()
    
    # Align dates for the backtest period
    common_dates = nifty_close.index.intersection(vix_close.index)
    nifty_close = nifty_close.loc[common_dates]
    vix_close = vix_close.loc[common_dates]
    
    print(f"Loaded {len(common_dates)} days of aligned data for backtest.")
    
    # Detect regimes on the underlying using the 15-year history for training
    print("Running HMM Regime Detection anchored on 15-year history...")
    regimes = detect_regime(prices=nifty_close, training_prices=nifty_long)
    
    print("\nRegime Distribution in Backtest Period:")
    print(regimes.value_counts())
    
    results = {}
    strategies = [
        StrategyType.IRON_CONDOR,
        StrategyType.BULL_PUT_SPREAD,
        StrategyType.BEAR_CALL_SPREAD
    ]
    
    for stype in strategies:
        print(f"\n--- Running Walk-Forward Validation for {stype.value} ---")
        res = walk_forward_validate(
            close_prices=nifty_close,
            strategy_type=stype,
            symbol="NIFTY",
            vix_series=vix_close,
            regime_series=regimes,
            capital=DEFAULT_CAPITAL,
            train_days=252, # 1 year train
            test_days=63,   # 1 quarter test
        )
        
        print(f"Validated: {res.validated}")
        print(f"Reason: {res.reason}")
        print(f"OOS Sharpe: {res.out_of_sample_sharpe}")
        print(f"OOS Return: {res.out_of_sample_return_pct}%")
        print(f"OOS Max DD: {res.out_of_sample_max_dd}%")
        
        results[stype.value] = {
            "validated": res.validated,
            "reason": res.reason,
            "oos_sharpe": res.out_of_sample_sharpe,
            "oos_return_pct": res.out_of_sample_return_pct,
            "oos_max_dd": res.out_of_sample_max_dd,
            "is_sharpe": res.in_sample_sharpe,
            "is_return_pct": res.in_sample_return_pct,
            "is_max_dd": res.in_sample_max_dd,
            "n_windows": res.n_windows
        }
        
    out_path = Path("docs/backtest_reports/options_backtest_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved full report to {out_path}")

if __name__ == "__main__":
    main()

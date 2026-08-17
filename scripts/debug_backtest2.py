import yfinance as yf
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, ".")
from app.quant.regime import detect_regime
from app.quant.options_backtest import precompute_signals, _is_entry_eligible
from app.quant.options_strategies import StrategyType

start = datetime.now() - timedelta(days=5*365)
df = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)
vix_df = yf.download("^INDIAVIX", start=start, auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)

close = df['Close'].squeeze().dropna()
vix = vix_df['Close'].squeeze().dropna()

common_dates = close.index.intersection(vix.index)
close = close.loc[common_dates]
vix = vix.loc[common_dates]

regimes = detect_regime(close)

signals = precompute_signals(close, vix, regimes)

bps_pass = sum(1 for dt, sig in signals.items() if _is_entry_eligible(StrategyType.BULL_PUT_SPREAD, sig))
ic_pass = sum(1 for dt, sig in signals.items() if _is_entry_eligible(StrategyType.IRON_CONDOR, sig))
bcs_pass = sum(1 for dt, sig in signals.items() if _is_entry_eligible(StrategyType.BEAR_CALL_SPREAD, sig))

print("IRON CONDOR allowed trades:", ic_pass)
print("BULL PUT SPREAD allowed trades:", bps_pass)
print("BEAR CALL SPREAD allowed trades:", bcs_pass)
print("VRP stats:", np.mean([s.vrp for s in signals.values()]), np.min([s.vrp for s in signals.values()]), np.max([s.vrp for s in signals.values()]))
print("IV pct stats:", np.mean([s.iv_percentile for s in signals.values()]), np.min([s.iv_percentile for s in signals.values()]), np.max([s.iv_percentile for s in signals.values()]))


import yfinance as yf
from datetime import datetime, timedelta
import sys
import pandas as pd
sys.path.insert(0, ".")
from app.quant.regime import detect_regime
from app.quant.options_backtest import precompute_signals

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
print(regimes.value_counts())

signals = precompute_signals(close, vix, regimes)
regime_counts = {}
for dt, sig in signals.items():
    r = sig.regime
    regime_counts[r] = regime_counts.get(r, 0) + 1
print("Signals regimes:", regime_counts)

import yfinance as yf
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, ".")
from app.quant.regime import detect_regime, RegimeDetector

end = datetime.now()
start = end - timedelta(days=15*365)
df = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df['Close'].squeeze().dropna()

detector3 = RegimeDetector(n_states=3)
detector3.fit(close)
print(detector3.predict(close.tail(1228)).value_counts())

"""
Chart indicators — deterministic, causal, NaN until warmed up.

Computed server-side so every number the chart shows comes from tested
code in app/quant (AGENTS.md rule 1), not ad-hoc frontend maths.

Conventions match the common charting definitions (TradingView/Wilder):
  * SMA/EMA on close; EMA seeded with the SMA of the first `n` values.
  * Bollinger: SMA(n) ± k·σ using the population std (ddof=0), as charting
    platforms do.
  * RSI: Wilder smoothing (α = 1/n) of gains/losses, seeded with simple means.
  * MACD: EMA12 − EMA26, signal EMA9 of MACD, histogram = MACD − signal.
  * ATR: Wilder-smoothed true range.
  * VWAP: cumulative Σ(typical price·volume)/Σvolume, reset every session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return pd.Series(out, index=close.index)
    alpha = 2.0 / (n + 1)
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=close.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * std, "bb_lower": mid - k * std})


def _wilder(values: np.ndarray, n: int, start: int) -> np.ndarray:
    """Wilder smoothing seeded with the simple mean of values[start:start+n]."""
    out = np.full(len(values), np.nan)
    if len(values) < start + n:
        return out
    out[start + n - 1] = values[start : start + n].mean()
    for i in range(start + n, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff().to_numpy(dtype=float)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder(gains, n, start=1)
    avg_loss = _wilder(losses, n, start=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((avg_loss == 0) & ~np.isnan(avg_gain), 100.0, out)
    out = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, out)
    return pd.Series(out, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    valid = line.dropna()
    sig = pd.Series(np.nan, index=close.index)
    if len(valid) >= signal:
        sig.loc[valid.index] = ema(valid, signal).to_numpy()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return pd.Series(_wilder(tr.to_numpy(dtype=float), n, start=0), index=close.index)


def vwap(frame: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP for intraday bars (`date` column = bar timestamp)."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    session = pd.to_datetime(frame["date"]).dt.normalize()
    pv = (typical * frame["volume"]).groupby(session).cumsum()
    vol = frame["volume"].groupby(session).cumsum()
    return (pv / vol.replace(0, np.nan)).astype(float)


AVAILABLE = ("sma20", "sma50", "sma200", "ema9", "ema21", "bb", "vwap", "rsi", "macd", "atr")


def compute(frame: pd.DataFrame, names: list[str], intraday: bool) -> dict[str, list]:
    """Requested indicators as aligned lists (None where not yet defined)."""
    close = frame["close"].astype(float)
    cols: dict[str, pd.Series] = {}
    for name in names:
        if name.startswith("sma") and name[3:].isdigit():
            cols[name] = sma(close, int(name[3:]))
        elif name.startswith("ema") and name[3:].isdigit():
            cols[name] = ema(close, int(name[3:]))
        elif name == "bb":
            cols.update(bollinger(close).to_dict("series"))
        elif name == "vwap" and intraday:
            cols["vwap"] = vwap(frame)
        elif name == "rsi":
            cols["rsi"] = rsi(close)
        elif name == "macd":
            cols.update(macd(close).to_dict("series"))
        elif name == "atr":
            cols["atr"] = atr(frame["high"].astype(float), frame["low"].astype(float), close)
    return {
        k: [None if (v is None or not np.isfinite(v)) else round(float(v), 4) for v in s.to_numpy()]
        for k, s in cols.items()
    }

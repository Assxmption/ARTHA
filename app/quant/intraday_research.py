"""
Intraday research harness — 5-minute bars, walk-forward OOS gate, real costs.

Every strategy is a *causal* function of a (days × slots × symbols) price cube
that returns target weights W[d, k, s], decided at the close of slot k from data
at or before that close.  The same functions drive the live paper engine
(app/quant/intraday_paper.py), so backtest and live cannot drift apart.

Execution model (deliberately conservative):
  * a weight decided at the close of slot k is filled at the OPEN of slot k+1;
  * every position is flat by ``EXIT_SLOT`` (15:20 open — the intraday square-off),
    so nothing is held overnight;
  * costs are charged on every unit of turnover, side-aware (see ``CostModel``).

Validation (AGENTS.md rule 6): the last ``MIN_OOS_TRADING_DAYS`` (504) sessions
are out-of-sample; a strategy is ``validated`` only if its OOS daily returns
clear the same gate as every other ARTHA signal (Sharpe ≥ 1, max DD ≤ 25 %).
Parameters below are predeclared and are not tuned on OOS data.  The number of
hypotheses tested is reported with the results.

Not investment advice.  Results are historical simulations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SLOTS = 75                                   # 09:15 … 15:25 five-minute bars
SLOT_TIMES = [(datetime(2000, 1, 1, 9, 15) + timedelta(minutes=5 * i)).time() for i in range(SLOTS)]
EXIT_SLOT = SLOT_TIMES.index(time(15, 20))   # positions closed at this bar's open
MIN_BARS_PER_DAY = 60                        # shorter (special) sessions are excluded
IMPLAUSIBLE_MOVE = 0.40                      # backstop for unlisted corporate actions (e.g. demergers)
GENUINE_SHOCK = 0.25                         # reported (never masked): real large moves stay in


# ── Costs ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostModel:
    """
    Per-side cost as a fraction of traded notional, NSE intraday equity (MIS).

    Defaults follow the dated equity-intraday schedule in ``app/quant/costs.py``
    (Zerodha, checked 2026-11-27): STT 0.025 % on the sell side, NSE
    transaction charge 0.00307 %, SEBI fee 0.0001 %, GST 18 % on (exchange +
    SEBI) charges, stamp duty 0.003 % on the buy side.  Brokerage is
    min(₹20, 0.03 %) per order — negligible at ₹-lakh order sizes, so it is
    folded into slippage.  ``slippage_bps`` covers spread and impact for
    large-cap names.  ``CostModel.for_segment`` builds other segments.
    """
    slippage_bps: float = 3.0
    stt_sell: float = 0.00025
    exchange: float = 0.0000307
    sebi: float = 0.000001
    gst: float = 0.18
    stamp_buy: float = 0.00003

    @classmethod
    def for_segment(cls, segment: str, slippage_bps: float, on: Optional[date] = None) -> "CostModel":
        from app.quant.costs import schedule_for

        sch = schedule_for(segment, on)
        return cls(slippage_bps=slippage_bps, stt_sell=sch.stt_sell, exchange=sch.exchange, stamp_buy=sch.stamp_buy)

    @property
    def buy(self) -> float:
        return self.slippage_bps / 1e4 + self.stamp_buy + (self.exchange + self.sebi) * (1 + self.gst)

    @property
    def sell(self) -> float:
        return self.slippage_bps / 1e4 + self.stt_sell + (self.exchange + self.sebi) * (1 + self.gst)


# ── Data cube ─────────────────────────────────────────────────────────────────

@dataclass
class Cube:
    days: list[date]
    symbols: list[str]
    open: np.ndarray        # (D, K, S) float64, NaN where missing
    close: np.ndarray       # (D, K, S)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.open.shape


def build_cube(open_wide: pd.DataFrame, close_wide: pd.DataFrame, partial_last_day: bool = False) -> Cube:
    """
    Pivot wide 5-minute frames (timestamp × symbol) into a day × slot × symbol cube.
    ``partial_last_day`` keeps a still-forming final day (live trading); its
    future slots stay NaN, which the causal strategies never read.
    """
    idx = pd.DatetimeIndex(close_wide.index)
    slot_of = {t: i for i, t in enumerate(SLOT_TIMES)}
    slots = np.array([slot_of.get(t, -1) for t in idx.time])
    keep = slots >= 0
    idx, slots = idx[keep], slots[keep]
    dates = idx.normalize()
    counts = pd.Series(1, index=dates).groupby(level=0).sum()
    good_days = sorted(d.date() for d, n in counts.items() if n >= MIN_BARS_PER_DAY)
    if partial_last_day and len(counts):
        last = max(counts.index).date()
        if last not in good_days:
            good_days.append(last)
    day_pos = {d: i for i, d in enumerate(good_days)}
    symbols = list(close_wide.columns)
    D, S = len(good_days), len(symbols)
    o = np.full((D, SLOTS, S), np.nan)
    c = np.full((D, SLOTS, S), np.nan)
    ov, cv = open_wide.values[keep], close_wide.values[keep]
    for row, (d, k) in enumerate(zip(dates, slots)):
        di = day_pos.get(d.date())
        if di is None:
            continue
        o[di, k] = ov[row]
        c[di, k] = cv[row]
    # Within a day, carry the last price over missing bars (no trades ⇒ no move),
    # but never into slots after the last observed bar of a partial (live) day.
    last_slot = int(slots[dates.normalize() == pd.Timestamp(good_days[-1])].max()) if (partial_last_day and D) else SLOTS - 1
    for arr in (o, c):
        for di in range(D):
            pd_frame = pd.DataFrame(arr[di])
            arr[di] = pd_frame.ffill().values
        if partial_last_day and D:
            arr[D - 1, last_slot + 1:] = np.nan
    return Cube(good_days, symbols, o, c)


def corporate_action_mask(cube: Cube, splits: Optional[dict[str, list[str]]] = None) -> np.ndarray:
    """
    (D, S) True on a symbol's corporate-action ex-dates (as-traded prices jump for
    non-economic reasons), from the split/bonus calendar plus a >40 % overnight
    backstop.  Genuine large moves (circuit hits, news shocks) are NOT masked
    (AGENTS.md rule 7) — they are counted in ``shock_days``.
    """
    D, _, S = cube.shape
    mask = np.zeros((D, S), dtype=bool)
    pos = {d: i for i, d in enumerate(cube.days)}
    for j, sym in enumerate(cube.symbols):
        for ds in (splits or {}).get(sym, []):
            i = pos.get(date.fromisoformat(ds))
            if i is not None:
                mask[i, j] = True
    prev = np.vstack([np.full((1, S), np.nan), cube.close[:-1, -1, :]])
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.abs(cube.open[:, 0, :] / prev - 1)
    return mask | (jump > IMPLAUSIBLE_MOVE)


def shock_days(cube: Cube, ca: np.ndarray) -> int:
    prev = np.vstack([np.full((1, cube.shape[2]), np.nan), cube.close[:-1, -1, :]])
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.abs(cube.close[:, -1, :] / prev - 1)
    return int(((move > GENUINE_SHOCK) & ~ca).sum())


# ── Strategies (causal: W[d, k] uses data ≤ close of slot k) ────────────────────

def _rank_long_short(score: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Long the lowest-score fraction, short the highest; dollar neutral, gross 1."""
    w = np.zeros_like(score)
    ok = np.isfinite(score)
    n = int(ok.sum())
    m = max(1, int(n * frac))
    if n < 10:
        return w
    order = np.argsort(np.where(ok, score, np.nan))
    valid = order[:n]
    w[valid[:m]] = 0.5 / m
    w[valid[-m:]] = -0.5 / m
    return w


def xs_reversal_30m(cube: Cube, ca: np.ndarray) -> np.ndarray:
    """Every 30 min after the first hour: long last-30-min losers, short winners (cross-sectional)."""
    D, K, S = cube.shape
    W = np.zeros((D, K, S))
    decisions = [k for k in range(11, EXIT_SLOT - 1) if (k - 11) % 6 == 0]   # 10:10 … 14:40 bar closes
    for d in range(D):
        for i, k in enumerate(decisions):
            r = cube.close[d, k] / cube.close[d, k - 6] - 1
            r[ca[d]] = np.nan
            w = _rank_long_short(r)
            end = decisions[i + 1] if i + 1 < len(decisions) else EXIT_SLOT
            W[d, k:end] = w
    return W


def gap_fade(cube: Cube, ca: np.ndarray) -> np.ndarray:
    """At 09:20: long the biggest gap-downs, short the biggest gap-ups; out by 11:00."""
    D, K, S = cube.shape
    W = np.zeros((D, K, S))
    exit_k = SLOT_TIMES.index(time(11, 0))
    for d in range(1, D):
        gap = cube.close[d, 0] / cube.close[d - 1, -1] - 1
        gap[ca[d] | ca[d - 1]] = np.nan
        W[d, 0:exit_k - 1] = _rank_long_short(gap)
    return W


def intraday_momentum(cube: Cube, ca: np.ndarray) -> np.ndarray:
    """Market intraday momentum (Gao, Han, Li & Zhou 2018): prev close→09:45 return sets the 15:00–15:20 trade."""
    D, K, S = cube.shape
    W = np.zeros((D, K, S))
    sig_k = SLOT_TIMES.index(time(9, 40))       # bar closing at 09:45
    trade_k = SLOT_TIMES.index(time(14, 55))    # decide at 15:00, fill 15:00 open
    for d in range(1, D):
        ok = ~ca[d] & np.isfinite(cube.close[d - 1, -1]) & np.isfinite(cube.close[d, sig_k])
        if ok.sum() < 10:
            continue
        basket = np.nanmean(cube.close[d, sig_k, ok] / cube.close[d - 1, -1, ok] - 1)
        if not np.isfinite(basket) or basket == 0:
            continue
        w = np.zeros(S)
        w[ok] = np.sign(basket) / ok.sum()
        W[d, trade_k:EXIT_SLOT - 1] = w
    return W


@dataclass
class PairSpec:
    a: int
    b: int
    beta: float
    name: str


def select_pairs(cube: Cube, train_days: int, top: int = 10) -> list[PairSpec]:
    """Engle-Granger pairs on the TRAIN window's daily closes only."""
    from app.quant.statarb import discover_pairs

    closes = cube.close[:train_days, -1, :]
    idx = pd.DatetimeIndex(pd.to_datetime(cube.days[:train_days]))
    series = {s: pd.Series(closes[:, i], index=idx).dropna() for i, s in enumerate(cube.symbols)
              if np.isfinite(closes[:, i]).mean() > 0.98}
    pairs = discover_pairs(series)
    out = []
    pos = {s: i for i, s in enumerate(cube.symbols)}
    for p in pairs[:top]:
        a, b = series[p.symbol_a], series[p.symbol_b]
        j = a.index.intersection(b.index)
        la, lb = np.log(a[j].values), np.log(b[j].values)
        beta = float(np.polyfit(lb, la, 1)[0])
        out.append(PairSpec(pos[p.symbol_a], pos[p.symbol_b], beta, f"{p.symbol_a}/{p.symbol_b}"))
    return out


def intraday_pairs(cube: Cube, ca: np.ndarray, pairs: list[PairSpec], lookback: int = 150,
                   z_entry: float = 2.0, z_exit: float = 0.5) -> np.ndarray:
    """Intraday spread mean reversion on train-selected pairs; flat at the square-off."""
    D, K, S = cube.shape
    W = np.zeros((D, K, S))
    if not pairs:
        return W
    per = 1.0 / len(pairs)
    flat = cube.close.reshape(D * K, S)
    for p in pairs:
        spread = np.log(flat[:, p.a]) - p.beta * np.log(flat[:, p.b])
        s = pd.Series(spread)
        mu = s.rolling(lookback, min_periods=lookback).mean().values
        sd = s.rolling(lookback, min_periods=lookback).std().values
        z = ((spread - mu) / sd).reshape(D, K)
        la, lb = 1 / (1 + abs(p.beta)), abs(p.beta) / (1 + abs(p.beta))
        for d in range(D):
            if ca[max(0, d - 2):d + 1, [p.a, p.b]].any():      # split nearby poisons the lookback
                continue
            pos = 0
            for k in range(6, EXIT_SLOT - 1):
                zk = z[d, k]
                if not np.isfinite(zk):
                    continue
                if pos == 0 and abs(zk) > z_entry:
                    pos = -1 if zk > 0 else 1                   # spread rich ⇒ short A, long B
                elif pos != 0 and abs(zk) < z_exit:
                    pos = 0
                if pos:
                    W[d, k, p.a] += pos * la * per
                    W[d, k, p.b] -= pos * np.sign(p.beta) * lb * per
    return W


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(cube: Cube, W: np.ndarray, costs: CostModel = CostModel()) -> pd.Series:
    """Daily returns (fraction of capital) of target weights W, next-open fills, flat by EXIT_SLOT."""
    D, K, S = cube.shape
    P = np.zeros_like(W)
    P[:, 1:, :] = W[:, :-1, :]                     # held from open k to open k+1
    P[:, EXIT_SLOT:, :] = 0.0
    o = cube.open
    r = np.zeros_like(o)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[:, :-1, :] = o[:, 1:, :] / o[:, :-1, :] - 1
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    gross = (P * r).sum(axis=(1, 2))
    dP = np.diff(np.concatenate([np.zeros((D, 1, S)), P], axis=1), axis=1)   # trades at each open
    buy_turn = np.clip(dP, 0, None).sum(axis=(1, 2))
    sell_turn = np.clip(-dP, 0, None).sum(axis=(1, 2))
    net = gross - buy_turn * costs.buy - sell_turn * costs.sell
    return pd.Series(net, index=pd.DatetimeIndex(pd.to_datetime(cube.days)), name="ret")


def trade_stats(cube: Cube, W: np.ndarray) -> dict:
    P = np.zeros_like(W)
    P[:, 1:, :] = W[:, :-1, :]
    P[:, EXIT_SLOT:, :] = 0.0
    dP = np.diff(np.concatenate([np.zeros_like(P[:, :1]), P], axis=1), axis=1)
    entries = ((np.abs(P) > 1e-12) & (np.abs(np.concatenate([np.zeros_like(P[:, :1]), P[:, :-1]], axis=1)) <= 1e-12)).sum()
    return {"orders": int((np.abs(dP) > 1e-12).sum()), "position_entries": int(entries),
            "avg_daily_turnover": round(float(np.abs(dP).sum(axis=(1, 2)).mean()), 3),
            "active_days": int((np.abs(P).sum(axis=(1, 2)) > 0).sum())}


# ── Walk-forward evaluation ───────────────────────────────────────────────────

STRATEGIES: dict[str, str] = {
    "xs_reversal_30m": "30-minute cross-sectional reversal (long losers / short winners), every 30 min",
    "gap_fade": "Opening-gap fade: long gap-downs / short gap-ups at 09:20, out by 11:00",
    "intraday_momentum": "Market intraday momentum: prev close→09:45 basket return sets a 15:00–15:20 trade",
    "intraday_pairs": "Intraday spread mean reversion on train-selected cointegrated pairs",
}


def strategy_weights(name: str, cube: Cube, ca: np.ndarray, pairs: Optional[list[PairSpec]] = None) -> np.ndarray:
    if name == "xs_reversal_30m":
        return xs_reversal_30m(cube, ca)
    if name == "gap_fade":
        return gap_fade(cube, ca)
    if name == "intraday_momentum":
        return intraday_momentum(cube, ca)
    if name == "intraday_pairs":
        return intraday_pairs(cube, ca, pairs or [])
    raise KeyError(name)


def _metrics(ret: pd.Series) -> dict:
    from app.quant.backtest import compute_metrics

    m = compute_metrics(ret.dropna())
    out = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
           for k, v in asdict(m).items() if k != "is_valid"}
    out["validated"] = bool(m.validate())
    out["positive_day_rate"] = round(float((ret > 0).sum() / max(1, (ret != 0).sum())), 4)
    return out


def evaluate(cube: Cube, oos_sessions: Optional[int] = None, costs: CostModel = CostModel(),
             stress: CostModel = CostModel(slippage_bps=5.0), splits: Optional[dict] = None) -> dict:
    from app.quant.backtest import MIN_OOS_TRADING_DAYS

    oos_sessions = oos_sessions or MIN_OOS_TRADING_DAYS
    D = cube.shape[0]
    if D <= oos_sessions + 120:
        raise ValueError(f"need > {oos_sessions + 120} sessions, have {D}")
    train_days = D - oos_sessions
    ca = corporate_action_mask(cube, splits)
    pairs = select_pairs(cube, train_days)
    results = {}
    free = CostModel(slippage_bps=0, stt_sell=0, exchange=0, sebi=0, stamp_buy=0)
    for name in STRATEGIES:
        W = strategy_weights(name, cube, ca, pairs)
        ret = simulate(cube, W, costs)
        ret_stress = simulate(cube, W, stress)
        gross_all = simulate(cube, W, free)
        gross = gross_all.iloc[train_days:]
        train_gross = gross_all.iloc[:train_days]
        P = np.zeros_like(W)
        P[:, 1:] = W[:, :-1]
        P[:, EXIT_SLOT:] = 0.0
        turnover = np.abs(np.diff(np.concatenate([np.zeros_like(P[:, :1]), P], axis=1), axis=1))[train_days:].sum()
        tr, te = ret.iloc[:train_days], ret.iloc[train_days:]
        results[name] = {
            "description": STRATEGIES[name],
            "train": _metrics(tr),
            "oos": _metrics(te),
            "oos_stress_5bp": _metrics(ret_stress.iloc[train_days:]),
            "oos_stress_2x": _metrics(simulate(cube, W, CostModel(slippage_bps=2 * costs.slippage_bps)).iloc[train_days:]),
            # Admission evidence for shadow forward tests — TRAIN window only (no OOS peeking).
            "train_gross_no_costs": _metrics(train_gross),
            "train_gross_mean_bp": round(float(train_gross.mean() * 1e4), 4),
            # Diagnostic only (never used for the gate): is there a signal before costs?
            "oos_gross_no_costs": _metrics(gross),
            "gross_edge_bp_per_unit_turnover": round(float(gross.sum() / max(turnover, 1e-12) * 1e4), 3),
            "avg_cost_bp_per_unit_turnover": round((costs.buy + costs.sell) / 2 * 1e4, 3),
            "trades_oos": trade_stats(Cube(cube.days[train_days:], cube.symbols, cube.open[train_days:],
                                           cube.close[train_days:]), W[train_days:]),
            "validated": _metrics(te)["validated"],
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": {"bars": "Kite 5-minute candles (as traded)", "sessions": D, "symbols": len(cube.symbols),
                 "first_day": str(cube.days[0]), "last_day": str(cube.days[-1]),
                 "train": [str(cube.days[0]), str(cube.days[train_days - 1])],
                 "oos": [str(cube.days[train_days]), str(cube.days[-1])],
                 "corporate_action_days_masked": int(ca.sum()),
                 "genuine_large_moves_kept": shock_days(cube, ca)},
        "execution": {"fill": "next 5-minute bar open after the decision bar's close",
                      "square_off": str(SLOT_TIMES[EXIT_SLOT]), "overnight": "never held"},
        "costs": {"base": asdict(costs) | {"buy_side": costs.buy, "sell_side": costs.sell},
                  "stress": asdict(stress) | {"buy_side": stress.buy, "sell_side": stress.sell}},
        "hypotheses_tested": len(STRATEGIES),
        "pairs_selected_on_train": [p.name for p in pairs],
        "gate": "walk-forward OOS: Sharpe ≥ 1, max drawdown ≤ 25 %, ≥ 504 sessions",
        "strategies": results,
        "disclaimer": "Historical simulation on as-traded 5-minute bars. Survivorship: universe is today's "
                      "active names. Not investment advice.",
    }


def load_cube(symbols: list[str]) -> Cube:
    from app.data import intraday_history as ih

    return build_cube(ih.panel(symbols, "open"), ih.panel(symbols, "close"))


def run(symbols: list[str], out_dir: Path = Path("docs/backtest_reports")) -> tuple[dict, Path]:
    from app.data import intraday_history as ih

    cube = load_cube(symbols)
    report = evaluate(cube, splits=ih.split_dates(cube.symbols))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"intraday_oos_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    return report, path

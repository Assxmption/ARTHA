"""
Dated Indian trading-cost schedules (statutory + exchange), per segment.

Source: Zerodha's published charges page and its STT article (checked
2026-11-27): "STT rates on futures and options (F&O) changed on 1 April 2026
… as per Budget 2026–27" — futures 0.02 % → 0.05 % (sell), options shorted
0.1 % → 0.15 % of premium, exercised 0.125 % → 0.15 % of intrinsic value.
Equity intraday is unchanged.

All rates are fractions of traded value (premium for options).  GST (18 %)
applies to brokerage + exchange + SEBI charges.  Brokerage is min(₹20, 0.03 %)
per order for intraday/futures and ₹20 flat for options; at ₹-lakh order sizes
the 0.03 % cap is never reached, so it is modelled per order elsewhere and
treated as part of slippage here.  Slippage is a modelling assumption (spread +
impact), not a statutory charge, and is set per use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

GST = 0.18
SEBI = 0.000001          # ₹10 / crore


@dataclass(frozen=True)
class Schedule:
    segment: str
    effective_from: date
    stt_sell: float          # on sell value (premium for options)
    stt_buy: float
    exchange: float          # NSE transaction charge
    stamp_buy: float
    note: str = ""

    def per_side(self, slippage_bps: float) -> tuple[float, float]:
        """(buy, sell) total cost per unit traded value, including slippage and GST on fees."""
        fees = (self.exchange + SEBI) * (1 + GST)
        slip = slippage_bps / 1e4
        return (slip + self.stamp_buy + self.stt_buy + fees, slip + self.stt_sell + fees)


SCHEDULES: list[Schedule] = [
    Schedule("equity_intraday", date(2000, 1, 1), stt_sell=0.00025, stt_buy=0.0, exchange=0.0000307,
             stamp_buy=0.00003, note="Unchanged by Budget 2026-27"),
    Schedule("index_futures", date(2000, 1, 1), stt_sell=0.0002, stt_buy=0.0, exchange=0.0000183,
             stamp_buy=0.00002, note="Pre 1-Apr-2026 STT 0.02 %"),
    Schedule("index_futures", date(2026, 4, 1), stt_sell=0.0005, stt_buy=0.0, exchange=0.0000183,
             stamp_buy=0.00002, note="Budget 2026-27: futures STT 0.05 % on sell"),
    Schedule("options_premium", date(2000, 1, 1), stt_sell=0.001, stt_buy=0.0, exchange=0.0003553,
             stamp_buy=0.00003, note="Pre 1-Apr-2026 STT 0.1 % of premium (shorted)"),
    Schedule("options_premium", date(2026, 4, 1), stt_sell=0.0015, stt_buy=0.0, exchange=0.0003553,
             stamp_buy=0.00003, note="Budget 2026-27: options STT 0.15 % of premium on sell"),
]


def schedule_for(segment: str, on: Optional[date] = None) -> Schedule:
    """The schedule in force for ``segment`` on ``on`` (default: today)."""
    on = on or date.today()
    candidates = [s for s in SCHEDULES if s.segment == segment and s.effective_from <= on]
    if not candidates:
        raise KeyError(f"no cost schedule for {segment!r} on {on}")
    return max(candidates, key=lambda s: s.effective_from)

"""
Paper-trading report and exports (CSV, XLSX, PDF).

Every figure is computed from the paper ledger (app/quant/paper_book.py):
positions and P&L are replayed from the order log, marks are the last
observed trade prices with their source and time.  Nothing is estimated or
projected.  "If closed now" is the unrealized P&L at the last mark minus the
same 15 bp exit cost the fills use — it is a liquidation value, not a price
target or a forecast.

Simulated trading only.  Not investment advice.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.paper_book import (
    BOOKS, COST_BPS, IST, MAX_PRICE_AGE, PaperLedger, PositionState, valuation,
)

DISCLAIMER = (
    "Simulated (paper) trading only — no real orders were placed with any broker. Fills are virtual, at the "
    "observed market price plus a 15 bp cost. Historical and simulated results do not predict future results. "
    "This is not investment advice and makes no recommendation to buy or sell any security."
)
SHORT_NOTE = (
    "Short positions are modelled as single-stock-futures-equivalent exposure; Indian cash equities cannot be "
    "held short overnight. Futures margin, lot sizes and rollover costs are not modelled."
)
MIN_SESSIONS_FOR_METRICS = 20


def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _metrics(nav_rows: list[dict], capital: float) -> dict:
    sessions = len(nav_rows)
    if sessions < MIN_SESSIONS_FOR_METRICS:
        return {"sessions": sessions, "status": "insufficient_history",
                "note": f"Risk metrics need at least {MIN_SESSIONS_FOR_METRICS} sessions of paper NAV; "
                        f"{sessions} so far. Not yet computed."}
    from app.quant.backtest import compute_metrics

    navs = pd.Series([capital] + [r["nav"] for r in nav_rows])
    m = compute_metrics(navs.pct_change().dropna())
    return {"sessions": sessions, "status": "computed", "sharpe_ratio": _r(m.sharpe_ratio, 3),
            "sortino_ratio": _r(m.sortino_ratio, 3), "max_drawdown": _r(m.max_drawdown, 4),
            "annualized_return": _r(m.annualized_return, 4), "volatility": _r(m.volatility, 4)}


def build_report(ledger: PaperLedger, now: Optional[datetime] = None, scheduler_status: Optional[dict] = None) -> dict:
    now = now or datetime.now(IST)
    marks = ledger.marks()
    books_meta = ledger.books()
    books = []
    for key, spec in BOOKS.items():
        meta = books_meta.get(key)
        if not meta:
            continue
        capital = meta["initial_capital"]
        orders = ledger.orders(key)
        positions, cash, realized, cost = ledger.state(key)
        val = valuation(positions, marks)
        nav = cash + val["long_value"] + val["short_value"]
        targets = ledger.targets(key)
        pos_rows = [_position_row(key, p, marks, targets, now) for p in positions.values()]
        pos_rows.sort(key=lambda r: -abs(r["market_value"] or 0))
        realized_rows = [
            {k: o[k] for k in ("id", "placed_at", "symbol", "side", "quantity", "fill_price", "position_before",
                               "position_after", "realized_pnl", "reason")}
            for o in orders if o["realized_pnl"] != 0 or o["position_after"] == 0
        ]
        nav_rows = ledger.nav_history(key)
        turnover = sum(o["notional"] for o in orders)
        total_pnl = nav - capital
        books.append({
            "book": key, "label": spec.label, "validated": spec.validated, "description": spec.description,
            "initial_capital": _r(capital), "cash": _r(cash), "long_value": _r(val["long_value"]),
            "short_value": _r(val["short_value"]),
            "gross_exposure": _r(val["long_value"] - val["short_value"]),
            "net_exposure": _r(val["long_value"] + val["short_value"]),
            "nav": _r(nav), "total_pnl": _r(total_pnl), "total_return_pct": _r(total_pnl / capital * 100, 4),
            "realized_pnl": _r(realized), "unrealized_pnl": _r(val["unrealized_pnl"]),
            "costs_paid": _r(cost), "turnover": _r(turnover),
            # Before costs = what the raw signal earned at observed prices; after costs (total_pnl) is official.
            "gross_pnl": _r(total_pnl + cost),
            "orders_count": len(orders), "buys": sum(o["side"] == "BUY" for o in orders),
            "sells": sum(o["side"] == "SELL" for o in orders), "open_positions": len(positions),
            "first_order_at": orders[0]["placed_at"] if orders else None,
            "last_order_at": orders[-1]["placed_at"] if orders else None,
            "unpriced_positions": val["unpriced"],
            "metrics": _metrics(nav_rows, capital),
            "positions": pos_rows, "orders": orders, "realized": realized_rows,
            "nav_history": nav_rows,
            "targets": [{"symbol": s, **{k: t[k] for k in ("weight", "reason", "signal_ref", "signal_as_of")}}
                        for s, t in sorted(targets.items())],
        })
    combined_capital = sum(b["initial_capital"] for b in books)
    combined_nav = sum(b["nav"] for b in books)
    return {
        "generated_at": now.isoformat(),
        "disclaimer": DISCLAIMER,
        "notes": [SHORT_NOTE,
                  f"Daily books fill at the reference price ± {COST_BPS:g} bp (as in the daily backtests). Intraday "
                  "books fill with side-aware NSE intraday costs (STT 0.025 % on sells, exchange, SEBI, stamp, GST, "
                  f"3 bp slippage). Orders fill only during NSE hours against a price observed within "
                  f"{int(MAX_PRICE_AGE.total_seconds() // 60)} minutes.",
                  "Before-cost P&L is a diagnostic of the raw signal at observed prices; after-cost P&L is the "
                  "official figure (real orders pay these charges even in a paper simulation's real-money twin).",
                  "Daily signals come from the live engine's last completed close; intraday signals from completed "
                  "5-minute bars. Shadow books are UNVALIDATED forward tests, reported separately.",
                  *[f"Restatement (epoch {e['epoch']} started {e['at'][:16].replace('T', ' ')}): {e['message']} "
                    f"Archive manifest: {e['ref']}" for e in ledger.events() if e["kind"] == "restatement"]],
        "epoch": ledger.epoch(),
        "events": ledger.events(),
        "research": _research_summary(),
        "scheduler": scheduler_status,
        "last_runs": ledger.runs(10),
        "combined": {
            "initial_capital": _r(combined_capital), "nav": _r(combined_nav),
            "total_pnl": _r(combined_nav - combined_capital),
            "total_return_pct": _r((combined_nav / combined_capital - 1) * 100, 4) if combined_capital else None,
            "realized_pnl": _r(sum(b["realized_pnl"] for b in books)),
            "unrealized_pnl": _r(sum(b["unrealized_pnl"] for b in books)),
            "gross_pnl": _r(sum(b["gross_pnl"] for b in books)),
            "costs_paid": _r(sum(b["costs_paid"] for b in books)),
            "orders_count": sum(b["orders_count"] for b in books),
            "note": "Sum of all books, including the UNVALIDATED shadow book.",
        },
        "books": books,
    }


def _research_summary() -> dict:
    """Gate constants and the latest intraday validation, straight from code and report JSON."""
    from app.quant import backtest

    out: dict = {"gate": {"min_oos_sharpe": backtest.MIN_OOS_SHARPE, "max_oos_drawdown": backtest.MAX_OOS_DRAWDOWN,
                          "min_oos_sessions": backtest.MIN_OOS_TRADING_DAYS, "risk_free_rate": 0.06}}
    try:
        from app.quant.intraday_paper import ADMISSION_RULE, admission, latest_validation

        rep = latest_validation()
    except Exception:
        rep = None
    if rep:
        adm = admission(rep)
        out["intraday"] = {
            "report_file": rep.get("_file"), "generated_at": rep.get("generated_at"),
            "hypotheses_tested": rep.get("hypotheses_tested"), "oos": rep.get("data", {}).get("oos"),
            "admission_rule": ADMISSION_RULE,
            "cost_per_unit_turnover_bp": next(iter(rep["strategies"].values())).get("avg_cost_bp_per_unit_turnover"),
            "strategies": {
                n: {"description": r.get("description"), "validated": r.get("validated"),
                    "oos_sharpe": r["oos"].get("sharpe_ratio"),
                    "oos_gross_sharpe": r.get("oos_gross_no_costs", {}).get("sharpe_ratio"),
                    "gross_edge_bp_per_unit_turnover": r.get("gross_edge_bp_per_unit_turnover"),
                    "train_gross_mean_bp": r.get("train_gross_mean_bp"),
                    "admission": adm.get(n)}
                for n, r in rep.get("strategies", {}).items()},
        }
    return out


def _position_row(book: str, p: PositionState, marks: dict, targets: dict, now: datetime) -> dict:
    m = marks.get(p.symbol)
    price = m.price if m else None
    basis = abs(p.quantity) * p.avg_price
    unreal = p.quantity * (price - p.avg_price) if price is not None else None
    exit_cost = abs(p.quantity) * price * COST_BPS / 10_000 if price is not None else None
    t = targets.get(p.symbol) or {}
    facts = t.get("facts") or {}
    comps = facts.get("components") or [facts]
    held_days = None
    if p.opened_at:
        held_days = (now.date() - datetime.fromisoformat(p.opened_at).astimezone(IST).date()).days
    return {
        "book": book, "symbol": p.symbol, "side": "LONG" if p.quantity > 0 else "SHORT",
        "quantity": p.quantity, "avg_price": _r(p.avg_price, 4), "last_price": _r(price, 4),
        "mark_time": m.observed_at.isoformat() if m else None, "mark_source": m.source if m else None,
        "market_value": _r(p.quantity * price) if price is not None else None, "cost_basis": _r(basis),
        "unrealized_pnl": _r(unreal), "unrealized_pct": _r(unreal / basis * 100, 3) if unreal is not None and basis else None,
        "exit_cost_estimate": _r(exit_cost),
        "if_closed_now": _r(unreal - exit_cost) if unreal is not None else None,
        "realized_pnl_so_far": _r(p.realized_pnl),
        "opened_at": p.opened_at, "held_days": held_days,
        "target_weight": _r(t.get("weight"), 6) if t else 0.0,
        "still_targeted": bool(t),
        "signal": t.get("reason") or "No longer targeted — exits at the next session with a fresh price",
        "exit_rule": "; ".join(sorted({c.get("exit_rule") for c in comps if c.get("exit_rule")})) or None,
        "z_score": next((c.get("z_score") for c in comps if c.get("z_score") is not None), None),
        "half_life_sessions": next((c.get("half_life_sessions") for c in comps if c.get("half_life_sessions")), None),
        "next_rebalance": facts.get("next_rebalance"),
    }


# ── Tables ────────────────────────────────────────────────────────────────────

ORDER_COLUMNS = [
    ("id", "Order #"), ("book", "Book"), ("placed_at", "Placed at (IST)"), ("symbol", "Symbol"),
    ("side", "Side"), ("quantity", "Qty"), ("ref_price", "Market price"), ("fill_price", "Fill price"),
    ("cost_bps", "Cost (bp)"), ("cost_inr", "Cost ₹"), ("notional", "Notional ₹"), ("position_before", "Position before"),
    ("position_after", "Position after"), ("realized_pnl", "Realized P&L ₹"), ("price_source", "Price source"),
    ("price_observed_at", "Price observed at"), ("signal_as_of", "Signal close"), ("signal_ref", "Signal"),
    ("reason", "Reason"),
]
POSITION_COLUMNS = [
    ("book", "Book"), ("symbol", "Symbol"), ("side", "Side"), ("quantity", "Qty"), ("avg_price", "Avg price"),
    ("last_price", "Last price"), ("mark_time", "Mark time"), ("mark_source", "Mark source"),
    ("market_value", "Market value ₹"), ("cost_basis", "Cost basis ₹"), ("unrealized_pnl", "Unrealized P&L ₹"),
    ("unrealized_pct", "Unrealized %"), ("exit_cost_estimate", "Exit cost ₹"), ("if_closed_now", "If closed now ₹"),
    ("opened_at", "Opened at"), ("held_days", "Days held"), ("target_weight", "Target weight"),
    ("z_score", "Pair z-score"), ("exit_rule", "Exit rule"), ("signal", "Signal"),
]
SUMMARY_COLUMNS = [
    ("label", "Book"), ("validated", "Validated"), ("initial_capital", "Start capital ₹"), ("nav", "NAV ₹"),
    ("total_pnl", "P&L after costs ₹"), ("total_return_pct", "Total return %"), ("gross_pnl", "P&L before costs ₹"),
    ("realized_pnl", "Realized ₹"),
    ("unrealized_pnl", "Unrealized ₹"), ("costs_paid", "Costs paid ₹"), ("cash", "Cash ₹"),
    ("long_value", "Long value ₹"), ("short_value", "Short value ₹"), ("gross_exposure", "Gross exposure ₹"),
    ("orders_count", "Orders"), ("buys", "Buys"), ("sells", "Sells"), ("open_positions", "Open positions"),
    ("turnover", "Turnover ₹"), ("first_order_at", "First order"), ("last_order_at", "Last order"),
]
REALIZED_COLUMNS = [
    ("book", "Book"), ("id", "Order #"), ("placed_at", "Placed at"), ("symbol", "Symbol"), ("side", "Side"),
    ("quantity", "Qty"), ("fill_price", "Fill price"), ("position_before", "Position before"),
    ("position_after", "Position after"), ("realized_pnl", "Realized P&L ₹"), ("reason", "Reason"),
]
NAV_COLUMNS = [
    ("book", "Book"), ("session_date", "Session"), ("ts", "Snapshot at"), ("nav", "NAV ₹"), ("cash", "Cash ₹"),
    ("long_value", "Long ₹"), ("short_value", "Short ₹"), ("realized_pnl", "Realized ₹"),
    ("unrealized_pnl", "Unrealized ₹"),
]


def table(report: dict, name: str) -> tuple[list[tuple[str, str]], list[dict]]:
    books = report["books"]
    if name == "orders":
        return ORDER_COLUMNS, [{**o, "cost_inr": round(abs(o["fill_price"] - o["ref_price"]) * o["quantity"], 2)}
                               for b in books for o in b["orders"]]
    if name == "positions":
        return POSITION_COLUMNS, [p for b in books for p in b["positions"]]
    if name == "summary":
        rows = list(books)
        c = report["combined"]
        rows.append({"label": "All books (incl. unvalidated shadow)", "validated": "mixed", **c})
        return SUMMARY_COLUMNS, rows
    if name == "realized":
        return REALIZED_COLUMNS, [{"book": b["book"], **r} for b in books for r in b["realized"]]
    if name == "nav":
        return NAV_COLUMNS, [n for b in books for n in b["nav_history"]]
    raise KeyError(name)


TABLES = ("summary", "orders", "positions", "realized", "nav")


def _cell(v):
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    return v


def to_csv(report: dict, name: str) -> str:
    cols, rows = table(report, name)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in cols])
    for r in rows:
        w.writerow([_cell(r.get(k)) for k, _ in cols])
    return buf.getvalue()


def to_csv_zip(report: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in TABLES:
            z.writestr(f"paper_{name}.csv", "﻿" + to_csv(report, name))   # BOM so Excel reads ₹
        z.writestr("README.txt", f"{DISCLAIMER}\n\n" + "\n".join(report["notes"]) + f"\n\nGenerated {report['generated_at']}\n")
    return buf.getvalue()


# ── XLSX ──────────────────────────────────────────────────────────────────────

_MONEY_HINT = ("₹", "price", "Price")


def to_xlsx(report: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F2937")
    pos_font, neg_font = Font(color="047857"), Font(color="B91C1C")

    def sheet(ws, name):
        cols, rows = table(report, name)
        ws.append([h for _, h in cols])
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(vertical="center", wrap_text=True)
        for r in rows:
            ws.append([_cell(r.get(k)) for k, _ in cols])
        for j, (key, header) in enumerate(cols, start=1):
            letter = get_column_letter(j)
            width = min(max(len(header), *(len(str(_cell(r.get(key)) or "")) for r in rows), 8) + 2, 60) if rows else len(header) + 2
            ws.column_dimensions[letter].width = width
            money = any(h in header for h in _MONEY_HINT)
            pct = "%" in header
            for cell in ws[letter][1:]:
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    if pct:
                        cell.number_format = "0.00"
                    elif money:
                        cell.number_format = "#,##0.00"
                    if "P&L" in header or header.startswith(("Unrealized", "Realized", "Total", "If closed")):
                        if cell.value:
                            cell.font = pos_font if cell.value > 0 else neg_font
        ws.freeze_panes = "A2"
        if rows:
            ws.auto_filter.ref = ws.dimensions

    ws = wb.active
    ws.title = "Summary"
    sheet(ws, "summary")
    ws.append([])
    ws.append([f"Generated {report['generated_at']}"])
    ws.append([report["disclaimer"]])
    for n in report["notes"]:
        ws.append([n])
    for b in report["books"]:
        m = b["metrics"]
        ws.append([f"{b['label']} risk metrics", m.get("note") or
                   f"Sharpe {m['sharpe_ratio']}, Sortino {m['sortino_ratio']}, max DD {m['max_drawdown']} over {m['sessions']} sessions"])
    for title, name in (("Orders", "orders"), ("Open positions", "positions"),
                        ("Realized", "realized"), ("NAV history", "nav")):
        sheet(wb.create_sheet(title), name)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── PDF ───────────────────────────────────────────────────────────────────────

def inr_group(v: float, decimals: int = 2) -> str:
    """Indian digit grouping: 12345678.9 → '1,23,45,678.90'."""
    sign = "-" if v < 0 else ""
    whole, _, frac = f"{abs(v):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    body = ",".join([*groups, tail]) if groups else tail
    return f"{sign}{body}" + (f".{frac}" if decimals else "")


def _fmt(v, money=False, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool)):
        if pct:
            return f"{v:+.2f}%"
        if money:
            return inr_group(v)
        return f"{v:,}" if isinstance(v, int) else f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def to_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from xml.sax.saxutils import escape

    from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    font, bold = "Helvetica", "Helvetica-Bold"
    rupee = "Rs "
    # Use a TTF that really contains the ₹ glyph (U+20B9) when one exists on this
    # machine (DejaVu in Docker/Linux, Geneva on macOS); otherwise write "Rs".
    from reportlab.lib.fonts import addMapping

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Geneva.ttf", "/Library/Fonts/DejaVuSans.ttf"):
        try:
            ttf = TTFont("ArthaSans", path)
            if 0x20B9 not in ttf.face.charToGlyph:
                continue
            pdfmetrics.registerFont(ttf)
            for b_, i_ in ((0, 0), (1, 0), (0, 1), (1, 1)):
                addMapping("ArthaSans", b_, i_, "ArthaSans")
            font = bold = "ArthaSans"
            rupee = "₹"
            break
        except Exception:
            continue

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName=font, fontSize=8, leading=10)
    small = ParagraphStyle("s", parent=body, fontSize=6.5, leading=8)
    head = ParagraphStyle("hd", parent=small, textColor=colors.white)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=16)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=11, spaceBefore=8)

    def tbl(header, rows, widths):
        data = [[Paragraph(f"<b>{escape(h)}</b>", head) for h in header]]
        data += [[Paragraph(escape(str(c)), small) for c in r] for r in rows]
        t = LongTable(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    def money(v):
        return "—" if v is None else f"{rupee}{inr_group(v)}"

    story = [Paragraph("ARTHA — Paper Trading Report", h1),
             Paragraph(f"Generated {report['generated_at']} (IST). Simulated orders only.", body),
             Spacer(1, 3 * mm), Paragraph(escape(report["disclaimer"]), body)]
    for n in report["notes"]:
        story.append(Paragraph("• " + escape(n), small))

    story.append(Paragraph("Summary", h2))
    rows = []
    for b in report["books"]:
        rows.append([b["label"], "Yes" if b["validated"] else "NO", money(b["initial_capital"]), money(b["nav"]),
                     money(b["total_pnl"]), _fmt(b["total_return_pct"], pct=True), money(b["gross_pnl"]),
                     money(b["costs_paid"]), money(b["unrealized_pnl"]), b["orders_count"], b["open_positions"]])
    c = report["combined"]
    rows.append(["All books (incl. unvalidated)", "mixed", money(c["initial_capital"]), money(c["nav"]),
                 money(c["total_pnl"]), _fmt(c["total_return_pct"], pct=True), money(c["gross_pnl"]),
                 money(c["costs_paid"]), money(c["unrealized_pnl"]), c["orders_count"], ""])
    ep = report.get("epoch") or {}
    if ep.get("epoch", 1) > 1:
        story.append(Paragraph(escape(f"Epoch {ep['epoch']} — books restated {str(ep.get('started_at'))[:16].replace('T', ' ')}; "
                                      f"the previous epoch is archived (see notes)."), body))
    story.append(tbl(["Book", "Validated", "Start capital", "NAV", "P&L after costs", "Return", "P&L before costs",
                      "Costs", "Unrealized", "Orders", "Open"], rows,
                     [52 * mm, 15 * mm, 28 * mm, 28 * mm, 24 * mm, 16 * mm, 22 * mm, 22 * mm, 24 * mm, 13 * mm, 11 * mm]))
    for b in report["books"]:
        m = b["metrics"]
        story.append(Paragraph(
            f"<b>{b['label']}</b> — " + (m.get("note") or
            f"Sharpe {m['sharpe_ratio']}, Sortino {m['sortino_ratio']}, max drawdown {m['max_drawdown']} "
            f"over {m['sessions']} sessions."), small))

    for b in report["books"]:
        story += [PageBreak(), Paragraph(f"{b['label']} — open positions", h2), Paragraph(escape(b["description"]), small)]
        if b["positions"]:
            story.append(tbl(
                ["Symbol", "Side", "Qty", "Avg", "Last", "Mkt value", "Unrealized", "%", "If closed now", "Since",
                 "Exit rule / signal"],
                [[p["symbol"], p["side"], inr_group(p['quantity'], 0), _fmt(p["avg_price"], money=True),
                  _fmt(p["last_price"], money=True), money(p["market_value"]), money(p["unrealized_pnl"]),
                  _fmt(p["unrealized_pct"], pct=True), money(p["if_closed_now"]),
                  (p["opened_at"] or "")[:16].replace("T", " "),
                  (p["exit_rule"] or "") + (f" (z = {p['z_score']})" if p.get("z_score") is not None else "")]
                 for p in b["positions"]],
                [20 * mm, 12 * mm, 16 * mm, 18 * mm, 18 * mm, 26 * mm, 24 * mm, 14 * mm, 24 * mm, 24 * mm, 68 * mm]))
        else:
            story.append(Paragraph("No open positions.", body))
        story.append(Paragraph(f"{b['label']} — every order placed ({b['orders_count']})", h2))
        if b["orders"]:
            story.append(tbl(
                ["#", "Placed at", "Symbol", "Side", "Qty", "Market", "Fill", "Notional", "Pos after",
                 "Realized", "Price source", "Reason"],
                [[o["id"], o["placed_at"][:19].replace("T", " "), o["symbol"], o["side"], inr_group(o['quantity'], 0),
                  _fmt(o["ref_price"], money=True), _fmt(o["fill_price"], money=True), money(o["notional"]),
                  inr_group(o['position_after'], 0), money(o["realized_pnl"]), o["price_source"], o["reason"]]
                 for o in b["orders"]],
                [10 * mm, 26 * mm, 18 * mm, 10 * mm, 14 * mm, 16 * mm, 16 * mm, 24 * mm, 14 * mm, 20 * mm, 22 * mm, 76 * mm]))
        else:
            story.append(Paragraph("No orders yet.", body))

    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 6.5)
        canvas.drawString(12 * mm, 7 * mm, "ARTHA paper trading — simulated orders only. Not investment advice.")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=10 * mm, rightMargin=10 * mm,
                      topMargin=10 * mm, bottomMargin=12 * mm, title="ARTHA Paper Trading Report",
                      author="ARTHA").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()

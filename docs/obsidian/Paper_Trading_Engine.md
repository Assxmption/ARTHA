# Paper Trading Engine (Paper Books)

**Path**: `app/quant/paper_book.py`, `app/quant/paper_report.py`, `app/api/paper_routes.py`

Simulated orders only — no broker order API is imported or called (AGENTS.md rule 9).

## Books
- **validated** — only signals that pass the walk-forward OOS gate (Sharpe ≥ 1, max DD ≤ 25 %, ≥ 504 sessions); mirrors each validated pair's backtest position.
- **shadow** — the predeclared sleeve portfolio, which fails the gate; paper-traded as a forward test, always labelled UNVALIDATED and reported separately.

## Mechanics
- Signals come from the live engine ([[Live_Strategy_Engine]] / `live_dashboard.py`) at each completed close; fills happen in the next NSE session, only against prices observed today within 20 minutes (Kite LTP, else Yahoo 1-minute bars).
- Fill = reference price ± 15 bp (the backtests' cost). Shorts are modelled as futures-equivalent exposure.
- Positions, cash and realized P&L are replayed from the SQLite order log (average cost), so the ledger is auditable.
- `PaperScheduler` runs every 5 min in session and every 30 min otherwise (marks only), behind a file lock; holiday-aware via `app/data/nse_calendar.py`.

## API
- `GET /api/paper/status`, `GET /api/paper/report`
- `GET /api/paper/export/{csv|zip|xlsx|pdf}`
- `POST /api/paper/run` (operator only)

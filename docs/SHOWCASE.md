# ARTHA — showcase notes (5 minutes)

Run it **locally** with live Kite data. Keep the exported PDF/XLSX and screenshots as a fallback.

## Before you start (demo morning)
1. Start the servers:
   - `uvicorn app.main:app --port 8000`
   - `cd frontend && npx vite`
2. Sign in to ARTHA.
3. Settings → Broker → **Connect Kite**. The token expires at 06:00 every day.
4. Open **Paper Trading**, and leave **Experiments** collapsed.
5. Have `docs/backtest_reports/intraday_oos_*.json` and the PDF export ready.

Saturday or Sunday: there is no NSE session, so the books show no live fills. Show the order log from Friday's archived epoch, and the session replay (Dashboard → Session replay), labelled **replay**.

## Talk track

**1. The problem (30 s).**
Most "AI trading" demos let a model invent numbers. In ARTHA, no LLM ever computes or asserts a number. Every figure comes from deterministic code, and every report figure traces back to a stored fact.

**2. The gate (45 s).** Point to the *Validation gate* panel.
- A signal only counts if, on data it never saw (the last 504 sessions), it beats Sharpe 1.0 after a 6% risk-free rate, with max drawdown ≤ 25%.
- Everything else is at best a labelled experiment.

**3. Live data, simulated fills (45 s).**
- Kite Connect is read-only: quotes, 5-minute bars, option chains.
- The paper engine fills at live prices and charges the real Indian costs: STT, exchange and SEBI fees, stamp duty, GST and slippage.
- No order is ever sent to a broker.

**4. One validated signal (45 s).** KOTAKBANK/PNB stat-arb pair:
- Out-of-sample Sharpe 1.31, max drawdown 6.6%, 504 sessions (source: `data_cache/live/dashboard.json`, recomputed daily).
- It is currently **flat**: its backtest holds no position. So the validated books sit at their starting capital. Flat is the honest result.

**5. Intraday research: rejected, with reasons (75 s).** Point to the *Why most retail intraday strategies fail* panel.
- 4 predeclared hypotheses on 4 years of Kite 5-minute data: 30-min reversal, gap fade, market intraday momentum, intraday pairs. **None passed.**
- The best pattern earns about 0.6 bp per ₹ traded before costs. Costs are about 4.8 bp per ₹ traded.
- The 1-Apr-2026 STT hike (futures 0.02% → 0.05%) narrows the futures shortcut.
- Source: `docs/backtest_reports/intraday_oos_*.json`.

**6. A bug we found and disclosed (30 s).**
- On 27-Nov the paper engine briefly traded on a stale signal. We fixed it, archived that epoch unchanged (sha256 in the manifest), and restarted all books together.
- The restatement note is on the page and in every export.

**7. What's next (30 s).**
- Index-futures intraday line v2, with pre-registered hypotheses, deflated Sharpe and PBO checks.
- Live recording of F&O chains, because real option history doesn't exist for free.
- A futures paper book with lot sizing.
- See the plan in `docs/` and the repo.

## If asked "why is it not making money?"
Because it refuses to trade anything that hasn't earned it. A system that is green on day one with unvalidated strategies is either lucky or lying. Today's honest numbers are:
- validated books flat;
- experiments clearly labelled;
- every cost shown.

*Not investment advice. Simulated trading only.*

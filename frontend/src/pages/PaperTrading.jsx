import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity, AlertTriangle, Download, FileSpreadsheet, FileText, Loader2, Play, ShieldCheck, Table2,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { setActiveSymbol } from '../lib/activeSymbol';
import { apiFetch } from '../lib/api';

// Paper books: simulated orders driven by ARTHA's deterministic signals.
// Every number on this page comes from /api/paper/report (the order ledger);
// nothing is computed or projected here except display formatting.

const inr = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const inr0 = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });
const money = (v) => (v == null ? '—' : `₹${inr.format(v)}`);
const moneyShort = (v) => {
  if (v == null) return '—';
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return `₹${inr.format(v)}`;
};
const pct = (v, d = 2) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}%`);
const tone = (v) => (v > 0 ? 'text-emerald-500' : v < 0 ? 'text-rose-500' : 'text-on-surface');
const when = (iso) => (iso ? new Date(iso).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—');

export default function PaperTrading() {
  const { session } = useAuth();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [book, setBook] = useState('validated');
  const [showExperiments, setShowExperiments] = useState(false);
  const [running, setRunning] = useState(false);
  const [downloading, setDownloading] = useState(null);
  const [orderFilter, setOrderFilter] = useState('');

  const auth = useMemo(
    () => (session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {}),
    [session?.access_token],
  );

  const load = useCallback(async () => {
    try {
      const res = await apiFetch('/api/paper/report', { headers: auth });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setReport(await res.json());
      setError(null);
    } catch (e) {
      setError(`Could not load the paper report (${e.message}).`);
    }
  }, [auth]);

  useEffect(() => {
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, [load]);

  const runNow = async () => {
    setRunning(true);
    try {
      const res = await apiFetch('/api/paper/run', { method: 'POST', headers: auth });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      await load();
    } catch (e) {
      setError(`Cycle failed: ${e.message}`);
    } finally {
      setRunning(false);
    }
  };

  // Downloads go through fetch so the bearer token is sent when auth is enabled.
  const download = async (fmt, table) => {
    const key = table ? `${fmt}:${table}` : fmt;
    setDownloading(key);
    try {
      const qs = table ? `?table=${encodeURIComponent(table)}` : '';
      const res = await apiFetch(`/api/paper/export/${fmt}${qs}`, { headers: auth });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const name = /filename="([^"]+)"/.exec(res.headers.get('content-disposition') || '')?.[1] || `artha_paper.${fmt}`;
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement('a'), { href: url, download: name });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError(`Export failed: ${e.message}`);
    } finally {
      setDownloading(null);
    }
  };

  const openChart = (symbol) => { setActiveSymbol(symbol); navigate(`/dashboard/${encodeURIComponent(symbol)}`); };

  if (!report) {
    return (
      <div className="p-6 bg-surface-dim min-h-full font-mono text-sm text-outline">
        {error || <span className="flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Loading paper books…</span>}
      </div>
    );
  }

  const allBooks = report.books || [];
  const validatedBooks = allBooks.filter((b) => b.validated);
  const experimentBooks = allBooks.filter((b) => !b.validated);
  const books = showExperiments ? allBooks : validatedBooks;
  const current = books.find((b) => b.book === book) || books[0];
  const sumOf = (list, key) => list.reduce((a, b) => a + (b[key] || 0), 0);
  const vSum = {
    capital: sumOf(validatedBooks, 'initial_capital'), nav: sumOf(validatedBooks, 'nav'),
    net: sumOf(validatedBooks, 'total_pnl'), gross: sumOf(validatedBooks, 'gross_pnl'),
    costs: sumOf(validatedBooks, 'costs_paid'), orders: sumOf(validatedBooks, 'orders_count'),
  };
  const epoch = report.epoch || {};
  const restatement = (report.events || []).filter((e) => e.kind === 'restatement').slice(-1)[0];
  const research = report.research || {};
  const intra = research.intraday;
  const sched = report.scheduler || {};
  const lastRun = report.last_runs?.[0];
  const orders = [...(current?.orders || [])].reverse()
    .filter((o) => !orderFilter || o.symbol.includes(orderFilter.toUpperCase()));

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full space-y-5">
      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between gap-4 border-b border-outline-variant pb-4">
        <div>
          <h1 className="font-display text-3xl text-on-surface">Paper Trading</h1>
          <p className="text-outline text-xs mt-1 font-mono">
            Simulated orders only — ARTHA never places a real order. Live Kite prices; every fill pays modelled Indian trading costs.
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-3 font-mono text-[11px]">
            <span className="flex items-center gap-1.5 px-2 py-1 border border-outline-variant bg-surface">
              <span className={`w-2 h-2 rounded-full ${sched.state === 'running' ? 'bg-emerald-500 animate-pulse' : 'bg-outline-variant'}`} />
              Engine {sched.state || 'unknown'}
            </span>
            <span className="text-outline">Last cycle {when(lastRun?.finished_at || lastRun?.started_at)}</span>
            {sched.next_run_at && <span className="text-outline">Next {when(sched.next_run_at)}</span>}
            {lastRun?.note && <span className="text-on-surface-variant">{lastRun.note}</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-start gap-2">
          <button onClick={runNow} disabled={running}
            className="flex items-center gap-2 px-3 py-2 border border-outline-variant bg-surface hover:border-primary text-on-surface font-ui text-[11px] uppercase tracking-widest disabled:opacity-50">
            {running ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />} Run cycle now
          </button>
          <ExportButton icon={<FileText size={13} />} label="PDF" busy={downloading === 'pdf'} onClick={() => download('pdf')} />
          <ExportButton icon={<FileSpreadsheet size={13} />} label="Excel" busy={downloading === 'xlsx'} onClick={() => download('xlsx')} />
          <ExportButton icon={<Download size={13} />} label="CSV (all)" busy={downloading === 'zip'} onClick={() => download('zip')} />
        </div>
      </div>

      {error && <p className="font-mono text-xs text-error border border-error/40 px-3 py-2">{error}</p>}

      {epoch.epoch > 1 && restatement && (
        <div className="px-3 py-2 border border-primary/40 bg-primary/5 font-mono text-[11px] text-on-surface-variant">
          <span className="text-primary font-bold">Epoch {epoch.epoch}</span> · books restated {when(epoch.started_at)}. {restatement.message}
          <span className="text-outline"> Previous epoch archived (manifest: {restatement.ref}).</span>
        </div>
      )}

      {/* Validated books — the headline */}
      <section>
        <h2 className="font-ui text-[10px] uppercase tracking-widest text-outline mb-2">Validated books <span className="normal-case tracking-normal">— only signals that passed the walk-forward out-of-sample gate</span></h2>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
          <Stat label="Start capital" value={moneyShort(vSum.capital)} />
          <Stat label="NAV" value={moneyShort(vSum.nav)} />
          <Stat label="P&L after costs" value={money(vSum.net)} sub="official" cls={tone(vSum.net)} />
          <Stat label="P&L before costs" value={money(vSum.gross)} sub="diagnostic" cls={tone(vSum.gross)} />
          <Stat label="Costs paid" value={money(vSum.costs)} />
          <Stat label="Orders" value={inr0.format(vSum.orders)} sub={vSum.orders === 0 ? 'flat = no validated signal active' : undefined} />
        </div>
      </section>

      {/* Rigor + cost reality */}
      <div className="grid lg:grid-cols-2 gap-3">
        <section className="bg-surface border border-outline-variant p-3 font-mono text-[11px] space-y-1.5">
          <h2 className="font-ui text-[10px] uppercase tracking-widest text-on-surface flex items-center gap-1.5"><ShieldCheck size={12} className="text-emerald-500" /> Validation gate</h2>
          {research.gate && (
            <p className="text-on-surface-variant">
              Out-of-sample Sharpe ≥ {research.gate.min_oos_sharpe} (after a {Math.round(research.gate.risk_free_rate * 100)}% risk-free rate),
              max drawdown ≤ {Math.round(research.gate.max_oos_drawdown * 100)}%, ≥ {research.gate.min_oos_sessions} out-of-sample sessions.
            </p>
          )}
          <p className="text-outline">Strategies that fail it never enter a validated book. They may only run as clearly labelled experiments.</p>
          {intra && (
            <p className="text-on-surface-variant">Intraday research: {Object.values(intra.strategies).filter((x) => x.validated).length} of {intra.hypotheses_tested} predeclared hypotheses passed (OOS {intra.oos?.[0]} → {intra.oos?.[1]}).</p>
          )}
        </section>
        {intra && (
          <section className="bg-surface border border-outline-variant p-3 font-mono text-[11px]">
            <h2 className="font-ui text-[10px] uppercase tracking-widest text-on-surface mb-2 flex items-center gap-1.5"><AlertTriangle size={12} className="text-amber-500" /> Why most retail intraday strategies fail</h2>
            <p className="text-outline mb-2">Edge earned per ₹ traded, before costs, vs what each ₹ traded costs (STT, fees, stamp, GST, slippage ≈ {intra.cost_per_unit_turnover_bp} bp):</p>
            <table className="w-full">
              <tbody>
                {Object.entries(intra.strategies).map(([name, st]) => {
                  const edge = st.gross_edge_bp_per_unit_turnover;
                  const cost = intra.cost_per_unit_turnover_bp || 1;
                  const w = Math.min(100, Math.max(0, (Math.abs(edge) / cost) * 100));
                  return (
                    <tr key={name} className="align-middle">
                      <td className="py-0.5 pr-2 text-on-surface-variant whitespace-nowrap">{name}</td>
                      <td className="w-full">
                        <div className="h-2 bg-outline-variant/30 relative" title={`${edge} bp vs ${cost} bp`}>
                          <div className={`h-2 ${edge > 0 ? 'bg-emerald-500/70' : 'bg-rose-500/70'}`} style={{ width: `${w}%` }} />
                        </div>
                      </td>
                      <td className={`pl-2 text-right whitespace-nowrap ${tone(edge)}`}>{edge > 0 ? '+' : ''}{edge} bp</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-outline mt-2">Full bar = the cost of trading. Even the best pattern earns about a tenth of it. Since 1 Apr 2026, futures STT is 0.05% on sells, so cheaper instruments only partly close the gap.</p>
          </section>
        )}
      </div>

      {/* Book tabs */}
      <div className="flex flex-wrap items-end justify-between gap-2 border-b border-outline-variant">
        <div className="flex flex-wrap gap-1">
          {books.map((b) => (
            <button key={b.book} onClick={() => setBook(b.book)}
              className={`px-4 py-2 font-ui text-[11px] uppercase tracking-widest border-b-2 -mb-px flex items-center gap-2 ${b.book === current?.book ? 'border-primary text-on-surface' : 'border-transparent text-outline hover:text-on-surface'}`}>
              {b.validated ? <ShieldCheck size={13} className="text-emerald-500" /> : <AlertTriangle size={13} className="text-amber-500" />}
              {b.label}
              <span className="font-mono normal-case tracking-normal text-outline">{b.orders_count}</span>
            </button>
          ))}
        </div>
        {experimentBooks.length > 0 && (
          <button onClick={() => { setShowExperiments((v) => !v); if (showExperiments && !current?.validated) setBook('validated'); }}
            className="mb-1 font-ui text-[10px] uppercase tracking-widest text-outline hover:text-primary">
            {showExperiments ? 'Hide experiments' : `Show experiments (${experimentBooks.length} unvalidated forward tests)`}
          </button>
        )}
      </div>

      {showExperiments && intra && (
        <div className="px-3 py-2 border border-amber-500/40 bg-amber-500/5 font-mono text-[10px] text-on-surface-variant space-y-0.5">
          <p className="text-amber-600 dark:text-amber-400">{intra.admission_rule}</p>
          {Object.entries(intra.strategies).map(([n, st]) => <p key={n}>• {n}: {st.admission?.reason}</p>)}
        </div>
      )}

      {current && (
        <>
          <div className={`px-3 py-2 border font-mono text-[11px] ${current.validated ? 'border-emerald-500/40 text-on-surface-variant' : 'border-amber-500/50 bg-amber-500/5 text-amber-600 dark:text-amber-400'}`}>
            {current.description}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
            <Stat label="NAV" value={moneyShort(current.nav)} />
            <Stat label="P&L after costs" value={money(current.total_pnl)} sub={pct(current.total_return_pct, 3)} cls={tone(current.total_pnl)} />
            <Stat label="P&L before costs" value={money(current.gross_pnl)} sub="diagnostic" cls={tone(current.gross_pnl)} />
            <Stat label="Costs paid" value={money(current.costs_paid)} />
            <Stat label="Unrealized" value={money(current.unrealized_pnl)} cls={tone(current.unrealized_pnl)} />
            <Stat label="Cash" value={moneyShort(current.cash)} />
            <Stat label="Long / short" value={`${moneyShort(current.long_value)} / ${moneyShort(current.short_value)}`} />
            <Stat label="Orders" value={`${current.orders_count}`} sub={`${current.buys} buy · ${current.sells} sell`} />
          </div>
          <p className="font-mono text-[10px] text-outline">
            Risk metrics: {current.metrics?.note || `Sharpe ${current.metrics?.sharpe_ratio}, Sortino ${current.metrics?.sortino_ratio}, max drawdown ${current.metrics?.max_drawdown} over ${current.metrics?.sessions} sessions`}
          </p>

          {/* Positions */}
          <Section title={`Still held (${current.positions.length})`} onCsv={() => download('csv', 'positions')} busy={downloading === 'csv:positions'}
            hint="“If closed now” = unrealized P&L at the last price minus the 15 bp exit cost. It is a liquidation value, not a forecast.">
            {current.positions.length === 0 ? <Empty text="No open positions." /> : (
              <Table head={['Symbol', 'Side', 'Qty', 'Avg', 'Last', 'Value', 'Unrealized', '%', 'If closed now', 'Since', 'Exit rule']}>
                {current.positions.map((p) => (
                  <tr key={p.symbol} className="border-t border-outline-variant/40">
                    <td className="px-2 py-1.5"><button onClick={() => openChart(p.symbol)} className="text-primary hover:underline">{p.symbol}</button></td>
                    <td className={`px-2 ${p.side === 'LONG' ? 'text-emerald-500' : 'text-rose-500'}`}>{p.side}</td>
                    <td className="px-2 text-right">{inr0.format(p.quantity)}</td>
                    <td className="px-2 text-right">{inr.format(p.avg_price)}</td>
                    <td className="px-2 text-right" title={`${p.mark_source} · ${when(p.mark_time)}`}>{p.last_price == null ? '—' : inr.format(p.last_price)}</td>
                    <td className="px-2 text-right">{moneyShort(p.market_value)}</td>
                    <td className={`px-2 text-right ${tone(p.unrealized_pnl)}`}>{money(p.unrealized_pnl)}</td>
                    <td className={`px-2 text-right ${tone(p.unrealized_pct)}`}>{pct(p.unrealized_pct)}</td>
                    <td className={`px-2 text-right ${tone(p.if_closed_now)}`}>{money(p.if_closed_now)}</td>
                    <td className="px-2 whitespace-nowrap">{when(p.opened_at)}</td>
                    <td className="px-2 text-on-surface-variant min-w-[220px]">
                      {p.exit_rule || '—'}{p.z_score != null && <span className="text-outline"> · z = {p.z_score}</span>}
                      {!p.still_targeted && <span className="text-amber-500"> · exits next session</span>}
                    </td>
                  </tr>
                ))}
              </Table>
            )}
          </Section>

          {/* Orders */}
          <Section title={`Every order placed (${current.orders_count})`} onCsv={() => download('csv', 'orders')} busy={downloading === 'csv:orders'}
            extra={<input value={orderFilter} onChange={(e) => setOrderFilter(e.target.value)} placeholder="Filter symbol"
              className="bg-surface border border-outline-variant px-2 py-1 text-[11px] font-mono w-32 focus:outline-none focus:border-primary" />}>
            {orders.length === 0 ? <Empty text={current.orders_count ? 'No orders match.' : 'No orders yet. Orders are placed during NSE hours once fresh prices arrive.'} /> : (
              <Table head={['#', 'Placed (IST)', 'Symbol', 'Side', 'Qty', 'Market', 'Fill', 'Notional', 'Position', 'Realized', 'Price', 'Reason']}>
                {orders.map((o) => (
                  <tr key={o.id} className="border-t border-outline-variant/40 align-top">
                    <td className="px-2 py-1.5 text-outline">{o.id}</td>
                    <td className="px-2 whitespace-nowrap">{when(o.placed_at)}</td>
                    <td className="px-2">{o.symbol}</td>
                    <td className={`px-2 ${o.side === 'BUY' ? 'text-emerald-500' : 'text-rose-500'}`}>{o.side}</td>
                    <td className="px-2 text-right">{inr0.format(o.quantity)}</td>
                    <td className="px-2 text-right">{inr.format(o.ref_price)}</td>
                    <td className="px-2 text-right">{inr.format(o.fill_price)}</td>
                    <td className="px-2 text-right">{moneyShort(o.notional)}</td>
                    <td className="px-2 text-right whitespace-nowrap">{inr0.format(o.position_before)} → {inr0.format(o.position_after)}</td>
                    <td className={`px-2 text-right ${tone(o.realized_pnl)}`}>{o.realized_pnl ? money(o.realized_pnl) : '—'}</td>
                    <td className="px-2 text-outline whitespace-nowrap" title={o.price_observed_at}>{o.price_source}</td>
                    <td className="px-2 text-on-surface-variant min-w-[280px]">{o.reason}</td>
                  </tr>
                ))}
              </Table>
            )}
          </Section>
        </>
      )}

      {/* Engine log */}
      <Section title="Engine cycles" icon={<Activity size={12} />}>
        <Table head={['Started', 'Session', 'Signals', 'Orders', 'Note']}>
          {(report.last_runs || []).map((r) => (
            <tr key={r.id} className="border-t border-outline-variant/40">
              <td className="px-2 py-1 whitespace-nowrap">{when(r.started_at)}</td>
              <td className="px-2">{r.session_open ? 'open' : 'closed'}</td>
              <td className="px-2">{r.signal_as_of || '—'}</td>
              <td className="px-2 text-right">{r.n_orders}</td>
              <td className="px-2 text-on-surface-variant">{r.note}</td>
            </tr>
          ))}
        </Table>
      </Section>

      <footer className="border-t border-outline-variant pt-3 space-y-1 font-mono text-[10px] text-outline">
        <p className="text-on-surface-variant">{report.disclaimer}</p>
        {(report.notes || []).map((n) => <p key={n}>• {n}</p>)}
        <p>Report generated {when(report.generated_at)}.</p>
      </footer>
    </div>
  );
}

function ExportButton({ icon, label, busy, onClick }) {
  return (
    <button onClick={onClick} disabled={busy}
      className="flex items-center gap-2 px-3 py-2 border border-outline-variant bg-surface hover:border-primary text-on-surface font-ui text-[11px] uppercase tracking-widest disabled:opacity-50">
      {busy ? <Loader2 size={13} className="animate-spin" /> : icon} {label}
    </button>
  );
}

function Stat({ label, value, sub, cls = 'text-on-surface' }) {
  return (
    <div className="bg-surface border border-outline-variant px-3 py-2 min-w-0">
      <div className="font-ui text-[9px] uppercase tracking-widest text-outline">{label}</div>
      <div className={`font-mono text-sm mt-0.5 truncate ${cls}`} title={typeof value === 'string' ? value : undefined}>{value}</div>
      {sub && <div className={`font-mono text-[10px] ${cls}`}>{sub}</div>}
    </div>
  );
}

function Section({ title, hint, children, onCsv, busy, extra, icon }) {
  return (
    <section className="bg-surface border border-outline-variant">
      <div className="flex flex-wrap justify-between items-center gap-2 px-3 py-2 border-b border-outline-variant">
        <h2 className="font-ui text-[10px] uppercase tracking-widest text-on-surface flex items-center gap-1.5">{icon}{title}</h2>
        <div className="flex items-center gap-2">
          {extra}
          {onCsv && (
            <button onClick={onCsv} disabled={busy} className="flex items-center gap-1 font-ui text-[10px] uppercase tracking-widest text-outline hover:text-primary disabled:opacity-50">
              {busy ? <Loader2 size={11} className="animate-spin" /> : <Table2 size={11} />} CSV
            </button>
          )}
        </div>
      </div>
      {hint && <p className="px-3 pt-2 font-mono text-[10px] text-outline">{hint}</p>}
      <div className="overflow-x-auto">{children}</div>
    </section>
  );
}

function Table({ head, children }) {
  return (
    <table className="w-full font-mono text-[11px] text-on-surface">
      <thead>
        <tr className="text-outline font-ui text-[9px] uppercase tracking-widest">
          {head.map((h) => <th key={h} className="px-2 py-1.5 text-left font-normal whitespace-nowrap">{h}</th>)}
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}

function Empty({ text }) {
  return <p className="px-3 py-4 font-mono text-[11px] text-outline">{text}</p>;
}

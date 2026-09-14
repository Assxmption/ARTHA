import { useState, useEffect, useRef } from 'react';
import {
  Activity, Play, Square, RefreshCw, TrendingUp, TrendingDown,
  Clock, Briefcase, ArrowUpDown, AlertCircle, Loader2, RotateCcw,
} from 'lucide-react';

export default function PaperTrading() {
  const [status, setStatus] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [trades, setTrades] = useState([]);
  const [navHistory, setNavHistory] = useState([]);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const pollingRef = useRef(null);

  // Fetch all data
  const fetchAll = async () => {
    try {
      const [sRes, pRes, tRes, nRes] = await Promise.all([
        fetch('/api/paper-trade/status'),
        fetch('/api/paper-trade/portfolio').catch(() => ({ ok: false })),
        fetch('/api/paper-trade/trades?limit=20').catch(() => ({ ok: false })),
        fetch('/api/paper-trade/nav-history?limit=30').catch(() => ({ ok: false })),
      ]);
      if (sRes.ok) setStatus(await sRes.json());
      if (pRes.ok) setPortfolio(await pRes.json());
      if (tRes.ok) { const d = await tRes.json(); setTrades(d.trades || []); }
      if (nRes.ok) { const d = await nRes.json(); setNavHistory(d.history || []); }
    } catch (err) {
      console.error('Failed to fetch paper trading data:', err);
    }
  };

  // Poll every 15 seconds when running
  useEffect(() => {
    fetchAll();
    pollingRef.current = setInterval(fetchAll, 15000);
    return () => clearInterval(pollingRef.current);
  }, []);

  const startDaemon = async () => {
    setStarting(true);
    try {
      await fetch('/api/paper-trade/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initial_capital: 50000000 }),
      });
      await fetchAll();
    } catch (err) { console.error(err); }
    setStarting(false);
  };

  const stopDaemon = async () => {
    setStopping(true);
    try {
      await fetch('/api/paper-trade/stop', { method: 'POST' });
      await fetchAll();
    } catch (err) { console.error(err); }
    setStopping(false);
  };

  const resetDaemon = async () => {
    if (!window.confirm('Reset will clear ALL positions, trades, and NAV history. Continue?')) return;
    try {
      await fetch('/api/paper-trade/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      });
      await fetchAll();
    } catch (err) { console.error(err); }
  };

  const fmt = (val) => `₹${Number(val || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  const fmtPct = (val) => `${Number(val || 0).toFixed(2)}%`;
  const isPositive = (val) => (val || 0) >= 0;
  const isRunning = status?.state === 'RUNNING' || status?.state === 'SLEEPING';

  const regimeColor = {
    BULL: 'text-emerald-400',
    BEAR: 'text-rose-400',
    SIDEWAYS: 'text-amber-400',
    UNKNOWN: 'text-outline',
  };

  const marketStatusColor = {
    OPEN: 'bg-emerald-500',
    PRE_MARKET: 'bg-amber-500',
    CLOSED: 'bg-outline-variant',
    'CLOSED (Weekend)': 'bg-outline-variant',
  };

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-4">
        <div>
          <h1 className="font-display text-3xl text-on-surface">Live Paper Trading</h1>
          <p className="text-outline text-sm mt-1 font-mono">
            Virtual execution engine — no real orders placed
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {/* Market Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-surface rounded border border-outline-variant">
            <span className={`w-2 h-2 rounded-full animate-pulse ${marketStatusColor[status?.market_status] || 'bg-outline-variant'}`} />
            <span className="text-xs font-mono text-outline">{status?.market_status || 'UNKNOWN'}</span>
          </div>

          {/* Start/Stop Button */}
          {!isRunning ? (
            <button
              id="start-daemon-btn"
              onClick={startDaemon}
              disabled={starting}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-ui text-xs uppercase tracking-widest rounded-sm transition-colors disabled:opacity-50"
            >
              {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              Start Daemon
            </button>
          ) : (
            <button
              id="stop-daemon-btn"
              onClick={stopDaemon}
              disabled={stopping}
              className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white font-ui text-xs uppercase tracking-widest rounded-sm transition-colors disabled:opacity-50"
            >
              {stopping ? <Loader2 size={14} className="animate-spin" /> : <Square size={14} />}
              Stop Daemon
            </button>
          )}
          <button
            id="reset-daemon-btn"
            onClick={resetDaemon}
            className="p-2 text-outline hover:text-on-surface transition-colors"
            title="Reset paper trading"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* Metrics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        <MetricCard
          label="Virtual NAV"
          value={fmt(status?.virtual_nav)}
          icon={<Briefcase size={14} />}
        />
        <MetricCard
          label="Today's PnL"
          value={fmt(status?.today_pnl)}
          positive={isPositive(status?.today_pnl)}
          showArrow
          icon={<ArrowUpDown size={14} />}
        />
        <MetricCard
          label="Total PnL"
          value={`${fmt(status?.total_pnl)} (${fmtPct(status?.total_pnl_pct)})`}
          positive={isPositive(status?.total_pnl)}
          showArrow
          icon={<TrendingUp size={14} />}
        />
        <MetricCard
          label="Regime"
          value={status?.current_regime || 'UNKNOWN'}
          className={regimeColor[status?.current_regime] || regimeColor.UNKNOWN}
          icon={<Activity size={14} />}
        />
        <MetricCard
          label="Hedge Ratio"
          value={`${((status?.hedge_ratio || 0) * 100).toFixed(0)}%`}
          icon={<AlertCircle size={14} />}
        />
      </div>

      {/* Two-column layout: Portfolio + Trades */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Current Portfolio */}
        <div className="bg-surface border border-outline-variant rounded-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-outline-variant flex items-center justify-between">
            <h2 className="font-ui text-xs uppercase tracking-widest text-outline">
              Current Portfolio ({status?.n_positions || 0} positions)
            </h2>
            {portfolio && (
              <span className="font-mono text-xs text-outline">
                Cash: {fmt(portfolio.cash)}
              </span>
            )}
          </div>
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-surface-dim">
                <tr className="text-outline font-mono uppercase tracking-wider border-b border-outline-variant">
                  <th className="text-left px-3 py-2">Symbol</th>
                  <th className="text-right px-3 py-2">Qty</th>
                  <th className="text-right px-3 py-2">Entry</th>
                  <th className="text-right px-3 py-2">Current</th>
                  <th className="text-right px-3 py-2">P&L</th>
                </tr>
              </thead>
              <tbody>
                {portfolio?.positions?.length > 0 ? (
                  portfolio.positions.map((pos, i) => (
                    <tr key={pos.symbol} className={`border-b border-outline-variant/30 ${i % 2 === 0 ? 'bg-surface' : 'bg-surface-dim/50'}`}>
                      <td className="px-3 py-2 font-mono text-on-surface font-medium">{pos.symbol}</td>
                      <td className="px-3 py-2 text-right font-mono text-outline">{pos.quantity}</td>
                      <td className="px-3 py-2 text-right font-mono text-outline">{fmt(pos.avg_entry_price)}</td>
                      <td className="px-3 py-2 text-right font-mono text-outline">{fmt(pos.current_price)}</td>
                      <td className={`px-3 py-2 text-right font-mono font-medium ${isPositive(pos.unrealized_pnl) ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {fmt(pos.unrealized_pnl)} ({fmtPct(pos.unrealized_pnl_pct)})
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-outline italic">
                      No positions yet — start the daemon to begin paper trading
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent Trades */}
        <div className="bg-surface border border-outline-variant rounded-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-outline-variant">
            <h2 className="font-ui text-xs uppercase tracking-widest text-outline">
              Recent Trades ({status?.n_trades_today || 0} today)
            </h2>
          </div>
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-surface-dim">
                <tr className="text-outline font-mono uppercase tracking-wider border-b border-outline-variant">
                  <th className="text-left px-3 py-2">Time</th>
                  <th className="text-left px-3 py-2">Side</th>
                  <th className="text-left px-3 py-2">Symbol</th>
                  <th className="text-right px-3 py-2">Qty</th>
                  <th className="text-right px-3 py-2">Price</th>
                </tr>
              </thead>
              <tbody>
                {trades.length > 0 ? (
                  trades.map((t, i) => (
                    <tr key={t.id || i} className={`border-b border-outline-variant/30 ${i % 2 === 0 ? 'bg-surface' : 'bg-surface-dim/50'}`}>
                      <td className="px-3 py-2 font-mono text-outline">
                        {t.timestamp ? new Date(t.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '—'}
                      </td>
                      <td className={`px-3 py-2 font-mono font-medium ${t.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {t.side}
                      </td>
                      <td className="px-3 py-2 font-mono text-on-surface">{t.symbol}</td>
                      <td className="px-3 py-2 text-right font-mono text-outline">{t.quantity}</td>
                      <td className="px-3 py-2 text-right font-mono text-outline">{fmt(t.fill_price)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-outline italic">
                      No trades yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Daemon Info Footer */}
      {status && (
        <div className="mt-4 flex flex-wrap gap-4 text-[10px] font-mono text-outline-variant/60">
          <span>State: {status.state}</span>
          {status.started_at && <span>Started: {new Date(status.started_at).toLocaleString('en-IN')}</span>}
          {status.last_signal_time && <span>Last Signal: {new Date(status.last_signal_time).toLocaleString('en-IN')}</span>}
          {status.last_error && <span className="text-rose-400">Error: {status.last_error}</span>}
        </div>
      )}
    </div>
  );
}

// Reusable Metric Card
function MetricCard({ label, value, positive, showArrow, icon, className }) {
  return (
    <div className="bg-surface border border-outline-variant rounded-sm px-4 py-3">
      <div className="flex items-center gap-1.5 text-outline text-[10px] font-ui uppercase tracking-widest mb-1">
        {icon}
        {label}
      </div>
      <div className={`font-mono text-lg ${
        className || (showArrow
          ? (positive ? 'text-emerald-400' : 'text-rose-400')
          : 'text-on-surface')
      }`}>
        {showArrow && (positive ? <TrendingUp size={12} className="inline mr-1" /> : <TrendingDown size={12} className="inline mr-1" />)}
        {value}
      </div>
    </div>
  );
}

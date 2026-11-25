import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getActiveSymbol } from '../lib/activeSymbol';
import { usePreferences } from '../lib/preferences';
import { TrendingUp, TrendingDown, Activity, BarChart2, Newspaper, Zap, ArrowUpDown } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Gainers, losers, most active and breadth all come from /api/quant/movers:
// one batched Kite quote across the whole ARTHA universe (Yahoo daily bars,
// labelled delayed, when Kite is not connected). Nothing is ranked client-side
// from a hand-picked list any more.

const num = (v, d = 2) =>
  typeof v === 'number' && Number.isFinite(v)
    ? v.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })
    : '—';
const pct = (v) => (typeof v === 'number' && Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—');
const tone = (v) => (typeof v !== 'number' ? 'text-outline' : v > 0 ? 'text-secondary' : v < 0 ? 'text-error' : 'text-on-surface-variant');
function inr(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  if (v >= 1e7) return `₹${(v / 1e7).toLocaleString('en-IN', { maximumFractionDigits: 0 })} Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(1)} L`;
  return `₹${v.toLocaleString('en-IN')}`;
}
function qty(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  return v.toLocaleString('en-IN');
}

const REGIME_TONE = {
  BULL: ['text-secondary', 'bg-secondary'],
  BEAR: ['text-error', 'bg-error'],
  SIDEWAYS: ['text-primary', 'bg-primary'],
};

const COLUMNS = [
  { id: 'symbol', label: 'Symbol', align: 'left' },
  { id: 'ltp', label: 'Price' },
  { id: 'change_pct', label: 'Change' },
  { id: 'range', label: 'Day range', sortable: false },
  { id: 'volume', label: 'Volume' },
  { id: 'traded_value', label: 'Traded value' },
];

export default function Home() {
  const navigate = useNavigate();
  const { moversCount } = usePreferences();
  const [movers, setMovers] = useState(null);
  const [regime, setRegime] = useState(null);
  const [error, setError] = useState(null);
  const [sort, setSort] = useState({ key: 'change_pct', dir: -1 });

  useEffect(() => {
    let stop = false;
    let timer;
    const load = async () => {
      try {
        const r = await apiFetch(`/api/quant/movers?top=${moversCount}`, { cache: 'no-store' });
        const body = await r.json();
        if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
        if (stop) return;
        setMovers(body);
        setError(null);
        timer = setTimeout(load, body.market_open ? (body.live ? 15000 : 60000) : 300000);
      } catch (e) {
        if (stop) return;
        setError(e.message || String(e));
        timer = setTimeout(load, 60000);
      }
    };
    load();
    apiFetch('/api/quant/regime').then((r) => (r.ok ? r.json() : null)).then((r) => { if (!stop) setRegime(r); }).catch(() => {});
    return () => { stop = true; clearTimeout(timer); };
  }, [moversCount]);

  const rows = useMemo(() => {
    const list = [...(movers?.rows ?? [])];
    const { key, dir } = sort;
    return list.sort((a, b) => {
      if (key === 'symbol') return dir * a.symbol.localeCompare(b.symbol);
      const av = a[key], bv = b[key];
      if (av == null && bv == null) return a.symbol.localeCompare(b.symbol);
      if (av == null) return 1; // unknowns always last
      if (bv == null) return -1;
      return dir * (av - bv) || a.symbol.localeCompare(b.symbol);
    });
  }, [movers, sort]);

  if (!movers && !error) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
          <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Loading market data...</p>
      </div>
    );
  }

  const b = movers?.breadth;
  const total = b ? b.advances + b.declines + b.unchanged : 0;
  const [regimeText, regimeDot] = REGIME_TONE[regime?.current_regime] ?? ['text-on-surface', 'bg-outline'];
  const activeByValue = movers?.active_metric === 'traded_value';

  const MoverList = ({ title, icon, items, metric }) => (
    <div className="border border-outline-variant bg-surface p-4 min-w-0">
      <div className="flex items-center gap-2 mb-3 border-b border-outline-variant pb-2">
        {icon}
        <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">{title}</h2>
      </div>
      {items?.length ? (
        <div className="flex flex-col">
          {items.map((s) => (
            <button
              key={s.symbol}
              onClick={() => navigate(`/dashboard/${s.symbol}`)}
              className="grid grid-cols-[1fr_auto_auto] gap-3 items-center px-2 py-2 hover:bg-surface-container transition-colors text-left group"
            >
              <span className="font-mono text-sm text-on-surface group-hover:text-primary transition-colors truncate">{s.symbol}</span>
              <span className="font-mono text-xs text-on-surface-variant tabular-nums">₹{num(s.ltp)}</span>
              <span className={`font-mono text-xs tabular-nums text-right w-20 ${metric ? 'text-primary' : tone(s.change_pct)}`}>
                {metric ? metric(s) : pct(s.change_pct)}
              </span>
            </button>
          ))}
        </div>
      ) : (
        <p className="font-mono text-[10px] text-outline px-2 py-2">None in this session.</p>
      )}
    </div>
  );

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {error && (
        <div className="mb-4 p-3 border border-error bg-error/10 text-error font-mono text-xs">
          Market data unavailable: {error}{movers ? ' — showing the last successful update.' : ''}
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between lg:items-end gap-4 mb-6">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Market Overview</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">
            {movers?.live ? 'Live · Kite Connect' : 'Delayed · Yahoo daily bars'}
            {movers && ` · ${movers.market_open ? 'Market open' : 'Market closed'} · updated ${movers.as_of?.slice(11, 19)} IST`}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {b && (
            <div className="border border-outline-variant bg-surface px-4 py-2 min-w-[14rem]" title="Advances / declines across the ARTHA universe">
              <span className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-1">Breadth</span>
              <div className="flex h-1.5 w-full overflow-hidden bg-surface-container mb-1" aria-hidden="true">
                <div className="bg-secondary" style={{ width: `${total ? (b.advances / total) * 100 : 0}%` }} />
                <div className="bg-outline" style={{ width: `${total ? (b.unchanged / total) * 100 : 0}%` }} />
                <div className="bg-error" style={{ width: `${total ? (b.declines / total) * 100 : 0}%` }} />
              </div>
              <span className="font-mono text-xs">
                <span className="text-secondary">{b.advances} up</span>
                <span className="text-outline"> · {b.unchanged} flat · </span>
                <span className="text-error">{b.declines} down</span>
              </span>
            </div>
          )}
          {regime && (
            <div className="flex items-center gap-3 border border-outline-variant bg-surface px-4 py-2" title={regime.as_of ? `HMM on NIFTY 50 · as of ${regime.as_of}` : ''}>
              <span className={`w-2 h-2 rounded-full ${regimeDot}`}></span>
              <div>
                <span className="font-ui text-[10px] uppercase tracking-widest text-outline block">Market Regime</span>
                <span className={`font-mono text-sm font-bold ${regimeText}`}>
                  {regime.current_regime}
                  {regime.sessions_in_state != null && <span className="font-normal text-[10px] text-outline ml-2">{regime.sessions_in_state} sessions</span>}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <MoverList title="Top Gainers" icon={<TrendingUp size={16} className="text-secondary" />} items={movers?.gainers} />
        <MoverList title="Top Losers" icon={<TrendingDown size={16} className="text-error" />} items={movers?.losers} />
        <MoverList
          title={activeByValue ? 'Most Active · value' : 'Most Active · volume'}
          icon={<Activity size={16} className="text-primary" />}
          items={movers?.most_active}
          metric={(s) => (activeByValue ? inr(s.traded_value) : qty(s.volume))}
        />

        {/* Quick Actions */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-3 border-b border-outline-variant pb-2">
            <Zap size={16} className="text-primary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Quick Actions</h2>
          </div>
          <div className="flex flex-col gap-2">
            {[
              { to: '/signals', icon: <Activity size={16} className="text-primary" />, title: 'Quant Signals', sub: 'HMM regime + pair signals' },
              { to: '/simulator', icon: <BarChart2 size={16} className="text-primary" />, title: 'Backtest Lab', sub: 'Historical replay, no live orders' },
              { to: `/news/${getActiveSymbol()}`, icon: <Newspaper size={16} className="text-primary" />, title: 'Sentiment', sub: 'VADER-scored market news' },
            ].map((a) => (
              <button key={a.title} onClick={() => navigate(a.to)}
                className="flex items-center gap-3 p-3 border border-outline-variant/50 bg-surface-container-low hover:border-primary hover:bg-primary/5 transition-colors text-left group">
                {a.icon}
                <div>
                  <span className="font-ui text-xs text-on-surface group-hover:text-primary transition-colors block">{a.title}</span>
                  <span className="font-mono text-[10px] text-outline">{a.sub}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Full universe */}
      <div className="mt-6 border border-outline-variant bg-surface overflow-hidden">
        <div className="border-b border-outline-variant bg-surface-container-low p-3 flex flex-wrap gap-2 justify-between items-center">
          <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">
            All instruments · {movers?.universe?.size ?? '—'} names
          </span>
          <span className="font-mono text-[10px] text-outline" title={movers?.universe?.note}>
            {movers?.universe?.name}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-outline-variant bg-surface-container-lowest text-outline">
                {COLUMNS.map((c) => (
                  <th key={c.id} className={`font-ui text-[10px] uppercase p-3 font-normal tracking-wider ${c.align === 'left' ? '' : 'text-right'}`}>
                    {c.sortable === false ? c.label : (
                      <button
                        onClick={() => setSort((s) => ({ key: c.id, dir: s.key === c.id ? -s.dir : c.id === 'symbol' ? 1 : -1 }))}
                        className={`inline-flex items-center gap-1 uppercase hover:text-on-surface ${sort.key === c.id ? 'text-primary' : ''}`}
                        aria-label={`Sort by ${c.label}`}
                      >
                        {c.label}<ArrowUpDown size={10} />
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((s, i) => (
                <tr key={s.symbol}
                  className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors cursor-pointer ${i % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}
                  onClick={() => navigate(`/dashboard/${s.symbol}`)}>
                  <td className="p-3 text-on-surface font-bold">{s.symbol}</td>
                  <td className="p-3 text-right text-on-surface tabular-nums">₹{num(s.ltp)}</td>
                  <td className={`p-3 text-right tabular-nums ${tone(s.change_pct)}`}>
                    {s.change == null ? '—' : `${s.change >= 0 ? '+' : ''}${num(s.change)}`} <span className="opacity-80">({pct(s.change_pct)})</span>
                  </td>
                  <td className="p-3 text-right text-on-surface-variant tabular-nums whitespace-nowrap">{num(s.low)} – {num(s.high)}</td>
                  <td className="p-3 text-right text-on-surface-variant tabular-nums">{qty(s.volume)}</td>
                  <td className="p-3 text-right text-on-surface-variant tabular-nums">{inr(s.traded_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {(movers?.unquoted?.length > 0) && (
          <p className="p-3 font-mono text-[10px] text-outline border-t border-outline-variant/50">
            No quote for {movers.unquoted.join(', ')}
            {movers.stale?.length ? ` (stale on Yahoo: ${movers.stale.join(', ')})` : ''} — left out of rankings and breadth.
          </p>
        )}
        <p className="p-3 font-mono text-[10px] text-outline-variant border-t border-outline-variant/50">
          Source: {movers?.source ?? '—'}. Market data for information only; not investment advice.
        </p>
      </div>
    </div>
  );
}

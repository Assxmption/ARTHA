import { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { History } from 'lucide-react';
import { apiFetch } from '../lib/api';

const pct = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%` : '—');

function toSeries(points, session) {
  return (points || []).map((p) => ({
    time: Math.floor(new Date(`${session}T${p.time}:00Z`).getTime() / 1000), // IST wall clock rendered as-is
    value: p.value * 100,
  }));
}

export default function SessionReplay() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const chartEl = useRef(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/quant/session-replay')
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
        return body;
      })
      .then((b) => { if (!cancelled) setData(b); })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!data || !chartEl.current) return;
    const chart = createChart(chartEl.current, {
      height: 140,
      width: chartEl.current.clientWidth,
      layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#cfc8bf', fontSize: 10 },
      grid: { vertLines: { visible: false }, horzLines: { color: '#3a3630' } },
      rightPriceScale: { borderColor: '#3a3630' },
      timeScale: { borderColor: '#3a3630', timeVisible: true, secondsVisible: false },
      localization: { priceFormatter: (v) => `${v.toFixed(2)}%` },
    });
    chart.addLineSeries({ color: '#a2d0c0', lineWidth: 2, title: 'Book' }).setData(toSeries(data.curves.book, data.session));
    if (data.curves.nifty?.length) {
      chart.addLineSeries({ color: '#e8c086', lineWidth: 1, lineStyle: 2, title: 'NIFTY' }).setData(toSeries(data.curves.nifty, data.session));
    }
    chart.timeScale().fitContent();
    const ro = new ResizeObserver(() => chartEl.current && chart.applyOptions({ width: chartEl.current.clientWidth }));
    ro.observe(chartEl.current);
    return () => { ro.disconnect(); chart.remove(); };
  }, [data]);

  return (
    <div className="border border-outline-variant bg-surface">
      <div className="border-b border-outline-variant p-3 bg-surface-container-low flex justify-between items-center">
        <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">
          Session Replay {data ? `· ${data.session}` : ''}
        </span>
        <History size={14} className="text-primary" />
      </div>
      {loading && <p className="p-3 font-mono text-[10px] text-outline animate-pulse">Replaying the latest session (first run ≈ 40s)…</p>}
      {error && <p className="p-3 font-mono text-[10px] text-error">Replay unavailable: {error}</p>}
      {data && (
        <div className="font-mono text-xs">
          <p className="px-3 pt-2 text-[10px] text-outline">
            Book fixed at {data.previous_session} close · {data.book.symbols} names · net {(data.book.net_exposure * 100).toFixed(0)}% · gross {(data.book.gross_exposure * 100).toFixed(0)}%
          </p>
          <div className="grid grid-cols-3 gap-2 p-3">
            {[
              ['Book', data.result.book_return_eod],
              ['NIFTY 50', data.result.nifty_return_eod],
              ['Difference', data.result.excess_vs_nifty],
            ].map(([label, v]) => (
              <div key={label} className="border-l-2 border-outline-variant pl-2">
                <p className="text-[10px] uppercase text-outline">{label}</p>
                <p className={v >= 0 ? 'text-secondary' : 'text-error'}>{pct(v)}</p>
              </div>
            ))}
          </div>
          <div ref={chartEl} className="w-full px-1" />
          <div className="grid grid-cols-2 gap-2 p-3 text-[10px]">
            <div>
              <p className="uppercase text-outline mb-1">Top contributors</p>
              {data.result.top_contributors.slice(0, 3).map((c) => (
                <p key={c.symbol} className="flex justify-between"><span>{c.symbol}</span><span className="text-secondary">{pct(c.contribution, 3)}</span></p>
              ))}
            </div>
            <div>
              <p className="uppercase text-outline mb-1">Bottom contributors</p>
              {data.result.bottom_contributors.slice(0, 3).map((c) => (
                <p key={c.symbol} className="flex justify-between"><span>{c.symbol}</span><span className="text-error">{pct(c.contribution, 3)}</span></p>
              ))}
            </div>
          </div>
          {Object.keys(data.result.pair_returns_eod || {}).length > 0 && (
            <p className="px-3 pb-2 text-[10px] text-on-surface-variant">
              Open validated pairs: {Object.entries(data.result.pair_returns_eod).map(([k, v]) => `${k} ${pct(v)}`).join(' · ')}
            </p>
          )}
          <p className="px-3 pb-1 text-[10px] text-outline">Data: {data.sources.minute}</p>
          <ul className="px-3 pb-3 text-[10px] text-outline-variant list-disc list-inside">
            {data.caveats.map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

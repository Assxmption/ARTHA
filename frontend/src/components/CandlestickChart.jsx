import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { Activity, BarChart2, AreaChart, CandlestickChart as HAIcon, Radio } from 'lucide-react';
import { getPreferences } from '../lib/preferences';
import { apiFetch } from '../lib/api';

// Indicator values come from the backend (app/quant/indicators.py, unit-tested);
// this component only draws them. Heikin-Ashi is a display transform.

export const INTERVALS = [
  { id: '1m', label: '1m', ranges: [1, 2, 5] },
  { id: '3m', label: '3m', ranges: [1, 2, 5] },
  { id: '5m', label: '5m', ranges: [1, 5, 10] },
  { id: '15m', label: '15m', ranges: [5, 10, 30] },
  { id: '30m', label: '30m', ranges: [10, 30, 60] },
  { id: '60m', label: '1h', ranges: [30, 90, 180] },
  { id: '1d', label: '1D', ranges: [90, 180, 365, 1095] },
  { id: '1wk', label: '1W', ranges: [730, 1825, 3650] },
  { id: '1mo', label: '1M', ranges: [1825, 3650] },
];
const INTRADAY = new Set(['1m', '3m', '5m', '15m', '30m', '60m']);
const RANGE_LABEL = (d) => (d < 30 ? `${d}D` : d < 365 ? `${Math.round(d / 30)}M` : `${Math.round(d / 365)}Y`);

export const OVERLAYS = [
  { id: 'sma20', label: 'SMA 20', color: '#4f8cff' },
  { id: 'sma50', label: 'SMA 50', color: '#f5a524' },
  { id: 'sma200', label: 'SMA 200', color: '#c084fc' },
  { id: 'ema9', label: 'EMA 9', color: '#22d3ee' },
  { id: 'ema21', label: 'EMA 21', color: '#f472b6' },
  { id: 'bb', label: 'BB 20,2', color: '#94a3b8' },
  { id: 'vwap', label: 'VWAP', color: '#facc15', intradayOnly: true },
];
export const PANES = [
  { id: 'rsi', label: 'RSI 14' },
  { id: 'macd', label: 'MACD' },
  { id: 'atr', label: 'ATR 14' },
];

const UP = '#a2d0c0';
const DOWN = '#ffb4ab';
const GRID = '#3a3630';
const TEXT = '#e6e1e5';

const fmt = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
const fmtVol = (v) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  return v.toLocaleString('en-IN');
};

function heikinAshi(bars) {
  const out = [];
  bars.forEach((b, i) => {
    const close = (b.open + b.high + b.low + b.close) / 4;
    const open = i === 0 ? (b.open + b.close) / 2 : (out[i - 1].open + out[i - 1].close) / 2;
    out.push({ time: b.time, open, close, high: Math.max(b.high, open, close), low: Math.min(b.low, open, close) });
  });
  return out;
}

const baseOptions = (height) => ({
  height,
  layout: { background: { type: 'solid', color: 'transparent' }, textColor: TEXT, fontSize: 11 },
  grid: { vertLines: { color: GRID }, horzLines: { color: GRID } },
  rightPriceScale: { borderColor: GRID },
  timeScale: { borderColor: GRID, timeVisible: true, secondsVisible: false, rightOffset: 4 },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: { color: '#e8c086', labelBackgroundColor: '#8a6d3b' },
    horzLine: { color: '#e8c086', labelBackgroundColor: '#8a6d3b' },
  },
});

function line(values, times) {
  const out = [];
  for (let i = 0; i < times.length; i++) if (values?.[i] != null) out.push({ time: times[i], value: values[i] });
  return out;
}

export default function CandlestickChart({ symbol }) {
  // Initial view comes from Settings → Chart defaults; changes here are per-view.
  const [initial] = useState(() => {
    const p = getPreferences();
    const iv = INTERVALS.find((i) => i.id === p.chartInterval) ?? INTERVALS.find((i) => i.id === '1d');
    return {
      interval: iv.id,
      days: iv.ranges.includes(p.chartDays) ? p.chartDays : iv.ranges[Math.min(1, iv.ranges.length - 1)],
      chartType: p.chartType,
      overlays: p.chartOverlays.filter((id) => OVERLAYS.some((o) => o.id === id)),
      panes: p.chartPanes.filter((id) => PANES.some((o) => o.id === id)),
      volume: p.chartVolume,
      liveTick: p.chartLiveTick,
    };
  });
  const mainRef = useRef(null);
  const paneRefs = useRef({});
  const charts = useRef([]);
  const mainSeries = useRef(null);
  const volumeSeries = useRef(null);
  const lastBar = useRef(null);

  const [interval, setInterval_] = useState(initial.interval);
  const [days, setDays] = useState(initial.days);
  const [chartType, setChartType] = useState(initial.chartType);
  const [overlays, setOverlays] = useState(() => new Set(initial.overlays));
  const [panes, setPanes] = useState(() => new Set(initial.panes));
  const [showVolume, setShowVolume] = useState(initial.volume);
  const [payload, setPayload] = useState(null);
  const [quote, setQuote] = useState(null);
  const [legend, setLegend] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDepth, setShowDepth] = useState(false);

  const intraday = INTRADAY.has(interval);
  const ranges = INTERVALS.find((i) => i.id === interval)?.ranges ?? [180];

  const requested = useMemo(() => {
    const list = [...overlays].filter((id) => intraday || !OVERLAYS.find((o) => o.id === id)?.intradayOnly);
    return [...list, ...panes];
  }, [overlays, panes, intraday]);

  const load = useCallback(async (signal) => {
    const qs = new URLSearchParams({ interval, days: String(days), indicators: requested.join(',') });
    const res = await apiFetch(`/api/quant/ohlcv/${symbol}?${qs}`, { signal });
    const body = await res.json();
    if (!res.ok) throw new Error(body?.detail || `HTTP ${res.status}`);
    return body;
  }, [symbol, interval, days, requested]);

  // Bars + indicators
  useEffect(() => {
    if (!symbol) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    load(ctrl.signal)
      .then(setPayload)
      .catch((e) => { if (e.name !== 'AbortError') setError(e.message || 'Failed to fetch chart data'); })
      .finally(() => setLoading(false));
    return () => ctrl.abort();
  }, [symbol, load]);

  // Quote header (+ live tick into the last candle while the market is open)
  useEffect(() => {
    if (!symbol) return;
    let stop = false;
    let timer;
    const tick = async () => {
      try {
        const res = await apiFetch(`/api/quant/quote/${symbol}`);
        if (res.ok) {
          const q = await res.json();
          if (stop) return;
          setQuote(q);
          const lb = lastBar.current;
          if (initial.liveTick && q.market_open && q.live && lb && mainSeries.current && typeof q.ltp === 'number' && chartType !== 'ha') {
            const updated = { ...lb, close: q.ltp, high: Math.max(lb.high, q.ltp), low: Math.min(lb.low, q.ltp) };
            lastBar.current = updated;
            mainSeries.current.update(chartType === 'candle' ? updated : { time: updated.time, value: updated.close });
          }
          timer = setTimeout(tick, q.market_open ? (q.live ? 2000 : 15000) : 60000);
          return;
        }
      } catch { /* network blip: retry below */ }
      if (!stop) timer = setTimeout(tick, 15000);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [symbol, chartType]);

  // Refresh completed candles periodically during the session
  useEffect(() => {
    if (!quote?.market_open) return;
    const id = setInterval(() => {
      load().then(setPayload).catch(() => {});
    }, intraday ? 30000 : 120000);
    return () => clearInterval(id);
  }, [quote?.market_open, intraday, load]);

  // Draw
  useEffect(() => {
    charts.current.forEach((c) => c.remove());
    charts.current = [];
    mainSeries.current = null;
    if (!payload?.data?.length || !mainRef.current) return;

    const bars = payload.data;
    const times = payload.indicators?.times ?? bars.map((b) => b.time);
    const ind = payload.indicators ?? {};

    const main = createChart(mainRef.current, { ...baseOptions(mainRef.current.clientHeight || 360), width: mainRef.current.clientWidth });
    charts.current.push(main);

    let series;
    if (chartType === 'line') {
      series = main.addLineSeries({ color: UP, lineWidth: 2 });
      series.setData(bars.map((b) => ({ time: b.time, value: b.close })));
    } else if (chartType === 'area') {
      series = main.addAreaSeries({ lineColor: UP, topColor: 'rgba(162,208,192,0.35)', bottomColor: 'rgba(162,208,192,0.02)', lineWidth: 2 });
      series.setData(bars.map((b) => ({ time: b.time, value: b.close })));
    } else {
      series = main.addCandlestickSeries({ upColor: UP, downColor: DOWN, borderVisible: false, wickUpColor: UP, wickDownColor: DOWN });
      series.setData(chartType === 'ha' ? heikinAshi(bars) : bars);
    }
    mainSeries.current = series;
    lastBar.current = { ...bars[bars.length - 1] };

    if (showVolume && bars.some((b) => b.volume > 0)) {
      const vol = main.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false });
      main.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      vol.setData(bars.map((b) => ({ time: b.time, value: b.volume, color: b.close >= b.open ? 'rgba(162,208,192,0.35)' : 'rgba(255,180,171,0.35)' })));
      volumeSeries.current = vol;
    }
    main.priceScale('right').applyOptions({ scaleMargins: { top: 0.08, bottom: showVolume ? 0.22 : 0.06 } });

    const overlaySeries = {};
    OVERLAYS.forEach((o) => {
      if (!overlays.has(o.id)) return;
      if (o.id === 'bb') {
        ['bb_upper', 'bb_mid', 'bb_lower'].forEach((k) => {
          if (!ind[k]) return;
          const s = main.addLineSeries({ color: o.color, lineWidth: 1, lineStyle: k === 'bb_mid' ? 2 : 0, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
          s.setData(line(ind[k], times));
          overlaySeries[k] = s;
        });
      } else if (ind[o.id]) {
        const s = main.addLineSeries({ color: o.color, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        s.setData(line(ind[o.id], times));
        overlaySeries[o.id] = s;
      }
    });

    // Sub-panes, time-synced with the main chart
    const subCharts = [];
    PANES.forEach((p) => {
      const el = paneRefs.current[p.id];
      if (!panes.has(p.id) || !el) return;
      const c = createChart(el, { ...baseOptions(110), width: el.clientWidth, timeScale: { ...baseOptions(110).timeScale, visible: false } });
      if (p.id === 'rsi' && ind.rsi) {
        const s = c.addLineSeries({ color: '#c084fc', lineWidth: 1.5, priceLineVisible: false });
        s.setData(line(ind.rsi, times));
        [70, 30].forEach((lvl) => s.createPriceLine({ price: lvl, color: '#6b6560', lineWidth: 1, lineStyle: 2, axisLabelVisible: true }));
      }
      if (p.id === 'macd' && ind.macd) {
        const h = c.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
        h.setData(times.map((t, i) => (ind.macd_hist[i] == null ? null : { time: t, value: ind.macd_hist[i], color: ind.macd_hist[i] >= 0 ? 'rgba(162,208,192,0.5)' : 'rgba(255,180,171,0.5)' })).filter(Boolean));
        c.addLineSeries({ color: '#4f8cff', lineWidth: 1.5, priceLineVisible: false }).setData(line(ind.macd, times));
        c.addLineSeries({ color: '#f5a524', lineWidth: 1.5, priceLineVisible: false }).setData(line(ind.macd_signal, times));
      }
      if (p.id === 'atr' && ind.atr) {
        c.addLineSeries({ color: '#22d3ee', lineWidth: 1.5, priceLineVisible: false }).setData(line(ind.atr, times));
      }
      subCharts.push(c);
      charts.current.push(c);
    });

    let syncing = false;
    const all = [main, ...subCharts];
    all.forEach((src) => src.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (syncing || !range) return;
      syncing = true;
      all.forEach((dst) => { if (dst !== src) dst.timeScale().setVisibleLogicalRange(range); });
      syncing = false;
    }));

    const byTime = new Map(bars.map((b, i) => [b.time, i]));
    main.subscribeCrosshairMove((param) => {
      const i = param?.time != null ? byTime.get(param.time) : undefined;
      if (i === undefined) { setLegend(null); return; }
      const b = bars[i];
      const prev = i > 0 ? bars[i - 1].close : b.open;
      const values = {};
      Object.keys(overlaySeries).forEach((k) => { values[k] = ind[k]?.[i]; });
      ['rsi', 'macd', 'macd_signal', 'atr'].forEach((k) => { if (ind[k]) values[k] = ind[k][i]; });
      setLegend({ ...b, chg: b.close - prev, chgPct: prev ? (b.close / prev - 1) * 100 : null, values });
    });

    main.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      all.forEach((c, idx) => {
        const el = idx === 0 ? mainRef.current : paneRefs.current[[...panes][idx - 1]];
        if (el) c.applyOptions({ width: el.clientWidth, ...(idx === 0 ? { height: el.clientHeight } : {}) });
      });
    });
    if (mainRef.current) ro.observe(mainRef.current);
    return () => ro.disconnect();
  }, [payload, chartType, overlays, panes, showVolume]);

  useEffect(() => () => charts.current.forEach((c) => c.remove()), []);

  const toggle = (set, setter, id) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setter(next);
  };

  const live = payload?.live && quote?.live;
  const up = (quote?.change ?? 0) >= 0;
  const shown = legend ?? (payload?.data?.length ? { ...payload.data[payload.data.length - 1], values: null } : null);

  return (
    <div className="relative border border-outline-variant bg-surface-dim overflow-hidden flex flex-col">
      {/* Quote header */}
      <div className="flex flex-wrap items-end justify-between gap-3 px-3 pt-3 pb-2 border-b border-outline-variant bg-surface-container-low">
        <div className="flex items-end gap-3 flex-wrap">
          <span className="font-display text-xl text-on-surface">{symbol}</span>
          <span className="font-mono text-xl text-on-surface">{fmt(quote?.ltp)}</span>
          <span className={`font-mono text-sm ${up ? 'text-secondary' : 'text-error'}`}>
            {quote?.change != null ? `${up ? '+' : ''}${fmt(quote.change)} (${up ? '+' : ''}${fmt(quote.change_pct)}%)` : ''}
          </span>
          <span className="font-mono text-[10px] text-outline">
            O {fmt(quote?.open)} · H {fmt(quote?.high)} · L {fmt(quote?.low)} · Vol {fmtVol(quote?.volume)}
            {quote?.upper_circuit ? ` · Circuit ${fmt(quote.lower_circuit)}–${fmt(quote.upper_circuit)}` : ''}
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span className={`px-1.5 py-0.5 border flex items-center gap-1 ${live ? 'border-secondary text-secondary' : 'border-outline text-outline'}`}
            title={payload?.source || quote?.source}>
            <Radio size={10} className={live && quote?.market_open ? 'animate-pulse' : ''} />
            {live ? 'KITE · LIVE' : 'DELAYED'}
          </span>
          <span className="px-1.5 py-0.5 border border-outline-variant text-outline">{quote?.market_open ? 'MARKET OPEN' : 'MARKET CLOSED'}</span>
          {quote?.depth && (
            <button onClick={() => setShowDepth(!showDepth)} className={`px-1.5 py-0.5 border ${showDepth ? 'border-primary text-primary' : 'border-outline-variant text-outline hover:text-on-surface'}`}>DEPTH</button>
          )}
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap gap-x-4 gap-y-2 items-center px-3 py-2 border-b border-outline-variant bg-surface-container-low">
        <div className="flex bg-surface-container rounded-sm p-0.5 border border-outline-variant">
          {[['candle', BarChart2, 'Candles'], ['ha', HAIcon, 'Heikin-Ashi'], ['line', Activity, 'Line'], ['area', AreaChart, 'Area']].map(([id, Icon, title]) => (
            <button key={id} onClick={() => setChartType(id)} title={title}
              className={`p-1 rounded-sm transition-colors ${chartType === id ? 'bg-primary text-surface' : 'text-outline hover:text-on-surface'}`}>
              <Icon size={14} />
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1">
          {INTERVALS.map((iv) => (
            <button key={iv.id} onClick={() => { setInterval_(iv.id); setDays(iv.ranges[Math.min(1, iv.ranges.length - 1)]); }}
              className={`font-mono text-[10px] px-2 py-1 border transition-colors ${interval === iv.id ? 'text-primary border-primary bg-primary/5' : 'text-outline border-transparent hover:border-outline-variant'}`}>
              {iv.label}
            </button>
          ))}
        </div>
        <div className="flex gap-1 border-l border-outline-variant pl-3">
          {ranges.map((d) => (
            <button key={d} onClick={() => setDays(d)}
              className={`font-mono text-[10px] px-2 py-1 border ${days === d ? 'text-primary border-primary' : 'text-outline border-transparent hover:border-outline-variant'}`}>
              {RANGE_LABEL(d)}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1 border-l border-outline-variant pl-3">
          <button onClick={() => setShowVolume(!showVolume)}
            className={`font-ui text-[10px] px-2 py-1 border ${showVolume ? 'text-primary border-primary bg-primary/5' : 'text-outline border-transparent hover:text-on-surface'}`}>VOL</button>
          {OVERLAYS.filter((o) => intraday || !o.intradayOnly).map((o) => (
            <button key={o.id} onClick={() => toggle(overlays, setOverlays, o.id)}
              className="font-ui text-[10px] px-2 py-1 border transition-colors"
              style={overlays.has(o.id) ? { color: o.color, borderColor: o.color } : { color: '#8f8a83', borderColor: 'transparent' }}>
              {o.label}
            </button>
          ))}
          {PANES.map((p) => (
            <button key={p.id} onClick={() => toggle(panes, setPanes, p.id)}
              className={`font-ui text-[10px] px-2 py-1 border ${panes.has(p.id) ? 'text-primary border-primary bg-primary/5' : 'text-outline border-transparent hover:text-on-surface'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Crosshair legend */}
      <div className="px-3 py-1 font-mono text-[10px] text-on-surface-variant flex flex-wrap gap-x-3 min-h-[20px] border-b border-outline-variant/40">
        {shown && (
          <>
            <span>O <span className="text-on-surface">{fmt(shown.open)}</span></span>
            <span>H <span className="text-on-surface">{fmt(shown.high)}</span></span>
            <span>L <span className="text-on-surface">{fmt(shown.low)}</span></span>
            <span>C <span className="text-on-surface">{fmt(shown.close)}</span></span>
            {shown.chg != null && <span className={shown.chg >= 0 ? 'text-secondary' : 'text-error'}>{shown.chg >= 0 ? '+' : ''}{fmt(shown.chg)} ({fmt(shown.chgPct)}%)</span>}
            <span>Vol <span className="text-on-surface">{fmtVol(shown.volume)}</span></span>
            {shown.values && Object.entries(shown.values).map(([k, v]) => (
              <span key={k}>{k.toUpperCase()} <span className="text-on-surface">{fmt(v)}</span></span>
            ))}
          </>
        )}
        {loading && <span className="text-primary animate-pulse ml-auto">Fetching…</span>}
        {!loading && payload?.source && <span className="ml-auto text-outline">{payload.source}{payload.last_bar?.provisional ? ' · last bar provisional' : ''}</span>}
      </div>

      <div className="flex">
        <div className="flex-1 min-w-0">
          {error ? (
            <div className="h-[380px] flex items-center justify-center text-error font-mono text-xs">{error}</div>
          ) : (
            <div ref={mainRef} className="w-full h-[380px] relative" />
          )}
          {PANES.filter((p) => panes.has(p.id)).map((p) => (
            <div key={p.id} className="border-t border-outline-variant/50 relative">
              <span className="absolute left-2 top-1 z-10 font-mono text-[9px] text-outline">{p.label}</span>
              <div ref={(el) => { paneRefs.current[p.id] = el; }} className="w-full h-[110px]" />
            </div>
          ))}
        </div>
        {showDepth && quote?.depth && (
          <div className="w-48 border-l border-outline-variant font-mono text-[10px] p-2 hidden sm:block">
            <p className="text-outline uppercase mb-1">Market depth</p>
            <table className="w-full">
              <thead><tr className="text-outline"><th className="text-left font-normal">Bid</th><th className="text-right font-normal">Qty</th></tr></thead>
              <tbody>{quote.depth.buy.map((l, i) => <tr key={`b${i}`}><td className="text-secondary">{fmt(l.price)}</td><td className="text-right">{l.quantity?.toLocaleString('en-IN')}</td></tr>)}</tbody>
            </table>
            <table className="w-full mt-2">
              <thead><tr className="text-outline"><th className="text-left font-normal">Ask</th><th className="text-right font-normal">Qty</th></tr></thead>
              <tbody>{quote.depth.sell.map((l, i) => <tr key={`s${i}`}><td className="text-error">{fmt(l.price)}</td><td className="text-right">{l.quantity?.toLocaleString('en-IN')}</td></tr>)}</tbody>
            </table>
            <p className="text-outline mt-2">Total bid {quote.buy_quantity?.toLocaleString('en-IN')} · ask {quote.sell_quantity?.toLocaleString('en-IN')}</p>
          </div>
        )}
      </div>
    </div>
  );
}

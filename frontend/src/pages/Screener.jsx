import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Filter, Search, TrendingUp, TrendingDown, Plus, X, Play,
  RefreshCw, Loader2, ChevronDown, Bookmark, ArrowUpDown, Database,
} from 'lucide-react';

const OPERATORS = [
  { value: 'gt', label: '>' },
  { value: 'lt', label: '<' },
  { value: 'gte', label: '≥' },
  { value: 'lte', label: '≤' },
  { value: 'eq', label: '=' },
  { value: 'between', label: 'between' },
];

const fmt = (val, unit) => {
  if (val === null || val === undefined) return '—';
  if (unit === 'inr_crore') return `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 0 })} Cr`;
  if (unit === 'inr') return `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
  if (unit === 'percent') return `${Number(val).toFixed(2)}%`;
  if (unit === 'ratio') return Number(val).toFixed(2);
  return String(val);
};

export default function Screener() {
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState([]);
  const [sectors, setSectors] = useState([]);
  const [presets, setPresets] = useState({});
  const [cacheInfo, setCacheInfo] = useState(null);

  // Filter state
  const [filters, setFilters] = useState([]);
  const [sectorFilter, setSectorFilter] = useState('');
  const [sortBy, setSortBy] = useState('market_cap');
  const [sortDesc, setSortDesc] = useState(true);

  // Results
  const [results, setResults] = useState([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [loading, setLoading] = useState(false);
  const [building, setBuilding] = useState(false);
  const [activePreset, setActivePreset] = useState('');

  // Display columns — which metrics to show in the table
  const displayCols = [
    'current_price', 'market_cap', 'pe', 'pb', 'roe',
    'debt_to_equity', 'operating_margin', 'revenue_growth', 'dividend_yield', 'beta',
  ];

  // Load metadata on mount
  useEffect(() => {
    Promise.all([
      fetch('/api/screener/metrics').then(r => r.ok ? r.json() : { metrics: [] }),
      fetch('/api/screener/sectors').then(r => r.ok ? r.json() : { sectors: [] }),
      fetch('/api/screener/presets').then(r => r.ok ? r.json() : { presets: {} }),
      fetch('/api/screener/info').then(r => r.ok ? r.json() : null),
    ]).then(([m, s, p, info]) => {
      setMetrics(m.metrics || []);
      setSectors(s.sectors || []);
      setPresets(p.presets || {});
      setCacheInfo(info);

      // Auto-run if cache exists
      if (info && info.total_symbols > 0) {
        runScreen([], '', 'market_cap', true);
      }
    });
  }, []);

  const filterableMetrics = metrics.filter(m => m.filterable);

  const addFilter = () => {
    setFilters(prev => [
      ...prev,
      { metric: filterableMetrics[0]?.key || 'pe', operator: 'lt', value: '', value2: '' },
    ]);
    setActivePreset('');
  };

  const updateFilter = (idx, field, val) => {
    setFilters(prev => prev.map((f, i) => i === idx ? { ...f, [field]: val } : f));
    setActivePreset('');
  };

  const removeFilter = (idx) => {
    setFilters(prev => prev.filter((_, i) => i !== idx));
    setActivePreset('');
  };

  const applyPreset = (key) => {
    const preset = presets[key];
    if (!preset) return;
    const newFilters = preset.filters.map(f => ({
      metric: f.metric,
      operator: f.operator,
      value: String(f.value),
      value2: '',
    }));
    setFilters(newFilters);
    setSortBy(preset.sort_by || 'market_cap');
    setSortDesc(preset.sort_desc !== false);
    setActivePreset(key);
    runScreen(newFilters, sectorFilter, preset.sort_by || 'market_cap', preset.sort_desc !== false);
  };

  const runScreen = async (overrideFilters, overrideSector, overrideSort, overrideSortDesc) => {
    setLoading(true);
    const f = overrideFilters !== undefined ? overrideFilters : filters;
    const validFilters = f
      .filter(fl => fl.value !== '' && fl.value !== undefined)
      .map(fl => ({
        metric: fl.metric,
        operator: fl.operator,
        value: parseFloat(fl.value),
        value2: fl.operator === 'between' && fl.value2 ? parseFloat(fl.value2) : undefined,
      }));

    try {
      const res = await fetch('/api/screener/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filters: validFilters,
          sort_by: overrideSort ?? sortBy,
          sort_desc: overrideSortDesc ?? sortDesc,
          limit: 100,
          sector: (overrideSector ?? sectorFilter) || undefined,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setResults(data.results || []);
        setTotalMatches(data.total_matches || 0);
        setCacheInfo(prev => ({
          ...prev,
          cache_age_hours: data.cache_age_hours,
          total_symbols: data.cache_total_symbols,
        }));
      }
    } catch (err) {
      console.error('Screen query failed:', err);
    }
    setLoading(false);
  };

  const buildCache = async () => {
    setBuilding(true);
    try {
      await fetch('/api/screener/build', { method: 'POST' });
      // Poll for completion
      const poll = setInterval(async () => {
        const res = await fetch('/api/screener/info');
        if (res.ok) {
          const info = await res.json();
          setCacheInfo(info);
          if (info.total_symbols > 50) {
            clearInterval(poll);
            setBuilding(false);
            runScreen(filters, sectorFilter, sortBy, sortDesc);
          }
        }
      }, 5000);
      // Safety timeout
      setTimeout(() => { clearInterval(poll); setBuilding(false); }, 300000);
    } catch (err) {
      console.error(err);
      setBuilding(false);
    }
  };

  const toggleSort = (col) => {
    if (sortBy === col) {
      setSortDesc(!sortDesc);
    } else {
      setSortBy(col);
      setSortDesc(true);
    }
    runScreen(filters, sectorFilter, sortBy === col ? col : col, sortBy === col ? !sortDesc : true);
  };

  const metricLabel = (key) => metrics.find(m => m.key === key)?.display_name || key;
  const metricUnit = (key) => metrics.find(m => m.key === key)?.unit || '';

  // If cache is empty, show build prompt
  if (cacheInfo && cacheInfo.total_symbols === 0 && !building) {
    return (
      <div className="p-4 md:p-6 bg-surface-dim min-h-full flex items-center justify-center">
        <div className="bg-surface border border-outline-variant rounded-sm p-8 max-w-lg text-center">
          <Database size={48} className="mx-auto text-outline mb-4" />
          <h2 className="font-display text-2xl text-on-surface mb-2">Build Screener Cache</h2>
          <p className="text-outline text-sm mb-6">
            The screener needs to fetch fundamental data for ~200 NIFTY stocks.
            This takes about 2-3 minutes and only needs to be done once.
          </p>
          <button onClick={buildCache}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-ui text-xs uppercase tracking-widest rounded-sm transition-colors">
            Build Cache Now
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-5 border-b border-outline-variant pb-4 gap-3">
        <div>
          <h1 className="font-display text-3xl text-on-surface">Stock Screener</h1>
          <p className="text-outline text-sm mt-1 font-mono">
            Filter NIFTY 200 by fundamentals — {cacheInfo?.total_symbols || 0} stocks cached
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {cacheInfo && (
            <span className="text-[10px] font-mono text-outline-variant">
              Cache: {cacheInfo.cache_age_hours?.toFixed(1)}h old
            </span>
          )}
          <button onClick={() => { fetch('/api/screener/refresh', { method: 'POST' }); }}
            className="p-2 text-outline hover:text-on-surface transition-colors" title="Refresh cache">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Preset Chips */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(presets).map(([key, preset]) => (
          <button key={key} onClick={() => applyPreset(key)}
            className={`px-3 py-1.5 text-xs font-ui rounded-sm border transition-colors ${
              activePreset === key
                ? 'bg-emerald-600/20 border-emerald-500/50 text-emerald-400'
                : 'bg-surface border-outline-variant text-outline hover:text-on-surface hover:border-outline'
            }`}>
            <Bookmark size={10} className="inline mr-1" />
            {preset.name}
          </button>
        ))}
      </div>

      {/* Filter Builder */}
      <div className="bg-surface border border-outline-variant rounded-sm mb-4 overflow-hidden">
        <div className="px-4 py-3 border-b border-outline-variant flex items-center justify-between">
          <h2 className="font-ui text-xs uppercase tracking-widest text-outline flex items-center gap-1.5">
            <Filter size={14} />Filters
          </h2>
          <div className="flex gap-2">
            <button onClick={addFilter}
              className="flex items-center gap-1 px-3 py-1 text-xs font-ui text-outline hover:text-on-surface border border-outline-variant rounded-sm transition-colors">
              <Plus size={12} />Add Filter
            </button>
            <button onClick={() => runScreen()} disabled={loading}
              className="flex items-center gap-1 px-4 py-1 text-xs font-ui bg-emerald-600 hover:bg-emerald-500 text-white rounded-sm transition-colors disabled:opacity-50">
              {loading ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
              Run Screen
            </button>
          </div>
        </div>

        {filters.length > 0 && (
          <div className="p-3 space-y-2">
            {filters.map((f, idx) => (
              <div key={idx} className="flex items-center gap-2 flex-wrap">
                {idx > 0 && <span className="text-[10px] font-mono text-amber-400 w-8">AND</span>}
                {idx === 0 && <span className="w-8" />}

                <select value={f.metric} onChange={e => updateFilter(idx, 'metric', e.target.value)}
                  className="bg-surface-dim border border-outline-variant rounded-sm px-2 py-1.5 text-xs font-mono text-on-surface focus:border-emerald-500 outline-none">
                  {filterableMetrics.map(m => (
                    <option key={m.key} value={m.key}>{m.display_name}</option>
                  ))}
                </select>

                <select value={f.operator} onChange={e => updateFilter(idx, 'operator', e.target.value)}
                  className="bg-surface-dim border border-outline-variant rounded-sm px-2 py-1.5 text-xs font-mono text-on-surface w-20 focus:border-emerald-500 outline-none">
                  {OPERATORS.map(op => (
                    <option key={op.value} value={op.value}>{op.label}</option>
                  ))}
                </select>

                <input type="number" value={f.value} onChange={e => updateFilter(idx, 'value', e.target.value)}
                  placeholder="Value"
                  className="bg-surface-dim border border-outline-variant rounded-sm px-2 py-1.5 text-xs font-mono text-on-surface w-24 focus:border-emerald-500 outline-none" />

                {f.operator === 'between' && (
                  <>
                    <span className="text-[10px] font-mono text-outline">and</span>
                    <input type="number" value={f.value2} onChange={e => updateFilter(idx, 'value2', e.target.value)}
                      placeholder="Value 2"
                      className="bg-surface-dim border border-outline-variant rounded-sm px-2 py-1.5 text-xs font-mono text-on-surface w-24 focus:border-emerald-500 outline-none" />
                  </>
                )}

                <button onClick={() => removeFilter(idx)}
                  className="p-1 text-outline hover:text-rose-400 transition-colors">
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Sector Filter */}
        {sectors.length > 0 && (
          <div className="px-4 py-2 border-t border-outline-variant/30 flex items-center gap-2">
            <span className="text-[10px] font-ui uppercase tracking-widest text-outline">Sector:</span>
            <select value={sectorFilter} onChange={e => { setSectorFilter(e.target.value); setActivePreset(''); }}
              className="bg-surface-dim border border-outline-variant rounded-sm px-2 py-1 text-xs font-mono text-on-surface focus:border-emerald-500 outline-none">
              <option value="">All Sectors</option>
              {sectors.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* Building indicator */}
      {building && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-sm px-4 py-3 mb-4 flex items-center gap-2">
          <Loader2 size={16} className="animate-spin text-amber-400" />
          <span className="text-xs font-mono text-amber-400">
            Building screener cache... fetching data for ~200 stocks ({cacheInfo?.total_symbols || 0} done)
          </span>
        </div>
      )}

      {/* Results */}
      <div className="bg-surface border border-outline-variant rounded-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-outline-variant flex items-center justify-between">
          <h2 className="font-ui text-xs uppercase tracking-widest text-outline">
            Results — {totalMatches} matches
          </h2>
          {loading && <Loader2 size={14} className="animate-spin text-outline" />}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-surface-dim z-10">
              <tr className="text-outline font-mono uppercase tracking-wider border-b border-outline-variant">
                <th className="text-left px-3 py-2 sticky left-0 bg-surface-dim">Symbol</th>
                <th className="text-left px-3 py-2">Sector</th>
                {displayCols.map(col => (
                  <th key={col} className="text-right px-3 py-2 cursor-pointer hover:text-on-surface transition-colors whitespace-nowrap"
                    onClick={() => toggleSort(col)}>
                    {metricLabel(col)}
                    {sortBy === col && (
                      <span className="ml-1 text-emerald-400">{sortDesc ? '↓' : '↑'}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.length > 0 ? (
                results.map((row, i) => (
                  <tr key={row.symbol}
                    className={`border-b border-outline-variant/30 cursor-pointer hover:bg-surface-dim/80 transition-colors ${
                      i % 2 === 0 ? 'bg-surface' : 'bg-surface-dim/30'
                    }`}
                    onClick={() => navigate(`/company/${row.symbol}`)}>
                    <td className="px-3 py-2 font-mono text-on-surface font-medium sticky left-0 bg-inherit">
                      {row.symbol}
                    </td>
                    <td className="px-3 py-2 text-outline truncate max-w-[120px]" title={row.sector}>
                      {row.sector || '—'}
                    </td>
                    {displayCols.map(col => {
                      const val = row[col];
                      const unit = metricUnit(col);
                      const isGreen = (col === 'roe' || col === 'revenue_growth' || col === 'operating_margin')
                        && val > 0;
                      const isRed = (col === 'debt_to_equity') && val > 2;
                      return (
                        <td key={col} className={`px-3 py-2 text-right font-mono ${
                          isGreen ? 'text-emerald-400' : isRed ? 'text-rose-400' : 'text-outline'
                        }`}>
                          {fmt(val, unit)}
                        </td>
                      );
                    })}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={displayCols.length + 2} className="px-3 py-12 text-center text-outline italic">
                    {loading ? 'Loading...' : 'No results — try adjusting filters or build the cache first'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Disclaimer */}
      <p className="mt-4 text-[10px] font-mono text-outline-variant/50 text-center">
        Data sourced from yfinance. This is for informational purposes only — not investment advice.
      </p>
    </div>
  );
}

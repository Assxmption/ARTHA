import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import CandlestickChart from './CandlestickChart';
import OptionsChain from './OptionsChain';
import SessionReplay from './SessionReplay';
import { useRememberSymbol } from '../lib/activeSymbol';
import { Filter, SlidersHorizontal, Activity, FileText, Newspaper } from 'lucide-react';
import { apiFetch } from '../lib/api';

export default function Dashboard({ theme }) {
  const { symbol } = useParams();
  useRememberSymbol(symbol);
  const navigate = useNavigate();
  const [portfolio, setPortfolio] = useState(null);
  const [regime, setRegime] = useState(null);
  const [loading, setLoading] = useState(true);
  const [ohlcData, setOhlcData] = useState([]);
  const [error, setError] = useState(null);
  const [showFactStore, setShowFactStore] = useState(false);
  const [rawFacts, setRawFacts] = useState({});
  
  useEffect(() => {
    let cancelled = false;
    let timer;
      const fetchData = async () => {
        try {
          const [portRes, regRes] = await Promise.all([
            apiFetch('/api/quant/portfolio').then(async res => {
              const body = await res.json();
              return res.ok ? body : { unavailable: body?.detail?.message || `HTTP ${res.status}` };
            }),
            apiFetch('/api/quant/regime').then(res => res.json())
          ]);
          if (cancelled) return;
          setPortfolio(portRes);
          setRegime(regRes);
          setRawFacts({ portfolio: portRes, regime: regRes });
          // A stale result is shown while the engine recomputes; poll until fresh.
          if (portRes?.refreshing) timer = setTimeout(fetchData, 15000);
        } catch (e) {
          console.error("Failed to fetch quant data", e);
          if (!cancelled) setError("Failed to connect to Quant Engine. Ensure backend is running.");
        } finally {
          if (!cancelled) setLoading(false);
        }
      };
    fetchData();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [symbol]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
          <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Initializing Quant Engine...</p>
        <p className="font-mono text-[10px] text-outline-variant mt-2">Fetching live market data for {symbol}</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="border border-error/50 bg-error/5 p-6 max-w-md text-center">
          <p className="font-ui text-xs uppercase tracking-widest text-error mb-2">System Error</p>
          <p className="font-mono text-sm text-on-surface">{error}</p>
          <button 
            onClick={() => { setError(null); setLoading(true); }} 
            className="mt-6 border border-error px-4 py-2 font-ui text-[10px] uppercase tracking-wider text-error hover:bg-error/10 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header Area */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-4">
        <div>
          <h2 className="font-display text-2xl md:text-3xl mb-1 text-on-surface">Instrument Register - {symbol}</h2>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Real-Time Analytical Dashboard</p>
        </div>
        <div className="flex flex-wrap gap-2 md:gap-4 w-full md:w-auto">
          <button 
            onClick={() => setShowFactStore(!showFactStore)}
            className={`flex items-center gap-2 border px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider transition-colors flex-1 md:flex-none justify-center ${showFactStore ? 'bg-primary text-surface border-primary' : 'border-outline-variant text-on-surface hover:border-primary'}`}
          >
            <Activity size={12} /> {showFactStore ? 'Hide Facts' : 'View Facts'}
          </button>
          <button 
            onClick={() => navigate(`/fundamentals/${symbol}`)}
            className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container flex-1 md:flex-none justify-center"
          >
            <FileText size={12} /> Fundamentals
          </button>
          <button 
            onClick={() => navigate(`/news/${symbol}`)}
            className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container flex-1 md:flex-none justify-center"
          >
            <Newspaper size={12} /> News
          </button>
          <button className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:border-primary transition-colors flex-1 md:flex-none justify-center">
            <Filter size={12} /> Filter
          </button>
          <button className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:border-primary transition-colors bg-surface-container flex-1 md:flex-none justify-center">
            <SlidersHorizontal size={12} /> Slicers
          </button>
        </div>
      </div>
      
      {/* Fact Store Debug Panel */}
      {showFactStore && (
        <div className="mb-6 border border-outline-variant bg-surface-container-low p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-ui text-[10px] uppercase tracking-widest text-primary">Fact Store State (Live)</h3>
            <span className="font-mono text-[10px] text-outline">Pydantic Models</span>
          </div>
          <pre className="font-mono text-[10px] text-on-surface-variant overflow-x-auto p-4 bg-surface border border-outline-variant/50 max-h-64 overflow-y-auto">
            {JSON.stringify(rawFacts, null, 2)}
          </pre>
        </div>
      )}

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        
        {/* Main Chart Area */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          <CandlestickChart symbol={symbol} />
          
          {/* Regime Data block */}
          <div className="border border-outline-variant bg-surface p-4">
            <h3 className="font-ui text-[10px] uppercase tracking-widest text-outline mb-4">Current Regime State</h3>
            {regime ? (
              <div className="grid grid-cols-3 gap-4">
                <div className="border-l-2 border-primary pl-3">
                  <p className="font-ui text-[10px] uppercase text-outline">Market State</p>
                  <p className={`font-mono text-lg ${regime.current_regime === 'BULL' ? 'text-secondary' : regime.current_regime === 'BEAR' ? 'text-error' : 'text-primary'}`}>{regime.current_regime || 'UNKNOWN'}</p>
                </div>
                <div className="border-l-2 border-outline-variant pl-3">
                  <p className="font-ui text-[10px] uppercase text-outline">In State Since</p>
                  <p className="font-mono text-sm text-on-surface">
                    {regime.state_since ? `${regime.state_since} (${regime.sessions_in_state} sessions)` : 'N/A'}
                  </p>
                </div>
                <div className="border-l-2 border-outline-variant pl-3">
                  <p className="font-ui text-[10px] uppercase text-outline">Date Range</p>
                  <p className="font-mono text-sm text-on-surface">{regime.date_range || 'N/A'}</p>
                </div>
              </div>
            ) : (
              <p className="font-mono text-xs text-outline">Regime data unavailable</p>
            )}
          </div>
          
          {/* Options Chain Data block */}
          <OptionsChain symbol={symbol} />
        </div>

        {/* Sidebar Data Area */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="border border-outline-variant bg-surface flex-1">
            <div className="border-b border-outline-variant p-3 bg-surface-container-low flex justify-between items-center">
              <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">Portfolio Metrics (Live OOS)</span>
              <Activity size={14} className="text-primary" />
            </div>
            {portfolio?.unavailable ? (
              <p className="p-3 font-mono text-xs text-error">Live engine unavailable: {portfolio.unavailable}</p>
            ) : (
            <div className="px-3 py-2 border-b border-outline-variant/50 font-mono text-[10px] text-outline space-y-1">
              <div className="flex flex-wrap gap-2 items-center">
                <span>As of {portfolio?.as_of ?? '—'}</span>
                {portfolio?.provisional && <span className="px-1 border border-primary text-primary">PROVISIONAL</span>}
                {portfolio?.stale && <span className="px-1 border border-outline text-outline">{portfolio?.refreshing ? 'REFRESHING' : 'STALE'}</span>}
                <span className={`px-1 border ${portfolio?.vt_validated ? 'border-secondary text-secondary' : 'border-error text-error'}`}>
                  GATE {portfolio?.vt_validated ? 'PASSED' : 'NOT PASSED'}
                </span>
              </div>
              {portfolio?.windows && (
                <div>OOS {portfolio.windows.test_start} → {portfolio.windows.test_end} ({portfolio.windows.oos_sessions} sessions)</div>
              )}
            </div>
            )}
            <div className="p-0">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-outline-variant bg-surface-container-lowest text-outline">
                    <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider">Metric</th>
                    <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['OOS Sharpe (vol-targeted)', fmtNum(portfolio?.vt_sharpe)],
                    ['OOS Max Drawdown', fmtPct(portfolio?.vt_max_dd)],
                    ['OOS Positive Sessions', fmtPct(portfolio?.win_rate)],
                    ['NIFTY 50 OOS Sharpe', fmtNum(portfolio?.benchmark_sharpe)],
                    ['Universe Size', portfolio?.universe_size ?? 'N/A'],
                    ['Pairs Validated', portfolio?.pairs_validated != null ? `${portfolio.pairs_validated} of ${portfolio.pairs_total}` : 'N/A'],
                  ].map(([label, val], idx) => (
                    <tr key={label} className={`border-b border-outline-variant/30 ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                      <td className="p-3 text-on-surface-variant">{label}</td>
                      <td className="p-3 text-right text-on-surface">{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {portfolio?.disclaimer && (
                <p className="p-3 font-mono text-[10px] text-outline-variant">{portfolio.disclaimer}</p>
              )}
            </div>
          </div>
          <SessionReplay />
        </div>
      </div>
    </div>
  );
}

// `0` and negative values are real results — only null/undefined is "N/A".
function fmtNum(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : 'N/A';
}

function fmtPct(v) {
  return typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : 'N/A';
}

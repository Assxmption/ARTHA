import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart2, Play, Loader2, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';

export default function Simulator() {
  const navigate = useNavigate();
  const [simulations, setSimulations] = useState([]);
  const [activeSim, setActiveSim] = useState(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [capital, setCapital] = useState(1500000);
  const [days, setDays] = useState(365);
  const [strategyType, setStrategyType] = useState('MULTI_STRATEGY');
  const pollingRef = useRef(null);

  // Load existing simulations
  useEffect(() => {
    fetch('/api/sim/list')
      .then(r => r.json())
      .then(data => setSimulations(data.simulations || []))
      .catch(() => {});
  }, []);

  const startSimulation = async () => {
    setStarting(true);
    try {
      const res = await fetch('/api/sim/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ capital, days, strategy_type: strategyType }),
      });
      const data = await res.json();
      if (data.sim_id) {
        pollStatus(data.sim_id);
      }
    } catch (err) {
      console.error(err);
      setStarting(false);
    }
  };

  const pollStatus = (simId) => {
    setLoading(true);
    const poll = async () => {
      try {
        const res = await fetch(`/api/sim/${simId}/status`);
        const data = await res.json();
        setActiveSim(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollingRef.current);
          setLoading(false);
          setStarting(false);
          // Refresh list
          fetch('/api/sim/list').then(r => r.json()).then(d => setSimulations(d.simulations || []));
        }
      } catch (err) {
        clearInterval(pollingRef.current);
        setLoading(false);
        setStarting(false);
      }
    };
    pollingRef.current = setInterval(poll, 2000);
    poll();
  };

  const loadSim = (simId) => {
    fetch(`/api/sim/${simId}/status`).then(r => r.json()).then(setActiveSim);
  };

  useEffect(() => () => clearInterval(pollingRef.current), []);

  const formatCurrency = (val) => `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  const isPositive = (val) => val >= 0;

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-4">
        <div>
          <h1 className="font-display text-3xl text-on-surface">Backtest Lab</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline mt-1">
            Automated Trading Simulator — Medallion-style high-frequency factor rotation
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="border border-outline-variant bg-surface p-4">
          <label className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-2">Starting Capital (₹)</label>
          <input
            type="number"
            value={capital}
            onChange={e => setCapital(Number(e.target.value))}
            className="w-full bg-surface-container border border-outline-variant px-3 py-2 font-mono text-sm text-on-surface focus:border-primary focus:outline-none"
          />
        </div>
        <div className="border border-outline-variant bg-surface p-4">
          <label className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-2">Simulation Period (Days)</label>
          <input
            type="number"
            value={days}
            onChange={e => setDays(Number(e.target.value))}
            className="w-full bg-surface-container border border-outline-variant px-3 py-2 font-mono text-sm text-on-surface focus:border-primary focus:outline-none"
          />
        </div>
                <div className="border border-outline-variant bg-surface p-4">
          <label className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-2">Strategy Type</label>
          <select
            value={strategyType}
            onChange={e => setStrategyType(e.target.value)}
            className="w-full bg-surface-container border border-outline-variant px-3 py-2 font-mono text-sm text-on-surface focus:border-primary focus:outline-none"
          >
            <option value="MULTI_STRATEGY">ARTHA Multi-Strategy (Full Quant Engine)</option>
            <option value="EQUITY_LONG_ONLY">Equity Long-Only</option>
            <option value="PCA_STATARB">PCA Stat-Arb</option>
            <option value="IRON_CONDOR">NIFTY Iron Condor</option>
          </select>
        </div>
        <div className="border border-outline-variant bg-surface p-4 flex items-end">
          <button
            onClick={startSimulation}
            disabled={starting || loading}
            className="w-full flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2.5 font-ui text-xs uppercase tracking-wider hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {starting ? 'Running...' : 'Run Simulation'}
          </button>
        </div>
      </div>

      {/* Active Simulation Results */}
      {activeSim && (
        <div className="mb-6">
          {/* Status banner */}
          {activeSim.status === 'running' && (
            <div className="border border-primary/30 bg-primary/5 p-4 mb-4 flex items-center gap-3">
              <Loader2 size={16} className="animate-spin text-primary" />
              <span className="font-mono text-xs text-primary">{activeSim.progress || 'Simulation in progress...'}</span>
            </div>
          )}

          {activeSim.status === 'completed' && (
            <>
              {/* Key Metrics Strip */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-4">
                {[
                  { label: 'Final NAV', value: formatCurrency(activeSim.final_nav), color: isPositive(activeSim.total_return_pct) ? 'text-secondary' : 'text-error' },
                  { label: 'Return', value: `${activeSim.total_return_pct >= 0 ? '+' : ''}${activeSim.total_return_pct}%`, color: isPositive(activeSim.total_return_pct) ? 'text-secondary' : 'text-error' },
                  { label: 'Sharpe', value: activeSim.sharpe_ratio?.toFixed(3), color: activeSim.sharpe_ratio >= 0.5 ? 'text-secondary' : activeSim.sharpe_ratio >= 0 ? 'text-on-surface' : 'text-error' },
                  { label: 'Sortino', value: activeSim.sortino_ratio?.toFixed(3), color: 'text-on-surface' },
                  { label: 'Max DD', value: `${activeSim.max_drawdown_pct?.toFixed(1)}%`, color: 'text-error' },
                  { label: 'Win Rate', value: `${activeSim.win_rate}%`, color: activeSim.win_rate >= 50 ? 'text-secondary' : 'text-on-surface' },
                ].map(m => (
                  <div key={m.label} className="border border-outline-variant bg-surface p-3">
                    <span className="font-ui text-[10px] uppercase tracking-widest text-outline block">{m.label}</span>
                    <span className={`font-mono text-lg font-bold tabular-nums ${m.color}`}>{m.value}</span>
                  </div>
                ))}
              </div>

              {/* Full Metrics Table */}
              <div className="border border-outline-variant bg-surface overflow-hidden">
                <div className="border-b border-outline-variant bg-surface-container-low p-3">
                  <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">Simulation Report</span>
                </div>
                <div className="p-0">
                  <table className="w-full text-left font-mono text-xs">
                    <tbody>
                      {[
                        ['Capital', formatCurrency(activeSim.initial_capital)],
                        ['Final NAV', formatCurrency(activeSim.final_nav)],
                        ['Total Return', `${activeSim.total_return_pct >= 0 ? '+' : ''}${activeSim.total_return_pct}%`],
                        ['Sharpe Ratio', activeSim.sharpe_ratio?.toFixed(3)],
                        ['Sortino Ratio', activeSim.sortino_ratio?.toFixed(3)],
                        ['Max Drawdown', `${activeSim.max_drawdown_pct?.toFixed(1)}%`],
                        ['Win Rate', `${activeSim.win_rate}%`],
                        ['Profit Factor', activeSim.profit_factor?.toFixed(3)],
                        ['Total Trades', activeSim.total_trades],
                        ['Positions', activeSim.n_positions],
                        ['Regime', String(activeSim.current_regime).replace('RegimeState.', '')],
                        ['Period', `${activeSim.start_date} → ${activeSim.end_date}`],
                      ].map(([label, value], idx) => (
                        <tr key={label} className={`border-b border-outline-variant/30 ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                          <td className="p-3 text-on-surface-variant">{label}</td>
                          <td className="p-3 text-right text-on-surface tabular-nums">{value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Trade Log Section */}
              <div className="border border-outline-variant bg-surface overflow-hidden mt-4">
                <div className="border-b border-outline-variant bg-surface-container-low p-3">
                  <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">Actual Trades Taken</span>
                </div>
                <div className="p-0 overflow-x-auto max-h-96">
                  {activeSim.trade_log && activeSim.trade_log.length > 0 ? (
                    <table className="w-full text-left font-mono text-xs relative">
                      <thead className="sticky top-0 bg-surface-container-low border-b border-outline-variant/50">
                        <tr>
                          <th className="p-3 font-normal text-outline">Date</th>
                          <th className="p-3 font-normal text-outline">Action</th>
                          <th className="p-3 font-normal text-outline">Symbol</th>
                          <th className="p-3 font-normal text-outline text-right">Price</th>
                          <th className="p-3 font-normal text-outline text-right">Qty</th>
                          <th className="p-3 font-normal text-outline text-right">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {activeSim.trade_log.map((trade, idx) => (
                          <tr key={idx} className={`border-b border-outline-variant/30 ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                            <td className="p-3 text-on-surface-variant">{trade.date}</td>
                            <td className={`p-3 ${trade.action === 'BUY' ? 'text-secondary' : trade.action === 'SELL' ? 'text-error' : 'text-on-surface-variant'}`}>{trade.action}</td>
                            <td className="p-3 text-on-surface">{trade.symbol}</td>
                            <td className="p-3 text-right tabular-nums">{formatCurrency(trade.price)}</td>
                            <td className="p-3 text-right tabular-nums">{trade.quantity || 1}</td>
                            <td className={`p-3 text-right tabular-nums ${trade.pnl > 0 ? 'text-secondary' : trade.pnl < 0 ? 'text-error' : 'text-on-surface-variant'}`}>
                              {trade.pnl !== undefined && trade.pnl !== null ? formatCurrency(trade.pnl) : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="p-8 text-center text-on-surface-variant font-mono text-xs">
                      {strategyType === 'MULTI_STRATEGY' ? 
                        "Multi-Strategy operates as a continuous portfolio allocator. It scales position weights dynamically based on vol-targets and tangency correlations rather than executing discrete standalone trades."
                        : "No discrete trades recorded for this simulation."}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Past Simulations */}
      {simulations.length > 0 && (
        <div className="border border-outline-variant bg-surface overflow-hidden">
          <div className="border-b border-outline-variant bg-surface-container-low p-3 flex justify-between items-center">
            <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">Simulation History</span>
            <span className="font-mono text-[10px] text-outline">{simulations.length} runs</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-outline-variant bg-surface-container-lowest text-outline">
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider">ID</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider">Status</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Capital</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">NAV</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Return</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Sharpe</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Trades</th>
                  <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right"></th>
                </tr>
              </thead>
              <tbody>
                {simulations.map((sim, idx) => (
                  <tr key={sim.sim_id} className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                    <td className="p-3 text-on-surface">{sim.sim_id}</td>
                    <td className="p-3">
                      <span className={`px-1.5 py-0.5 font-ui text-[9px] uppercase tracking-wider border ${sim.status === 'completed' ? 'border-secondary/30 bg-secondary/10 text-secondary' : 'border-primary/30 bg-primary/10 text-primary'}`}>
                        {sim.status}
                      </span>
                    </td>
                    <td className="p-3 text-right text-on-surface-variant tabular-nums">{formatCurrency(sim.initial_capital)}</td>
                    <td className="p-3 text-right text-on-surface tabular-nums">{sim.final_nav ? formatCurrency(sim.final_nav) : '—'}</td>
                    <td className={`p-3 text-right tabular-nums ${sim.total_return_pct >= 0 ? 'text-secondary' : 'text-error'}`}>
                      {sim.total_return_pct !== undefined ? `${sim.total_return_pct >= 0 ? '+' : ''}${sim.total_return_pct}%` : '—'}
                    </td>
                    <td className="p-3 text-right text-on-surface tabular-nums">{sim.sharpe_ratio ?? '—'}</td>
                    <td className="p-3 text-right text-on-surface tabular-nums">{sim.total_trades ?? '—'}</td>
                    <td className="p-3 text-right">
                      <button onClick={() => loadSim(sim.sim_id)} className="text-[10px] font-ui uppercase tracking-wider text-primary hover:underline">
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

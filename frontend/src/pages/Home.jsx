import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { TrendingUp, TrendingDown, Activity, BarChart2, Newspaper, Zap, RefreshCw } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();
  const [watchlist, setWatchlist] = useState([]);
  const [regime, setRegime] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [wlRes, regRes] = await Promise.all([
          fetch('/api/watchlist?symbols=RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL').then(r => r.json()),
          fetch('/api/quant/regime').then(r => r.json()),
        ]);
        setWatchlist(wlRes.stocks || []);
        setRegime(regRes);
      } catch (e) {
        console.error('Failed to fetch home data', e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const gainers = [...watchlist].filter(s => !s.error).sort((a, b) => (b.changePct || 0) - (a.changePct || 0)).slice(0, 3);
  const losers = [...watchlist].filter(s => !s.error).sort((a, b) => (a.changePct || 0) - (b.changePct || 0)).slice(0, 3);

  const regimeColor = {
    'BULL': 'text-secondary',
    'BEAR': 'text-error',
    'SIDEWAYS': 'text-primary',
  };
  const regimeDot = {
    'BULL': 'bg-secondary',
    'BEAR': 'bg-error',
    'SIDEWAYS': 'bg-primary',
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
          <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Loading market data...</p>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header */}
      <div className="flex justify-between items-end mb-6">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Market Overview</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Live NSE Data</p>
        </div>
        {/* Regime Badge */}
        {regime && (
          <div className="flex items-center gap-3 border border-outline-variant bg-surface px-4 py-2">
            <span className={`w-2 h-2 rounded-full ${regimeDot[regime.current_regime] || 'bg-outline'} animate-pulse`}></span>
            <div>
              <span className="font-ui text-[10px] uppercase tracking-widest text-outline block">Market Regime</span>
              <span className={`font-mono text-sm font-bold ${regimeColor[regime.current_regime] || 'text-on-surface'}`}>
                {regime.current_regime}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Top Gainers — LIVE */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <TrendingUp size={16} className="text-secondary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Top Gainers</h2>
          </div>
          <div className="flex flex-col gap-1">
            {gainers.map(stock => (
              <button 
                key={stock.symbol}
                onClick={() => navigate(`/dashboard/${stock.symbol}`)}
                className="flex justify-between items-center p-2.5 hover:bg-surface-container transition-colors text-left group"
              >
                <div>
                  <span className="font-mono text-sm text-on-surface group-hover:text-primary transition-colors">{stock.symbol}</span>
                  <span className="font-ui text-[10px] text-outline ml-2 hidden sm:inline">{stock.name?.split(' ')[0]}</span>
                </div>
                <div className="text-right">
                  <span className="font-mono text-xs text-on-surface-variant mr-3">₹{stock.price?.toLocaleString('en-IN')}</span>
                  <span className="font-mono text-xs text-secondary">+{stock.changePct}%</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Top Losers — LIVE */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <TrendingDown size={16} className="text-error" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Top Losers</h2>
          </div>
          <div className="flex flex-col gap-1">
            {losers.map(stock => (
              <button 
                key={stock.symbol}
                onClick={() => navigate(`/dashboard/${stock.symbol}`)}
                className="flex justify-between items-center p-2.5 hover:bg-surface-container transition-colors text-left group"
              >
                <div>
                  <span className="font-mono text-sm text-on-surface group-hover:text-primary transition-colors">{stock.symbol}</span>
                  <span className="font-ui text-[10px] text-outline ml-2 hidden sm:inline">{stock.name?.split(' ')[0]}</span>
                </div>
                <div className="text-right">
                  <span className="font-mono text-xs text-on-surface-variant mr-3">₹{stock.price?.toLocaleString('en-IN')}</span>
                  <span className="font-mono text-xs text-error">{stock.changePct}%</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <Zap size={16} className="text-primary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Quick Actions</h2>
          </div>
          <div className="flex flex-col gap-2">
            <button 
              onClick={() => navigate('/signals')}
              className="flex items-center gap-3 p-3 border border-outline-variant/50 bg-surface-container-low hover:border-primary hover:bg-primary/5 transition-colors text-left group"
            >
              <Activity size={16} className="text-primary" />
              <div>
                <span className="font-ui text-xs text-on-surface group-hover:text-primary transition-colors block">Quant Signals</span>
                <span className="font-mono text-[10px] text-outline">HMM regime + pair signals</span>
              </div>
            </button>
            <button 
              onClick={() => navigate('/simulator')}
              className="flex items-center gap-3 p-3 border border-outline-variant/50 bg-surface-container-low hover:border-primary hover:bg-primary/5 transition-colors text-left group"
            >
              <BarChart2 size={16} className="text-primary" />
              <div>
                <span className="font-ui text-xs text-on-surface group-hover:text-primary transition-colors block">Trading Simulator</span>
                <span className="font-mono text-[10px] text-outline">Backtest strategy with ₹15L</span>
              </div>
            </button>
            <button 
              onClick={() => navigate('/news/RELIANCE')}
              className="flex items-center gap-3 p-3 border border-outline-variant/50 bg-surface-container-low hover:border-primary hover:bg-primary/5 transition-colors text-left group"
            >
              <Newspaper size={16} className="text-primary" />
              <div>
                <span className="font-ui text-xs text-on-surface group-hover:text-primary transition-colors block">News & Sentiment</span>
                <span className="font-mono text-[10px] text-outline">VADER-scored market news</span>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Full watchlist row */}
      <div className="mt-6 border border-outline-variant bg-surface overflow-hidden">
        <div className="border-b border-outline-variant bg-surface-container-low p-3 flex justify-between items-center">
          <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">All Instruments</span>
          <button onClick={() => navigate('/watchlist')} className="font-ui text-[10px] uppercase tracking-wider text-primary hover:underline">
            Full Watchlist →
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-outline-variant bg-surface-container-lowest text-outline">
                <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider">Symbol</th>
                <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Price</th>
                <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Change</th>
                <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">Mkt Cap</th>
                <th className="font-ui text-[10px] uppercase p-3 font-normal tracking-wider text-right">P/E</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.filter(s => !s.error).map((s, i) => (
                <tr key={s.symbol} className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors cursor-pointer ${i % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}
                    onClick={() => navigate(`/dashboard/${s.symbol}`)}>
                  <td className="p-3 text-on-surface font-bold">{s.symbol}</td>
                  <td className="p-3 text-right text-on-surface tabular-nums">₹{s.price?.toLocaleString('en-IN')}</td>
                  <td className={`p-3 text-right tabular-nums ${s.changePct >= 0 ? 'text-secondary' : 'text-error'}`}>
                    {s.changePct >= 0 ? '+' : ''}{s.changePct}%
                  </td>
                  <td className="p-3 text-right text-on-surface-variant tabular-nums">{s.marketCap}</td>
                  <td className="p-3 text-right text-on-surface tabular-nums">{s.pe ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

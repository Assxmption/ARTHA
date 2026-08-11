import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, TrendingUp, TrendingDown, Loader2, RefreshCw } from 'lucide-react';

export default function Watchlist() {
  const navigate = useNavigate();
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const DEFAULT_SYMBOLS = 'RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL';

  const fetchWatchlist = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/watchlist?symbols=${DEFAULT_SYMBOLS}`);
      if (!res.ok) throw new Error('Failed to fetch watchlist');
      const data = await res.json();
      setStocks(data.stocks || []);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchWatchlist(); }, []);

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      <div className="flex justify-between items-end mb-6 border-b border-outline-variant pb-4">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Watchlist</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Tracked Instruments (NSE) — Live Data</p>
        </div>
        <button 
          onClick={fetchWatchlist}
          disabled={loading}
          className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container disabled:opacity-50"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center min-h-[40vh]">
          <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
            <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
          </div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Fetching live market data...</p>
        </div>
      )}

      {error && (
        <div className="border border-error/50 bg-error/5 p-6 text-center">
          <p className="font-mono text-sm text-error">{error}</p>
          <button onClick={fetchWatchlist} className="mt-4 border border-error px-4 py-2 font-ui text-[10px] uppercase tracking-wider text-error hover:bg-error/10 transition-colors">
            Retry
          </button>
        </div>
      )}

      {!loading && !error && (
        <div className="border border-outline-variant bg-surface overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-sm">
              <thead>
                <tr className="border-b border-outline-variant bg-surface-container-low text-outline">
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">Symbol</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">Name</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">LTP</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Change</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Mkt Cap</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">P/E</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Sector</th>
                  <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {stocks.map((item, idx) => {
                  if (item.error) return null;
                  const isUp = item.changePct >= 0;
                  return (
                    <tr key={item.symbol} className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                      <td className="p-4">
                        <button 
                          onClick={() => navigate(`/dashboard/${item.symbol}`)}
                          className="font-bold text-on-surface hover:text-primary transition-colors"
                        >
                          {item.symbol}
                        </button>
                      </td>
                      <td className="p-4 text-on-surface-variant text-xs">{item.name}</td>
                      <td className="p-4 text-right text-on-surface tabular-nums">₹{item.price?.toLocaleString('en-IN')}</td>
                      <td className={`p-4 text-right tabular-nums ${isUp ? 'text-secondary' : 'text-error'}`}>
                        <span className="flex items-center justify-end gap-1">
                          {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                          {isUp ? '+' : ''}{item.changePct}%
                        </span>
                      </td>
                      <td className="p-4 text-right text-on-surface-variant text-xs tabular-nums">{item.marketCap}</td>
                      <td className="p-4 text-right text-on-surface tabular-nums">{item.pe ?? '—'}</td>
                      <td className="p-4 text-right text-on-surface-variant text-xs">{item.sector}</td>
                      <td className="p-4 text-right">
                        <button 
                          onClick={() => navigate(`/dashboard/${item.symbol}`)}
                          className="text-[10px] font-ui uppercase tracking-wider border border-outline-variant px-2 py-1 text-on-surface-variant hover:text-primary hover:border-primary transition-colors"
                        >
                          Analyze
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

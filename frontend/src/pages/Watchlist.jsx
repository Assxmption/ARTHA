import { useNavigate } from 'react-router-dom';
import { Eye, TrendingUp, TrendingDown, Activity } from 'lucide-react';

export default function Watchlist() {
  const navigate = useNavigate();

  const watchlistData = [
    { symbol: 'RELIANCE', price: '2,945.20', change: '+1.2%', trend: 'up' },
    { symbol: 'HDFCBANK', price: '1,642.10', change: '-0.4%', trend: 'down' },
    { symbol: 'TCS', price: '4,012.00', change: '+0.8%', trend: 'up' },
    { symbol: 'INFY', price: '1,520.45', change: '-1.1%', trend: 'down' },
    { symbol: 'GOLD', price: '71,200.00', change: '+0.1%', trend: 'up' },
  ];

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      <div className="flex justify-between items-end mb-6 border-b border-outline-variant pb-4">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Watchlist</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Tracked Instruments (NSE & MCX)</p>
        </div>
        <button className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container">
          <Eye size={12} /> Add Symbol
        </button>
      </div>

      <div className="border border-outline-variant bg-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-sm">
            <thead>
              <tr className="border-b border-outline-variant bg-surface-container-low text-outline">
                <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">Symbol</th>
                <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">LTP</th>
                <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">Change</th>
                <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {watchlistData.map((item, idx) => (
                <tr key={item.symbol} className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                  <td className="p-4">
                    <button 
                      onClick={() => navigate(`/dashboard/${item.symbol}`)}
                      className="font-bold text-on-surface hover:text-primary transition-colors"
                    >
                      {item.symbol}
                    </button>
                  </td>
                  <td className="p-4 text-on-surface">{item.price}</td>
                  <td className={`p-4 flex items-center gap-1 ${item.trend === 'up' ? 'text-secondary' : 'text-error'}`}>
                    {item.trend === 'up' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                    {item.change}
                  </td>
                  <td className="p-4 text-right">
                    <button 
                      onClick={() => navigate(`/dashboard/${item.symbol}`)}
                      className="text-[10px] font-ui uppercase tracking-wider border border-outline-variant px-2 py-1 text-on-surface-variant hover:text-primary hover:border-primary transition-colors"
                    >
                      Analyze
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

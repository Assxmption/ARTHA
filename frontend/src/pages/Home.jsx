import { useNavigate } from 'react-router-dom';
import { TrendingUp, Activity, BarChart2 } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();
  
  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      <div className="mb-8">
        <h1 className="font-display text-3xl mb-1 text-on-surface">Market Overview</h1>
        <p className="font-ui text-xs uppercase tracking-widest text-outline">Trending & Top Movers</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Trending Section */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <TrendingUp size={16} className="text-primary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Trending (NSE)</h2>
          </div>
          <div className="flex flex-col gap-2">
            {['RELIANCE', 'HDFCBANK', 'INFY'].map(sym => (
              <button 
                key={sym} 
                onClick={() => navigate(`/dashboard/${sym}`)}
                className="flex justify-between items-center p-2 hover:bg-surface-container transition-colors text-left group"
              >
                <span className="font-mono text-sm text-on-surface group-hover:text-primary transition-colors">{sym}</span>
                <span className="font-mono text-xs text-secondary">+1.2%</span>
              </button>
            ))}
          </div>
        </div>

        {/* Top Gainers */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <BarChart2 size={16} className="text-secondary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Top Gainers</h2>
          </div>
          <div className="flex flex-col gap-2">
            {['TATAMOTORS', 'WIPRO', 'ITC'].map(sym => (
              <button 
                key={sym} 
                onClick={() => navigate(`/dashboard/${sym}`)}
                className="flex justify-between items-center p-2 hover:bg-surface-container transition-colors text-left group"
              >
                <span className="font-mono text-sm text-on-surface group-hover:text-primary transition-colors">{sym}</span>
                <span className="font-mono text-xs text-secondary">+3.4%</span>
              </button>
            ))}
          </div>
        </div>

        {/* Quant Signals Highlight */}
        <div className="border border-outline-variant bg-surface p-4">
          <div className="flex items-center gap-2 mb-4 border-b border-outline-variant pb-2">
            <Activity size={16} className="text-primary" />
            <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">Active Signals</h2>
          </div>
          <div className="flex flex-col gap-2">
            <div className="p-2 border border-outline-variant/50 bg-surface-container-low">
              <span className="font-ui text-[10px] text-primary uppercase">Long</span>
              <p className="font-mono text-xs text-on-surface">RELIANCE</p>
            </div>
            <div className="p-2 border border-outline-variant/50 bg-surface-container-low">
              <span className="font-ui text-[10px] text-error uppercase">Short</span>
              <p className="font-mono text-xs text-on-surface">TCS</p>
            </div>
            <button 
              onClick={() => navigate('/signals')}
              className="mt-2 text-center font-ui text-[10px] uppercase tracking-widest text-primary hover:bg-primary/10 p-2 transition-colors border border-primary/20"
            >
              View All Signals
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

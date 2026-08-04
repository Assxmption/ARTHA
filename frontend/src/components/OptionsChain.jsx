import { useState, useEffect } from 'react';

export default function OptionsChain({ symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const res = await fetch(`/api/quant/options/${symbol}`);
        const json = await res.json();
        setData(json);
      } catch (e) {
        console.error("Failed to fetch options data", e);
      } finally {
        setLoading(false);
      }
    };
    
    setLoading(true);
    fetchOptions();
  }, [symbol]);

  if (loading) {
    return <div className="p-4 font-mono text-xs text-outline border border-outline-variant bg-surface">LOADING OPTIONS DATA...</div>;
  }

  if (!data || !data.calls || data.calls.length === 0) {
    return (
      <div className="border border-outline-variant bg-surface p-4 text-center">
        <p className="font-mono text-xs text-outline">No F&O data available for {symbol}</p>
      </div>
    );
  }

  return (
    <div className="border border-outline-variant bg-surface overflow-hidden">
      <div className="border-b border-outline-variant p-3 bg-surface-container-low flex justify-between items-center">
        <div>
          <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface block">Options Chain</span>
          <span className="font-mono text-xs text-primary">Expiry: {data.expiry} | Spot: {data.spotPrice}</span>
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-center font-mono text-xs">
          <thead>
            <tr className="border-b border-outline-variant bg-surface-container-lowest">
              <th colSpan="3" className="p-2 border-r border-outline-variant text-secondary">CALLS</th>
              <th className="p-2 border-r border-outline-variant bg-surface-variant font-ui text-[10px] tracking-widest uppercase">Strike</th>
              <th colSpan="3" className="p-2 text-error">PUTS</th>
            </tr>
            <tr className="border-b border-outline-variant bg-surface-container text-[10px] text-outline uppercase font-ui tracking-widest">
              <th className="p-2 border-r border-outline-variant/30 font-normal">OI</th>
              <th className="p-2 border-r border-outline-variant/30 font-normal">Vol</th>
              <th className="p-2 border-r border-outline-variant font-normal">LTP</th>
              <th className="p-2 border-r border-outline-variant bg-surface-variant font-normal">Price</th>
              <th className="p-2 border-r border-outline-variant/30 font-normal">LTP</th>
              <th className="p-2 border-r border-outline-variant/30 font-normal">Vol</th>
              <th className="p-2 font-normal">OI</th>
            </tr>
          </thead>
          <tbody>
            {data.calls.map((call, idx) => {
              const put = data.puts[idx];
              const isCallITM = call.strike < data.spotPrice;
              const isPutITM = put.strike > data.spotPrice;
              
              return (
                <tr key={call.strike} className="border-b border-outline-variant/30 hover:bg-surface-container-low transition-colors">
                  <td className={`p-2 border-r border-outline-variant/30 ${isCallITM ? 'bg-primary/5 text-primary' : 'text-on-surface-variant'}`}>{call.openInterest.toLocaleString()}</td>
                  <td className={`p-2 border-r border-outline-variant/30 ${isCallITM ? 'bg-primary/5 text-on-surface' : 'text-on-surface'}`}>{call.volume.toLocaleString()}</td>
                  <td className={`p-2 border-r border-outline-variant font-bold ${isCallITM ? 'bg-primary/5 text-secondary' : 'text-secondary'}`}>{call.lastPrice.toFixed(2)}</td>
                  
                  <td className="p-2 border-r border-outline-variant bg-surface-variant font-bold text-on-surface">
                    {call.strike}
                  </td>
                  
                  <td className={`p-2 border-r border-outline-variant/30 font-bold ${isPutITM ? 'bg-error/5 text-error' : 'text-error'}`}>{put.lastPrice.toFixed(2)}</td>
                  <td className={`p-2 border-r border-outline-variant/30 ${isPutITM ? 'bg-error/5 text-on-surface' : 'text-on-surface'}`}>{put.volume.toLocaleString()}</td>
                  <td className={`p-2 ${isPutITM ? 'bg-error/5 text-error' : 'text-on-surface-variant'}`}>{put.openInterest.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

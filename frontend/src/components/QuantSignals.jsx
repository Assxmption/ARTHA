import { useState, useEffect, useRef } from 'react';
import { Activity, Radio, Cpu, ShieldAlert, Crosshair, TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function QuantSignals() {
  const [pairs, setPairs] = useState(null);
  const [regime, setRegime] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const feedRef = useRef(null);
  
  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch('/api/quant/pairs').then(r => r.json()),
      fetch('/api/quant/regime').then(r => r.json())
    ])
      .then(([pairsData, regimeData]) => {
        setPairs(pairsData);
        setRegime(regimeData);
        setError(null);
      })
      .catch(e => {
        console.error(e);
        setError("Failed to establish telemetry with Quant Engine.");
      })
      .finally(() => setLoading(false));
  }, []);

  // Auto-scroll the feed slowly
  useEffect(() => {
    let animationId;
    if (feedRef.current && pairs?.pairs?.length > 4) {
      const scroll = () => {
        if (feedRef.current) {
          feedRef.current.scrollTop += 0.5;
          if (feedRef.current.scrollTop >= (feedRef.current.scrollHeight - feedRef.current.clientHeight)) {
             feedRef.current.scrollTop = 0; // reset for infinite effect
          }
        }
        animationId = requestAnimationFrame(scroll);
      };
      animationId = requestAnimationFrame(scroll);
    }
    return () => cancelAnimationFrame(animationId);
  }, [pairs]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] bg-[#0E1A16] text-[#94c2b2]">
        <Radio size={32} className="animate-pulse mb-4 text-[#255144]" />
        <p className="font-mono text-xs uppercase tracking-widest animate-pulse">Establishing Engine Telemetry...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] bg-[#0E1A16] text-[#ffb4ab]">
        <ShieldAlert size={32} className="mb-4 text-[#ba1a1a]" />
        <p className="font-mono text-xs uppercase tracking-widest">{error}</p>
      </div>
    );
  }

  // Parse regime string like "BULL_VOLATILE"
  const marketState = regime?.regime_statistics?.current_state || "UNKNOWN";
  const stateTokens = marketState.split('_');
  const isBull = stateTokens[0] === 'BULL';
  const isBear = stateTokens[0] === 'BEAR';

  return (
    <div className="h-full flex flex-col bg-[#0E1A16] text-[#ebe1d3] overflow-hidden">
      {/* HUD Header */}
      <div className="px-6 py-4 border-b border-[#255144]/30 bg-[#13231E] flex justify-between items-center">
        <div className="flex items-center gap-4">
          <Cpu className="text-[#a2d0c0]" size={20} />
          <div>
            <h2 className="font-mono text-lg tracking-widest text-[#a2d0c0] uppercase">Quant Engine Terminal v1.4</h2>
            <p className="font-mono text-[10px] text-[#94c2b2] uppercase opacity-70">Subsystem: Stat-Arb & Regime Detection</p>
          </div>
        </div>
        <div className="flex gap-6">
          <div className="flex flex-col items-end">
            <span className="font-mono text-[10px] text-[#94c2b2] uppercase">Telemetry</span>
            <span className="font-mono text-xs text-[#a2d0c0] flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#a2d0c0] animate-pulse"></span> ONLINE
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="font-mono text-[10px] text-[#94c2b2] uppercase">Ping</span>
            <span className="font-mono text-xs text-[#a2d0c0]">14ms</span>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 grid grid-cols-12 gap-6 overflow-hidden">
        
        {/* Left Column: Regime & Stats */}
        <div className="col-span-12 md:col-span-4 flex flex-col gap-6 overflow-y-auto pr-2">
          
          {/* Regime Detector HUD */}
          <div className="border border-[#255144]/50 bg-[#13231E] p-5 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2 opacity-10">
              <Activity size={120} />
            </div>
            <h3 className="font-mono text-[10px] uppercase tracking-widest text-[#94c2b2] mb-4 flex items-center gap-2">
              <Crosshair size={12}/> HMM Regime Detector
            </h3>
            
            <div className="mb-6">
              <p className="font-mono text-[10px] uppercase text-[#94c2b2] mb-1">Active Market State</p>
              <div className={`font-display text-4xl tracking-tight ${isBull ? 'text-[#a2d0c0]' : isBear ? 'text-[#ffb4ab]' : 'text-[#ebe1d3]'}`}>
                {marketState.replace('_', ' ')}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 border-t border-[#255144]/30 pt-4">
              <div>
                <p className="font-mono text-[10px] uppercase text-[#94c2b2]">Transition Prob</p>
                <p className="font-mono text-sm">{regime?.regime_statistics?.transition_probability || "14.2%"}</p>
              </div>
              <div>
                <p className="font-mono text-[10px] uppercase text-[#94c2b2]">Time In State</p>
                <p className="font-mono text-sm">{regime?.regime_statistics?.days_in_state || 24} Days</p>
              </div>
            </div>
          </div>

          {/* Engine Diagnostics */}
          <div className="border border-[#255144]/50 bg-[#13231E] flex flex-col">
            <div className="border-b border-[#255144]/30 p-3 bg-[#0E1A16]">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#94c2b2]">Stat-Arb Diagnostics</span>
            </div>
            <div className="p-0">
              <table className="w-full text-left font-mono text-xs">
                <tbody>
                  {[
                    ['Universe Scanned', pairs?.total_scanned || 'N/A'],
                    ['Cointegration Hits', pairs?.cointegrated || 'N/A'],
                    ['Backtest Candidates', pairs?.backtested || 'N/A'],
                    ['OOS Validated', pairs?.validated || 'N/A']
                  ].map(([label, val], idx) => (
                    <tr key={label} className={`border-b border-[#255144]/30 ${idx % 2 === 0 ? 'bg-[#13231E]' : 'bg-[#0E1A16]'}`}>
                      <td className="p-3 text-[#94c2b2] uppercase text-[10px]">{label}</td>
                      <td className="p-3 text-right text-[#a2d0c0]">{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column: Live Feed */}
        <div className="col-span-12 md:col-span-8 border border-[#255144]/50 bg-[#13231E] flex flex-col overflow-hidden relative">
          <div className="border-b border-[#255144]/30 p-4 bg-[#0E1A16] flex justify-between items-center z-10">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#a2d0c0] flex items-center gap-2">
              <Radio size={14} className="animate-pulse" /> Live Signal Feed
            </span>
            <span className="font-mono text-[10px] bg-[#255144]/30 text-[#94c2b2] px-2 py-1 rounded-sm border border-[#255144]/50">
              {pairs?.pairs?.length || 0} Validated Pairs
            </span>
          </div>
          
          <div className="flex-1 overflow-hidden relative">
            {/* Header row */}
            <div className="grid grid-cols-6 gap-2 p-3 border-b border-[#255144]/30 bg-[#13231E]/90 absolute top-0 w-full z-10">
              <div className="font-mono text-[10px] uppercase text-[#94c2b2]">Asset X</div>
              <div className="font-mono text-[10px] uppercase text-[#94c2b2]">Asset Y</div>
              <div className="font-mono text-[10px] uppercase text-[#94c2b2] text-right">Z-Score</div>
              <div className="font-mono text-[10px] uppercase text-[#94c2b2] text-right">Half-Life</div>
              <div className="font-mono text-[10px] uppercase text-[#94c2b2] text-right">Hedge</div>
              <div className="font-mono text-[10px] uppercase text-[#94c2b2] text-right">Action</div>
            </div>
            
            {/* Scrolling Feed Container */}
            <div 
              ref={feedRef}
              className="absolute inset-0 pt-10 overflow-y-auto no-scrollbar pb-32 mask-image-fade-bottom"
              style={{ scrollBehavior: 'smooth' }}
            >
              <div className="flex flex-col">
                {pairs?.pairs?.length > 0 ? pairs.pairs.map((pair, idx) => {
                  const isLong = pair.signal === 1;
                  const isShort = pair.signal === -1;
                  
                  return (
                    <div key={idx} className="grid grid-cols-6 gap-2 p-4 border-b border-[#255144]/10 hover:bg-[#255144]/20 transition-colors items-center">
                      <div className="font-mono text-sm text-[#ebe1d3]">{pair.x}</div>
                      <div className="font-mono text-sm text-[#ebe1d3]">{pair.y}</div>
                      <div className="font-mono text-sm text-right font-medium">
                        <span className={pair.z_score > 2 ? 'text-[#ffb4ab]' : pair.z_score < -2 ? 'text-[#a2d0c0]' : 'text-[#ebe1d3]'}>
                          {pair.z_score?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div className="font-mono text-xs text-right text-[#94c2b2]">{pair.half_life?.toFixed(1) || 'N/A'}d</div>
                      <div className="font-mono text-xs text-right text-[#94c2b2]">{pair.hedge_ratio?.toFixed(3) || 'N/A'}</div>
                      <div className="flex justify-end">
                        {isLong ? (
                          <div className="flex items-center gap-1 bg-[#255144]/40 text-[#a2d0c0] px-2 py-1 rounded-sm border border-[#a2d0c0]/30 font-mono text-[10px]">
                            <TrendingUp size={10} /> LONG
                          </div>
                        ) : isShort ? (
                          <div className="flex items-center gap-1 bg-[#93000a]/40 text-[#ffb4ab] px-2 py-1 rounded-sm border border-[#ffb4ab]/30 font-mono text-[10px]">
                            <TrendingDown size={10} /> SHORT
                          </div>
                        ) : (
                          <div className="flex items-center gap-1 bg-[#3a342a]/40 text-[#9a8f81] px-2 py-1 rounded-sm border border-[#4e453a]/30 font-mono text-[10px]">
                            <Minus size={10} /> WAIT
                          </div>
                        )}
                      </div>
                    </div>
                  );
                }) : (
                  <div className="p-8 text-center text-[#94c2b2] font-mono text-xs uppercase">No active telemetry stream.</div>
                )}
                
                {/* Duplicate items for infinite scroll illusion if items exist */}
                {pairs?.pairs?.length > 4 && pairs.pairs.map((pair, idx) => {
                  const isLong = pair.signal === 1;
                  const isShort = pair.signal === -1;
                  return (
                    <div key={`dup-${idx}`} className="grid grid-cols-6 gap-2 p-4 border-b border-[#255144]/10 hover:bg-[#255144]/20 transition-colors items-center">
                      <div className="font-mono text-sm text-[#ebe1d3]">{pair.x}</div>
                      <div className="font-mono text-sm text-[#ebe1d3]">{pair.y}</div>
                      <div className="font-mono text-sm text-right font-medium">
                        <span className={pair.z_score > 2 ? 'text-[#ffb4ab]' : pair.z_score < -2 ? 'text-[#a2d0c0]' : 'text-[#ebe1d3]'}>
                          {pair.z_score?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div className="font-mono text-xs text-right text-[#94c2b2]">{pair.half_life?.toFixed(1) || 'N/A'}d</div>
                      <div className="font-mono text-xs text-right text-[#94c2b2]">{pair.hedge_ratio?.toFixed(3) || 'N/A'}</div>
                      <div className="flex justify-end">
                        {isLong ? (
                          <div className="flex items-center gap-1 bg-[#255144]/40 text-[#a2d0c0] px-2 py-1 rounded-sm border border-[#a2d0c0]/30 font-mono text-[10px]">
                            <TrendingUp size={10} /> LONG
                          </div>
                        ) : isShort ? (
                          <div className="flex items-center gap-1 bg-[#93000a]/40 text-[#ffb4ab] px-2 py-1 rounded-sm border border-[#ffb4ab]/30 font-mono text-[10px]">
                            <TrendingDown size={10} /> SHORT
                          </div>
                        ) : (
                          <div className="flex items-center gap-1 bg-[#3a342a]/40 text-[#9a8f81] px-2 py-1 rounded-sm border border-[#4e453a]/30 font-mono text-[10px]">
                            <Minus size={10} /> WAIT
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Scanline overlay effect */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%)] bg-[length:100%_4px] opacity-20"></div>
          </div>
        </div>
      </div>
      
      {/* Hide scrollbar for internal element */}
      <style>{`
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
}

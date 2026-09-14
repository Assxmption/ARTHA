import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, ArrowLeft, Loader2, AlertTriangle, ChevronRight, Activity, TrendingUp, Search } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function Fundamentals() {
  const { symbol } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loadingData, setLoadingData] = useState(true);
  const [narrative, setNarrative] = useState(null);
  const [loadingNarrative, setLoadingNarrative] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!symbol) return;
    
    setLoadingData(true);
    setLoadingNarrative(true);
    setError(null);

    // Fetch tabular data first (fast)
    fetch(`/api/quant/fundamentals/${symbol}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch fundamentals for ${symbol}`);
        return res.json();
      })
      .then(res => {
        setData({ metrics: res.data });
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
      })
      .finally(() => setLoadingData(false));

    // Fetch narrative as SSE stream
    fetch(`/api/quant/fundamentals/${symbol}/narrative_stream`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to start narrative stream for ${symbol}`);
        
        // Hide loading spinner as soon as the connection opens
        setLoadingNarrative(false); 
        
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let streamText = "";
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          streamText += decoder.decode(value, { stream: true });
          // Strip out the [abc12345...] citation tags for cleaner reading
          const cleaned = streamText.replace(/\[[a-f0-9]{8}(?:…|\.\.\.)\]/gi, '');
          setNarrative(cleaned);
        }
      })
      .catch(err => {
        console.error(err);
        setNarrative("## Fundamental Analysis\n\n*Error generating narrative.*");
        setLoadingNarrative(false);
      });
  }, [symbol]);

  // Transform data into an array of years for the table
  const years = data ? Object.keys(data.metrics).sort((a, b) => b - a) : []; // Descending (latest first)
  const metrics = data && years.length > 0 ? Object.keys(data.metrics[years[0]]) : [];

  if (loadingData) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] bg-surface-dim">
        <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
          <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Compiling Financial Statements...</p>
        <p className="font-mono text-[10px] text-outline-variant mt-2">Retrieving row-level data for {symbol}</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 flex flex-col items-center justify-center h-full min-h-[60vh] bg-surface-dim">
        <div className="border border-error/50 bg-error/5 p-6 max-w-md text-center">
          <p className="font-ui text-xs uppercase tracking-widest text-error mb-2">System Error</p>
          <p className="font-mono text-sm text-on-surface">{error}</p>
          <button 
            onClick={() => { setError(null); setLoadingData(true); setLoadingNarrative(true); }} 
            className="mt-6 border border-error px-4 py-2 font-ui text-[10px] uppercase tracking-wider text-error hover:bg-error/10 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface overflow-hidden">
      {/* Editorial Header */}
      <div className="px-12 pt-12 pb-8 border-b border-outline-variant bg-surface-dim flex items-end justify-between">
        <div className="max-w-3xl">
          <button onClick={() => navigate(`/dashboard/${symbol}`)} className="text-outline hover:text-primary transition-colors flex items-center gap-2 mb-6 font-ui text-[10px] uppercase tracking-widest">
            <ArrowLeft size={14} /> Back to Instrument
          </button>
          <h2 className="font-display text-5xl md:text-6xl text-on-surface tracking-tight leading-none mb-4">{symbol}</h2>
          <p className="font-body text-xl text-on-surface-variant max-w-2xl leading-relaxed">
            A comprehensive, row-level breakdown of financial health, operating metrics, and institutional factors driving the instrument's valuation.
          </p>
        </div>
        <div className="hidden md:flex flex-col items-end">
          <span className="font-ui text-[10px] text-outline uppercase tracking-widest mb-1">Last Updated</span>
          <span className="font-mono text-sm text-on-surface">{new Date().toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric'})}</span>
          <div className="flex gap-2 mt-4">
             <span className="px-2 py-1 border border-primary/30 bg-primary/10 text-primary font-ui text-[10px] uppercase tracking-widest">Audited</span>
             <span className="px-2 py-1 border border-outline-variant text-outline font-ui text-[10px] uppercase tracking-widest">FY {years[0]}</span>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-y-auto">
        {/* Narrative Panel (Editorial Top) */}
        <div className="border-b border-outline-variant bg-surface p-12 lg:px-24">
          <div className="flex items-center gap-2 mb-8 pb-4 border-b border-outline-variant max-w-4xl mx-auto">
            <Activity size={16} className="text-primary" />
            <h3 className="font-ui text-xs uppercase tracking-widest text-primary">Intelligence Synthesis</h3>
          </div>
          
          <div className="max-w-4xl mx-auto">
            {loadingNarrative ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 size={24} className="text-primary animate-spin mb-4" />
                <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Synthesizing Fundamental Narrative...</p>
              </div>
            ) : (
              <article className="prose prose-sm md:prose-base dark:prose-invert font-body text-on-surface leading-loose max-w-none prose-p:mb-6">
                <ReactMarkdown
                  components={{
                    h3: ({node, ...props}) => (
                      <h3 className="font-ui uppercase tracking-widest text-primary text-xs mt-10 mb-4 pb-2 border-b border-outline-variant/50" {...props} />
                    ),
                    strong: ({node, ...props}) => (
                      <strong className="text-on-surface font-mono font-medium bg-surface-container-low px-1 py-0.5 rounded" {...props} />
                    )
                  }}
                >
                  {narrative}
                </ReactMarkdown>
                
                <div className="mt-8 p-4 bg-surface-container-low border-l-2 border-primary">
                  <h4 className="font-ui text-[10px] uppercase tracking-widest text-on-surface mb-2 flex items-center gap-2"><TrendingUp size={12}/> Quant Note</h4>
                  <p className="text-xs text-on-surface-variant m-0 font-body">
                    The current print aligns with a statistical regime shift detected in the volatility surface. Margin compression is heavily monitored by the automated engine.
                  </p>
                </div>
              </article>
            )}
          </div>
        </div>

        {/* Multi-year Data Table (Data Grid) */}
        <div className="bg-surface-container-lowest flex flex-col flex-1">
          <div className="p-4 px-12 lg:px-24 border-b border-outline-variant bg-surface flex justify-between items-center">
            <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface flex items-center gap-2">
              <FileText size={14} className="text-outline" />
              Row-Level Ledger (INR Crore)
            </span>
            <div className="flex gap-2">
               <button className="p-1.5 border border-outline-variant text-outline hover:text-primary hover:border-primary transition-colors"><Search size={14}/></button>
            </div>
          </div>
          
          <div className="flex-1 w-full max-w-7xl mx-auto p-0 lg:p-12 overflow-x-auto">
            <table className="w-full text-left font-mono text-sm whitespace-nowrap mb-12">
              <thead className="bg-surface sticky top-0 z-10 shadow-sm">
                <tr>
                  <th scope="col" className="p-4 px-8 border-b border-outline-variant font-normal text-on-surface-variant text-xs uppercase tracking-wider font-ui w-1/4">Metric</th>
                  {years.map(year => (
                    <th scope="col" key={year} className="p-4 px-8 border-b border-outline-variant font-normal text-on-surface-variant text-xs uppercase tracking-wider font-ui text-right">FY {year}</th>
                  ))}
                  <th scope="col" className="p-4 px-8 border-b border-outline-variant font-normal text-outline text-xs uppercase tracking-wider font-ui text-right">Sparkline</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric, idx) => {
                  // Calculate a simple sparkline trend path
                  const values = years.map(y => data.metrics[y][metric]?.value || 0).reverse();
                  const maxVal = Math.max(...values, 1);
                  const minVal = Math.min(...values, 0);
                  const range = maxVal - minVal || 1;
                  const pts = values.map((v, i) => `${(i / (values.length - 1 || 1)) * 40},${15 - ((v - minVal) / range) * 15}`).join(' L ');
                  const pathD = `M ${pts}`;

                  return (
                    <tr key={metric} className={`border-b border-outline-variant/30 hover:bg-surface-dim transition-colors group ${idx % 2 === 0 ? 'bg-surface-container-lowest' : 'bg-surface/30'}`}>
                      <td className="p-4 px-8 text-on-surface font-medium capitalize flex items-center gap-2">
                        {metric.replace(/_/g, ' ')}
                      </td>
                      {years.map(year => {
                        const cell = data.metrics[year][metric];
                        return (
                          <td key={year} className="p-4 px-8 text-right text-on-surface-variant group-hover:text-primary transition-colors">
                            {cell ? (
                              cell.value !== null ? Number(cell.value).toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '-'
                            ) : '-'}
                          </td>
                        );
                      })}
                      <td className="p-4 px-8 text-right flex justify-end items-center h-full">
                         {values.length > 1 ? (
                           <svg width="40" height="15" className="stroke-outline-variant group-hover:stroke-primary fill-none transition-colors" strokeWidth="1.5">
                             <path d={pathD} />
                           </svg>
                         ) : <span className="text-outline">-</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

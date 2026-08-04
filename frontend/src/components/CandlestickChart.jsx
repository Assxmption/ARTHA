import { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { Filter, SlidersHorizontal, MousePointer2, Activity, BarChart2 } from 'lucide-react';

export default function CandlestickChart({ symbol }) {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  
  const [activeInterval, setActiveInterval] = useState('1d');
  const [chartType, setChartType] = useState('candle'); // 'candle' or 'line'
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch data when symbol or interval changes
  useEffect(() => {
    if (!symbol) return;
    
    setLoading(true);
    setError(null);
    
    // Map our UI intervals to yfinance intervals
    // UI: '5m', '15m', '1h', '1d', '1mo', '1y'
    let fetchInterval = activeInterval;
    let daysToFetch = 180;
    
    if (activeInterval === '5m') {
      daysToFetch = 7; // yfinance limit is 60, but 7 is fast and good for 5m
    } else if (activeInterval === '15m') {
      daysToFetch = 14;
    } else if (activeInterval === '1h') {
      fetchInterval = '60m';
      daysToFetch = 60;
    } else if (activeInterval === '1y') {
      fetchInterval = '1d';
      daysToFetch = 365;
    } else if (activeInterval === '1mo') {
      fetchInterval = '1d';
      daysToFetch = 30;
    } else if (activeInterval === 'ytd') {
      fetchInterval = '1d';
      daysToFetch = 365; // approximation, handled via setVisibleRange later
    }

    // Default to '1d' if not one of the intraday ones
    const isIntraday = ['5m', '15m', '60m'].includes(fetchInterval);
    const actualInterval = isIntraday ? fetchInterval : '1d';

    fetch(`/api/quant/ohlcv/${symbol}?days=${daysToFetch}&interval=${actualInterval}`)
      .then(r => r.json())
      .then(res => {
        if (res.data) {
          setData(res.data);
        } else {
          setError("No data returned");
        }
      })
      .catch(e => {
        console.error(e);
        setError("Failed to fetch chart data");
      })
      .finally(() => setLoading(false));
  }, [symbol, activeInterval]);

  // Render chart when data or chartType changes
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // Destroy existing chart if it exists to cleanly swap types
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: 'var(--outline)',
      },
      grid: {
        vertLines: { color: 'var(--outline-variant)' },
        horzLines: { color: 'var(--outline-variant)' },
      },
      timeScale: {
        borderColor: 'var(--outline-variant)',
        timeVisible: true, // Show time for intraday
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'var(--outline-variant)',
      },
      crosshair: {
        mode: 1, // Normal mode
        vertLine: { color: 'var(--primary)', labelBackgroundColor: 'var(--primary)' },
        horzLine: { color: 'var(--primary)', labelBackgroundColor: 'var(--primary)' }
      }
    });
    
    chartRef.current = chart;
    let series;

    if (chartType === 'candle') {
      series = chart.addCandlestickSeries({
        upColor: '#a2d0c0',
        downColor: '#ffb4ab',
        borderVisible: false,
        wickUpColor: '#a2d0c0',
        wickDownColor: '#ffb4ab'
      });
    } else {
      series = chart.addLineSeries({
        color: '#a2d0c0',
        lineWidth: 2,
      });
    }
    
    seriesRef.current = series;

    // Format data and extract signals
    const formattedData = [];
    const markers = [];

    data.forEach((d) => {
      // Deduplicate timestamps (lightweight-charts requires strictly ascending time)
      const time = d.time; 
      
      if (formattedData.length > 0 && formattedData[formattedData.length - 1].time >= time) {
        return; // Skip duplicate or out-of-order
      }

      if (chartType === 'candle') {
        formattedData.push({
          time: time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        });
      } else {
        formattedData.push({
          time: time,
          value: d.close,
        });
      }

      // Add markers if signal exists
      if (d.signal === 1) {
        markers.push({
          time: time,
          position: 'belowBar',
          color: '#a2d0c0',
          shape: 'arrowUp',
          text: 'BUY',
        });
      } else if (d.signal === -1) {
        markers.push({
          time: time,
          position: 'aboveBar',
          color: '#ffb4ab',
          shape: 'arrowDown',
          text: 'SELL',
        });
      }
    });

    try {
      series.setData(formattedData);
      if (markers.length > 0 && typeof series.setMarkers === 'function') {
        series.setMarkers(markers);
      }
      
      // Auto scale
      chart.timeScale().fitContent();
    } catch (e) {
      console.error("Error setting chart data", e);
    }

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, chartType]);

  return (
    <div className="relative border border-outline-variant bg-surface-dim overflow-hidden flex flex-col h-[500px]">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center p-3 border-b border-outline-variant bg-surface-container-low gap-3">
        <div className="flex items-center gap-2">
          <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface flex items-center gap-2">
            <Activity size={12} className={loading ? "animate-spin text-primary" : "text-primary"} /> 
            Live Price Action
          </span>
          {loading && <span className="font-mono text-[9px] text-primary animate-pulse ml-2">Fetching...</span>}
        </div>
        
        <div className="flex flex-wrap gap-4 items-center">
          
          {/* Chart Type Toggle */}
          <div className="flex bg-surface-container rounded-sm p-0.5 border border-outline-variant">
            <button 
              onClick={() => setChartType('line')}
              className={`p-1 rounded-sm transition-colors ${chartType === 'line' ? 'bg-primary text-surface' : 'text-outline hover:text-on-surface'}`}
              title="Line Chart"
            >
              <Activity size={14} />
            </button>
            <button 
              onClick={() => setChartType('candle')}
              className={`p-1 rounded-sm transition-colors ${chartType === 'candle' ? 'bg-primary text-surface' : 'text-outline hover:text-on-surface'}`}
              title="Candlestick Chart"
            >
              <BarChart2 size={14} />
            </button>
          </div>

          {/* Timeframe/Interval Toggle */}
          <div className="flex flex-wrap gap-1">
            {['5m', '15m', '1h', '1d', '1mo', '1y', 'ytd'].map(tf => (
              <button 
                key={tf} 
                onClick={() => setActiveInterval(tf)}
                className={`font-mono text-[10px] px-2 py-1 transition-colors border uppercase ${
                  activeInterval === tf 
                    ? 'text-primary border-primary bg-primary/5' 
                    : 'text-outline hover:text-primary border-transparent hover:border-outline-variant bg-surface'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          <div className="flex gap-2 border-l border-outline-variant pl-4 hidden sm:flex">
            <button className="text-outline hover:text-primary"><Filter size={14}/></button>
            <button className="text-outline hover:text-primary"><SlidersHorizontal size={14}/></button>
            <button className="text-outline hover:text-primary"><MousePointer2 size={14}/></button>
          </div>
        </div>
      </div>
      
      {error ? (
        <div className="flex-1 w-full flex items-center justify-center text-error font-mono text-xs">
          {error}
        </div>
      ) : (
        <div ref={chartContainerRef} className="flex-1 w-full relative" />
      )}
    </div>
  );
}

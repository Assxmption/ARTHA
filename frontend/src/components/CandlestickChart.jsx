import { useEffect, useRef, useState, useMemo } from 'react';
import { createChart } from 'lightweight-charts';
import { Activity, BarChart2, TrendingUp } from 'lucide-react';

// Helper to calculate Simple Moving Average
function calculateSMA(data, count) {
  const result = [];
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i].close;
    if (i >= count) {
      sum -= data[i - count].close;
      result.push({ time: data[i].time, value: sum / count });
    } else if (i === count - 1) {
      result.push({ time: data[i].time, value: sum / count });
    }
  }
  return result;
}

export default function CandlestickChart({ symbol }) {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  
  const [activeInterval, setActiveInterval] = useState('1d');
  const [chartType, setChartType] = useState('candle');
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Indicators State
  const [showSMA, setShowSMA] = useState(false);
  const [showVolume, setShowVolume] = useState(true);

  useEffect(() => {
    if (!symbol) return;
    
    setLoading(true);
    setError(null);
    
    let fetchInterval = activeInterval;
    let daysToFetch = 180;
    
    if (activeInterval === '5m') daysToFetch = 7;
    else if (activeInterval === '15m') daysToFetch = 14;
    else if (activeInterval === '1h') { fetchInterval = '60m'; daysToFetch = 60; }
    else if (activeInterval === '1y') { fetchInterval = '1d'; daysToFetch = 365; }
    else if (activeInterval === '1mo') { fetchInterval = '1d'; daysToFetch = 30; }
    else if (activeInterval === 'ytd') { fetchInterval = '1d'; daysToFetch = 365; }

    const isIntraday = ['5m', '15m', '60m'].includes(fetchInterval);
    const actualInterval = isIntraday ? fetchInterval : '1d';

    fetch(`/api/quant/ohlcv/${symbol}?days=${daysToFetch}&interval=${actualInterval}`)
      .then(r => r.json())
      .then(res => {
        if (res.data) setData(res.data);
        else setError("No data returned");
      })
      .catch(e => {
        console.error(e);
        setError("Failed to fetch chart data");
      })
      .finally(() => setLoading(false));
  }, [symbol, activeInterval]);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: '#e6e1e5',
      },
      grid: {
        vertLines: { color: '#494640' },
        horzLines: { color: '#494640' },
      },
      timeScale: {
        borderColor: '#494640',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: '#494640',
        scaleMargins: {
          top: 0.1,
          bottom: showVolume ? 0.3 : 0.1, // Leave space for volume at bottom
        },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: '#e8c086', labelBackgroundColor: '#e8c086' },
        horzLine: { color: '#e8c086', labelBackgroundColor: '#e8c086' }
      }
    });
    
    chartRef.current = chart;

    let mainSeries;
    if (chartType === 'candle') {
      mainSeries = chart.addCandlestickSeries({
        upColor: '#a2d0c0',
        downColor: '#ffb4ab',
        borderVisible: false,
        wickUpColor: '#a2d0c0',
        wickDownColor: '#ffb4ab'
      });
    } else {
      mainSeries = chart.addLineSeries({
        color: '#a2d0c0',
        lineWidth: 2,
      });
    }

    const formattedData = [];
    const volumeData = [];
    const markers = [];

    data.forEach((d) => {
      const time = d.time; 
      if (formattedData.length > 0 && formattedData[formattedData.length - 1].time >= time) return;

      if (chartType === 'candle') {
        formattedData.push({ time, open: d.open, high: d.high, low: d.low, close: d.close });
      } else {
        formattedData.push({ time, value: d.close });
      }

      if (showVolume && d.volume) {
        const isUp = d.close >= d.open;
        volumeData.push({
          time,
          value: d.volume,
          color: isUp ? 'rgba(162, 208, 192, 0.4)' : 'rgba(255, 180, 171, 0.4)'
        });
      }

      if (d.signal === 1) {
        markers.push({ time, position: 'belowBar', color: '#a2d0c0', shape: 'arrowUp', text: 'BUY' });
      } else if (d.signal === -1) {
        markers.push({ time, position: 'aboveBar', color: '#ffb4ab', shape: 'arrowDown', text: 'SELL' });
      }
    });

    try {
      mainSeries.setData(formattedData);
      if (markers.length > 0 && typeof mainSeries.setMarkers === 'function') {
        mainSeries.setMarkers(markers);
      }

      if (showVolume) {
        const volumeSeries = chart.addHistogramSeries({
          color: '#26a69a',
          priceFormat: { type: 'volume' },
          priceScaleId: '', // set as an overlay
          scaleMargins: {
            top: 0.8, // 20% height from bottom
            bottom: 0,
          },
        });
        volumeSeries.setData(volumeData);
      }

      if (showSMA) {
        const smaData = calculateSMA(data, 20); // 20-period SMA
        const smaSeries = chart.addLineSeries({
          color: '#2962FF',
          lineWidth: 2,
          crosshairMarkerVisible: false,
        });
        smaSeries.setData(smaData);
      }

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
  }, [data, chartType, showSMA, showVolume]);

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

          {/* Indicators Toggles */}
          <div className="flex gap-2 border-l border-outline-variant pl-4">
            <button 
              onClick={() => setShowVolume(!showVolume)}
              className={`font-ui text-[10px] px-2 py-1 border transition-colors ${showVolume ? 'text-primary border-primary bg-primary/5' : 'text-outline border-transparent hover:text-on-surface'}`}
              title="Toggle Volume"
            >
              VOL
            </button>
            <button 
              onClick={() => setShowSMA(!showSMA)}
              className={`font-ui text-[10px] px-2 py-1 border transition-colors flex items-center gap-1 ${showSMA ? 'text-blue-500 border-blue-500 bg-blue-500/10' : 'text-outline border-transparent hover:text-on-surface'}`}
              title="Toggle Moving Average"
            >
              <TrendingUp size={10} /> SMA 20
            </button>
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

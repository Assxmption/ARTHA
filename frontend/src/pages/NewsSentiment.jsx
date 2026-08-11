import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Newspaper, TrendingUp, TrendingDown, Minus, ExternalLink, RefreshCw, ArrowLeft } from 'lucide-react';

const SENTIMENT_COLORS = {
  bullish: { bg: 'bg-secondary/15', text: 'text-secondary', border: 'border-secondary/30', icon: TrendingUp },
  bearish: { bg: 'bg-error/15', text: 'text-error', border: 'border-error/30', icon: TrendingDown },
  neutral: { bg: 'bg-outline/15', text: 'text-outline', border: 'border-outline/30', icon: Minus },
};

export default function NewsSentiment() {
  const { symbol = 'RELIANCE' } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchNews = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/news/${symbol}`);
      if (!res.ok) throw new Error('Failed to fetch news');
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchNews(); }, [symbol]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
          <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Scanning news sources for {symbol}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 flex flex-col items-center justify-center h-full min-h-[60vh]">
        <div className="border border-error/50 bg-error/5 p-6 max-w-md text-center">
          <p className="font-mono text-sm text-error">{error}</p>
          <button onClick={fetchNews} className="mt-4 border border-error px-4 py-2 font-ui text-[10px] uppercase text-error hover:bg-error/10 transition-colors">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const sentiment = data?.sentiment || {};
  const articles = data?.articles || [];
  const sentimentStyle = SENTIMENT_COLORS[sentiment.overall] || SENTIMENT_COLORS.neutral;
  const SentimentIcon = sentimentStyle.icon;
  const dist = sentiment.distribution || {};

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-4">
        <div>
          <button onClick={() => navigate(`/dashboard/${symbol}`)} className="text-outline hover:text-primary transition-colors flex items-center gap-2 mb-3 font-ui text-[10px] uppercase tracking-widest">
            <ArrowLeft size={14} /> Back to Instrument
          </button>
          <h1 className="font-display text-3xl text-on-surface">{symbol} — News & Sentiment</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline mt-1">VADER-scored financial news from {data?.total || 0} sources</p>
        </div>
        <button 
          onClick={fetchNews}
          disabled={loading}
          className="flex items-center gap-2 border border-outline-variant px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container"
        >
          <RefreshCw size={12} /> Refresh
        </button>
      </div>

      {/* Sentiment Summary Strip */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {/* Overall */}
        <div className={`border ${sentimentStyle.border} ${sentimentStyle.bg} p-4 flex items-center gap-4`}>
          <SentimentIcon size={24} className={sentimentStyle.text} />
          <div>
            <span className="font-ui text-[10px] uppercase tracking-widest text-outline block">Overall Sentiment</span>
            <span className={`font-mono text-xl font-bold ${sentimentStyle.text} uppercase`}>{sentiment.overall}</span>
          </div>
        </div>

        {/* Score */}
        <div className="border border-outline-variant bg-surface p-4">
          <span className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-1">Compound Score</span>
          <span className={`font-mono text-2xl font-bold ${sentiment.score >= 0.1 ? 'text-secondary' : sentiment.score <= -0.1 ? 'text-error' : 'text-on-surface'}`}>
            {sentiment.score >= 0 ? '+' : ''}{sentiment.score?.toFixed(3)}
          </span>
        </div>

        {/* Distribution bar */}
        <div className="border border-outline-variant bg-surface p-4 col-span-1 md:col-span-2">
          <span className="font-ui text-[10px] uppercase tracking-widest text-outline block mb-2">Distribution ({data?.total || 0} articles)</span>
          <div className="flex h-3 overflow-hidden rounded-sm">
            {dist.bullish > 0 && <div className="bg-secondary" style={{ width: `${(dist.bullish / data?.total) * 100}%` }}></div>}
            {dist.neutral > 0 && <div className="bg-outline/50" style={{ width: `${(dist.neutral / data?.total) * 100}%` }}></div>}
            {dist.bearish > 0 && <div className="bg-error" style={{ width: `${(dist.bearish / data?.total) * 100}%` }}></div>}
          </div>
          <div className="flex justify-between mt-2 font-mono text-[10px]">
            <span className="text-secondary">🟢 {dist.bullish || 0} bullish</span>
            <span className="text-outline">⚪ {dist.neutral || 0} neutral</span>
            <span className="text-error">🔴 {dist.bearish || 0} bearish</span>
          </div>
        </div>
      </div>

      {/* Articles List */}
      <div className="border border-outline-variant bg-surface">
        <div className="border-b border-outline-variant bg-surface-container-low p-3">
          <span className="font-ui text-[10px] uppercase tracking-widest text-on-surface">Recent Headlines</span>
        </div>
        <div className="divide-y divide-outline-variant/30">
          {articles.map((article, idx) => {
            const artSentiment = article.sentiment || {};
            const artStyle = SENTIMENT_COLORS[artSentiment.label] || SENTIMENT_COLORS.neutral;
            return (
              <div key={idx} className={`p-4 hover:bg-surface-container transition-colors ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                      <span className={`px-1.5 py-0.5 ${artStyle.bg} ${artStyle.text} font-ui text-[9px] uppercase tracking-wider border ${artStyle.border}`}>
                        {artSentiment.label || 'neutral'}
                      </span>
                      <span className="font-mono text-[10px] text-outline">{article.date || 'Unknown date'}</span>
                      {article.source && (
                        <span className="font-ui text-[10px] text-outline-variant">{article.source}</span>
                      )}
                    </div>
                    <a 
                      href={article.link} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="font-body text-sm text-on-surface hover:text-primary transition-colors leading-snug block"
                    >
                      {article.title}
                    </a>
                    {article.description && (
                      <p className="font-body text-xs text-on-surface-variant mt-1 line-clamp-2">{article.description}</p>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <span className={`font-mono text-xs tabular-nums ${artSentiment.compound >= 0.15 ? 'text-secondary' : artSentiment.compound <= -0.15 ? 'text-error' : 'text-outline'}`}>
                      {artSentiment.compound >= 0 ? '+' : ''}{artSentiment.compound?.toFixed(3)}
                    </span>
                    {article.link && (
                      <a href={article.link} target="_blank" rel="noopener noreferrer" className="text-outline hover:text-primary transition-colors">
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

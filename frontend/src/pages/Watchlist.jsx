import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, TrendingUp, TrendingDown, RefreshCw, Plus, Trash2, Edit2, Search, X, Check, List, MoreVertical } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { supabase } from '../lib/supabase';

export default function Watchlist() {
  const navigate = useNavigate();
  
  const { user } = useAuth();
  
  // State
  const [watchlists, setWatchlists] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingLists, setLoadingLists] = useState(true);
  const [error, setError] = useState(null);
  
  // UI State
  const [newSymbol, setNewSymbol] = useState('');
  const [isCreatingList, setIsCreatingList] = useState(false);
  const [newListName, setNewListName] = useState('');
  
  const activeList = watchlists.find(w => w.id === activeId) || watchlists[0];

  // Load Watchlists from Supabase
  useEffect(() => {
    if (!user) return;
    
    const loadWatchlists = async () => {
      setLoadingLists(true);
      const { data, error } = await supabase
        .from('watchlists')
        .select('*')
        .order('created_at', { ascending: true });
        
      if (error) {
        console.error('Error fetching watchlists:', error);
      } else if (data && data.length > 0) {
        setWatchlists(data);
        if (!activeId) setActiveId(data[0].id);
      } else {
        // Seed default watchlists for new user
        const { data: newLists, error: seedError } = await supabase
          .from('watchlists')
          .insert([
            { user_id: user.id, name: 'Nifty 50 Core', symbols: ['^NSEI', 'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL'] },
            { user_id: user.id, name: 'Auto Sector', symbols: ['^CNXAUTO', 'TATAMOTORS', 'M&M', 'MARUTI', 'BAJAJ-AUTO', 'HEROMOTOCO'] },
            { user_id: user.id, name: 'IT Sector', symbols: ['^CNXIT', 'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'] }
          ])
          .select();
        
        if (!seedError && newLists) {
          setWatchlists(newLists);
          setActiveId(newLists[0].id);
        }
      }
      setLoadingLists(false);
    };
    
    loadWatchlists();
  }, [user]);

  // Fetch live data
  const fetchWatchlist = async (isBackground = false) => {
    if (!activeList || activeList.symbols.length === 0) {
      setStocks([]);
      if (!isBackground) setLoading(false);
      return;
    }
    
    if (!isBackground) setLoading(true);
    if (!isBackground) setError(null);
    try {
      const symbolString = encodeURIComponent(activeList.symbols.join(','));
      const res = await fetch(`/api/watchlist?symbols=${symbolString}`);
      if (!res.ok) throw new Error('Failed to fetch watchlist');
      
      const contentType = res.headers.get("content-type");
      if (contentType && contentType.indexOf("application/json") === -1) {
          throw new Error("Backend is not running. Please start the FastAPI server.");
      }
      
      const data = await res.json();
      setStocks(data.stocks || []);
    } catch (err) {
      console.error(err);
      if (!isBackground) setError(err.message);
    } finally {
      if (!isBackground) setLoading(false);
    }
  };

  useEffect(() => { 
    fetchWatchlist(); 
  }, [activeId]);

  // Smart Polling
  useEffect(() => {
    let intervalId;
    
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // Clear any existing just in case
        if (intervalId) clearInterval(intervalId);
        // Fetch immediately on tab return, then resume polling
        fetchWatchlist(true);
        intervalId = setInterval(() => fetchWatchlist(true), 15000);
      } else {
        // Pause polling when tab is hidden
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      }
    };

    // Initial setup
    if (document.visibilityState === 'visible') {
      intervalId = setInterval(() => fetchWatchlist(true), 15000);
    }

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      if (intervalId) clearInterval(intervalId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [activeList?.symbols]); // Re-run effect if symbols change so it polls with new symbols


  // Actions
  const handleCreateList = async () => {
    if (!newListName.trim() || !user) return;
    const name = newListName.trim();
    setIsCreatingList(false);
    setNewListName('');
    
    const { data, error } = await supabase
      .from('watchlists')
      .insert([{ user_id: user.id, name, symbols: [] }])
      .select()
      .single();
      
    if (data && !error) {
      setWatchlists(prev => [...prev, data]);
      setActiveId(data.id);
    }
  };

  const handleDeleteList = async (id) => {
    const filtered = watchlists.filter(w => w.id !== id);
    if (filtered.length === 0) {
      // Must have at least one list, seed a fallback list
      const { data } = await supabase
        .from('watchlists')
        .insert([{ user_id: user.id, name: 'My Watchlist', symbols: [] }])
        .select()
        .single();
      if (data) {
        setWatchlists([data]);
        setActiveId(data.id);
      }
    } else {
      setWatchlists(filtered);
      if (activeId === id) {
        setActiveId(filtered[0].id);
      }
    }
    
    // Delete from Supabase silently in background
    await supabase.from('watchlists').delete().eq('id', id);
  };

  const handleAddSymbol = async (e) => {
    e.preventDefault();
    if (!newSymbol.trim()) return;
    const symbol = newSymbol.trim().toUpperCase();
    setNewSymbol('');
    
    const list = watchlists.find(w => w.id === activeId);
    if (!list || list.symbols.includes(symbol)) return;
    
    const newSymbols = [...list.symbols, symbol];
    
    // Optimistic UI update
    setWatchlists(prev => prev.map(w => w.id === activeId ? { ...w, symbols: newSymbols } : w));
    setTimeout(() => fetchWatchlist(), 50);
    
    // Persist to Supabase
    await supabase.from('watchlists').update({ symbols: newSymbols }).eq('id', activeId);
  };

  const handleRemoveSymbol = async (symbolToRemove) => {
    const list = watchlists.find(w => w.id === activeId);
    if (!list) return;
    
    const newSymbols = list.symbols.filter(s => s !== symbolToRemove);
    
    // Optimistic UI update
    setWatchlists(prev => prev.map(w => w.id === activeId ? { ...w, symbols: newSymbols } : w));
    setStocks(prev => prev.filter(s => s.symbol !== symbolToRemove));
    
    // Persist to Supabase
    await supabase.from('watchlists').update({ symbols: newSymbols }).eq('id', activeId);
  };

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full flex flex-col h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-4">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Watchlist</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Manage Portfolios & Track Live NSE Data</p>
        </div>
        <div className="flex items-center gap-2 w-full md:w-auto">
          <button 
            onClick={fetchWatchlist}
            disabled={loading}
            className="flex-1 md:flex-none flex justify-center items-center gap-2 border border-outline-variant px-4 py-2 font-ui text-[10px] uppercase tracking-wider text-on-surface hover:text-primary hover:border-primary transition-colors bg-surface-container disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 flex-1 h-full min-h-0">
        
        {/* LEFT PANE: Watchlist Groups */}
        <div className="lg:w-64 flex flex-col gap-2 shrink-0">
          <div className="flex justify-between items-center mb-2 px-1">
            <h2 className="font-ui text-xs uppercase tracking-widest text-outline">Your Lists</h2>
            <button 
              onClick={() => setIsCreatingList(!isCreatingList)}
              className="text-on-surface-variant hover:text-primary transition-colors p-1"
            >
              <Plus size={14} />
            </button>
          </div>
          
          {isCreatingList && (
            <div className="mb-4 flex gap-2">
              <input 
                type="text" 
                value={newListName}
                onChange={e => setNewListName(e.target.value)}
                placeholder="List Name..."
                className="flex-1 bg-surface border border-outline-variant px-3 py-2 text-sm text-on-surface focus:outline-none focus:border-primary transition-colors font-mono"
                onKeyDown={e => e.key === 'Enter' && handleCreateList()}
                autoFocus
              />
              <button 
                onClick={handleCreateList}
                className="bg-primary/20 text-primary border border-primary px-3 py-2 hover:bg-primary/30 transition-colors"
              >
                <Check size={14} />
              </button>
              <button 
                onClick={() => setIsCreatingList(false)}
                className="bg-surface border border-outline-variant text-on-surface px-3 py-2 hover:bg-surface-container transition-colors"
              >
                <X size={14} />
              </button>
            </div>
          )}

          <div className="flex flex-row lg:flex-col gap-2 overflow-x-auto lg:overflow-x-visible pb-2 lg:pb-0 hide-scrollbar">
            {watchlists.map(w => (
              <div 
                key={w.id}
                onClick={() => setActiveId(w.id)}
                className={`
                  group flex items-center justify-between px-4 py-3 border cursor-pointer transition-all whitespace-nowrap lg:whitespace-normal shrink-0
                  ${activeId === w.id 
                    ? 'border-primary bg-primary/5 text-primary' 
                    : 'border-outline-variant bg-surface text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <List size={16} className={activeId === w.id ? "text-primary" : "text-outline"} />
                  <span className="font-mono text-sm">{w.name}</span>
                </div>
                {watchlists.length > 1 && activeId === w.id && (
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleDeleteList(w.id); }}
                    className="opacity-0 group-hover:opacity-100 text-error hover:bg-error/20 p-1 rounded transition-all"
                    title="Delete List"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT PANE: Active Watchlist Content */}
        <div className="flex-1 flex flex-col min-h-0 bg-surface border border-outline-variant">
          
          {/* Active List Toolbar */}
          <div className="p-4 border-b border-outline-variant flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-surface-container-low">
            <div>
              <h2 className="font-display text-xl text-on-surface flex items-center gap-2">
                {activeList?.name}
                <span className="font-mono text-xs text-outline px-2 py-0.5 border border-outline-variant bg-surface rounded-full">
                  {activeList?.symbols.length} items
                </span>
              </h2>
            </div>
            
            <form onSubmit={handleAddSymbol} className="relative w-full sm:w-64">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
              <input 
                type="text" 
                value={newSymbol}
                onChange={e => setNewSymbol(e.target.value.toUpperCase())}
                placeholder="Add Symbol (e.g. INFY)"
                className="w-full bg-surface border border-outline-variant pl-9 pr-10 py-2 text-sm text-on-surface focus:outline-none focus:border-primary transition-colors font-mono uppercase"
              />
              <button 
                type="submit"
                disabled={!newSymbol.trim()}
                className="absolute right-1 top-1 bottom-1 px-2 flex items-center justify-center bg-primary/20 text-primary hover:bg-primary/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Plus size={14} />
              </button>
            </form>
          </div>

          {/* List Content */}
          <div className="flex-1 overflow-auto">
            {loading && (
              <div className="flex flex-col items-center justify-center h-64">
                <div className="w-48 h-[1px] bg-surface-container relative overflow-hidden mb-4">
                  <div className="absolute top-0 left-0 h-full bg-primary w-1/3 animate-[marquee_1.5s_ease-in-out_infinite]"></div>
                </div>
                <p className="font-mono text-[10px] uppercase tracking-widest text-outline animate-pulse">Fetching live market data...</p>
              </div>
            )}

            {error && !loading && (
              <div className="m-6 border border-error/50 bg-error/5 p-6 text-center">
                <p className="font-mono text-sm text-error">{error}</p>
                <button onClick={fetchWatchlist} className="mt-4 border border-error px-4 py-2 font-ui text-[10px] uppercase tracking-wider text-error hover:bg-error/10 transition-colors">
                  Retry
                </button>
              </div>
            )}

            {!loading && !error && activeList?.symbols.length === 0 && (
              <div className="flex flex-col items-center justify-center h-64 text-center p-6">
                <div className="w-16 h-16 rounded-full bg-surface-container flex items-center justify-center text-outline-variant mb-4">
                  <List size={24} />
                </div>
                <h3 className="font-display text-xl text-on-surface mb-2">Watchlist is Empty</h3>
                <p className="font-mono text-xs text-outline max-w-sm">
                  Use the search bar above to add NSE symbols to '{activeList.name}'.
                </p>
              </div>
            )}

            {!loading && !error && activeList?.symbols.length > 0 && (
              <table className="w-full text-left font-mono text-sm">
                <thead className="sticky top-0 bg-surface z-10 shadow-sm shadow-black/50">
                  <tr className="border-b border-outline-variant text-outline">
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider">Symbol</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider hidden sm:table-cell">Name</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">LTP</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right">Change</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right hidden lg:table-cell">Vol</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right hidden xl:table-cell">Mkt Cap</th>
                    <th className="font-ui text-[10px] uppercase p-4 font-normal tracking-wider text-right"></th>
                  </tr>
                </thead>
                <tbody>
                  {stocks.map((item, idx) => {
                    if (item.error) {
                      return (
                         <tr key={item.symbol} className={`border-b border-outline-variant/30 ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                           <td className="p-4 font-bold text-on-surface">{item.symbol}</td>
                           <td colSpan="5" className="p-4 text-error text-xs">Failed to fetch data. Invalid symbol?</td>
                           <td className="p-4 text-right">
                             <button onClick={() => handleRemoveSymbol(item.symbol)} className="text-outline hover:text-error transition-colors p-1"><Trash2 size={14}/></button>
                           </td>
                         </tr>
                      );
                    }
                    const isUp = item.changePct >= 0;
                    return (
                      <tr key={item.symbol} className={`border-b border-outline-variant/30 hover:bg-surface-container transition-colors ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-container-lowest'}`}>
                        <td className="p-4">
                          <button 
                            onClick={() => navigate(`/dashboard/${item.symbol}`)}
                            className="font-bold text-on-surface hover:text-primary transition-colors flex items-center gap-2"
                          >
                            {item.symbol}
                          </button>
                        </td>
                        <td className="p-4 text-on-surface-variant text-xs hidden sm:table-cell truncate max-w-[150px]">{item.name}</td>
                        <td className="p-4 text-right text-on-surface tabular-nums">₹{item.price?.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
                        <td className={`p-4 text-right tabular-nums ${isUp ? 'text-secondary' : 'text-error'}`}>
                          <span className="flex items-center justify-end gap-1">
                            {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                            {isUp ? '+' : ''}{item.changePct}%
                          </span>
                        </td>
                        <td className="p-4 text-right text-on-surface-variant text-xs tabular-nums hidden lg:table-cell">{item.volumeFormatted || '—'}</td>
                        <td className="p-4 text-right text-on-surface-variant text-xs tabular-nums hidden xl:table-cell">{item.marketCap}</td>
                        <td className="p-4 text-right w-24">
                          <div className="flex items-center justify-end gap-2 opacity-0 hover:opacity-100 transition-opacity" style={{ opacity: 1 }}>
                            <button 
                              onClick={() => navigate(`/dashboard/${item.symbol}`)}
                              className="text-on-surface-variant hover:text-primary transition-colors p-1"
                              title="Analyze"
                            >
                              <Eye size={14} />
                            </button>
                            <button 
                              onClick={() => handleRemoveSymbol(item.symbol)}
                              className="text-on-surface-variant hover:text-error transition-colors p-1"
                              title="Remove Symbol"
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

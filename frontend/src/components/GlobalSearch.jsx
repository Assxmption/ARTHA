import { useEffect, useId, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { setActiveSymbol } from '../lib/activeSymbol';
import { Search, Clock, X } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Instrument search. Results come from /api/search: a local ranked index over
// ARTHA's universe (with former tickers, e.g. ZOMATO → ETERNAL) and Kite's
// NSE instrument master. Keyboard: "/" focuses, ↑/↓ move, Enter opens, Esc closes.

const RECENT_KEY = 'artha.recentSearches';
const MAX_RECENT = 6;

function loadRecent() {
  try {
    const v = JSON.parse(window.localStorage.getItem(RECENT_KEY) || '[]');
    return Array.isArray(v) ? v.filter((x) => x && typeof x.symbol === 'string').slice(0, MAX_RECENT) : [];
  } catch {
    return [];
  }
}

function saveRecent(item) {
  const next = [
    { symbol: item.symbol, name: item.name ?? '', type: item.type ?? 'EQ' },
    ...loadRecent().filter((r) => r.symbol !== item.symbol),
  ].slice(0, MAX_RECENT);
  try { window.localStorage.setItem(RECENT_KEY, JSON.stringify(next)); } catch { /* storage unavailable */ }
  return next;
}

function Highlight({ text, query }) {
  const i = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  if (i < 0) return text;
  return (
    <>
      {text.slice(0, i)}
      <span className="text-primary font-bold">{text.slice(i, i + query.length)}</span>
      {text.slice(i + query.length)}
    </>
  );
}

export default function GlobalSearch() {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState('');   // query the current results belong to
  const [active, setActive] = useState(0);
  const [recent, setRecent] = useState(loadRecent);
  const navigate = useNavigate();
  const wrapperRef = useRef(null);
  const inputRef = useRef(null);
  const listId = useId();

  const q = query.trim();
  const showingRecent = open && !q;
  const items = showingRecent ? recent : results;

  // Debounced search; an in-flight request is aborted when the query changes,
  // so a slow early response can never overwrite a newer one.
  useEffect(() => {
    if (!q) { setResults([]); setSearched(''); setLoading(false); return undefined; }
    const ctrl = new AbortController();
    setLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await apiFetch(`/api/search?q=${encodeURIComponent(q)}&limit=8`, { signal: ctrl.signal });
        const data = await res.json();
        setResults(data.results || []);
        setSearched(q);
        setActive(0);
      } catch (e) {
        if (e.name !== 'AbortError') { setResults([]); setSearched(q); }
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 120);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [q]);

  useEffect(() => {
    const onDown = (e) => { if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false); };
    // "/" focuses search from anywhere except while typing in another field.
    const onKey = (e) => {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || document.activeElement?.isContentEditable) return;
      e.preventDefault();
      inputRef.current?.focus();
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  const select = (item) => {
    setRecent(saveRecent(item));
    setActiveSymbol(item.symbol);
    navigate(`/dashboard/${encodeURIComponent(item.symbol)}`);
    setQuery('');
    setOpen(false);
    inputRef.current?.blur();
  };

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setActive((i) => Math.min(i + 1, Math.max(items.length - 1, 0))); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
    else if (e.key === 'Escape') { setOpen(false); inputRef.current?.blur(); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      // Enter opens the highlighted result (so "zomato" opens ETERNAL); raw text
      // is only used when the search has finished and found nothing.
      if (items[active]) select(items[active]);
      else if (q && !loading && searched === q) select({ symbol: q.toUpperCase() });
    }
  };

  const clearRecent = () => {
    try { window.localStorage.removeItem(RECENT_KEY); } catch { /* ignore */ }
    setRecent([]);
  };

  const noMatches = open && q && !loading && searched === q && results.length === 0;

  return (
    <div ref={wrapperRef} className="relative w-full sm:w-72">
      <div className="relative w-full">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={open && (items.length > 0 || !!noMatches)}
          aria-controls={listId}
          aria-activedescendant={items[active] ? `${listId}-${active}` : undefined}
          aria-autocomplete="list"
          aria-label="Search instruments"
          placeholder="Search stocks, indices…"
          autoComplete="off"
          spellCheck={false}
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0); }}
          onFocus={() => { setOpen(true); setActive(0); }}
          onKeyDown={onKeyDown}
          className="bg-surface border border-outline-variant pl-9 pr-14 py-1.5 text-xs font-mono text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary w-full rounded-none transition-colors duration-300"
        />
        <span className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          {loading && <span className="w-3 h-3 border border-outline border-t-primary rounded-full animate-spin" aria-hidden="true" />}
          {query ? (
            <button onClick={() => { setQuery(''); inputRef.current?.focus(); }} className="text-outline hover:text-on-surface" aria-label="Clear search"><X size={12} /></button>
          ) : (
            <kbd className="hidden sm:inline font-mono text-[9px] text-outline border border-outline-variant px-1">/</kbd>
          )}
        </span>
      </div>

      {open && (items.length > 0 || noMatches) && (
        <div className="absolute top-full left-0 right-0 sm:left-auto sm:w-96 mt-1 bg-surface border border-outline-variant shadow-lg z-50">
          {showingRecent && (
            <div className="flex justify-between items-center px-3 pt-2 pb-1">
              <span className="font-ui text-[9px] uppercase tracking-widest text-outline flex items-center gap-1"><Clock size={10} /> Recent</span>
              <button onMouseDown={(e) => { e.preventDefault(); clearRecent(); }} className="font-ui text-[9px] uppercase tracking-widest text-outline hover:text-error">Clear</button>
            </div>
          )}
          {noMatches ? (
            <p className="px-4 py-3 font-mono text-[11px] text-outline">No NSE instrument matches “{q}”.</p>
          ) : (
            <ul id={listId} role="listbox" className="py-1 max-h-80 overflow-y-auto">
              {items.map((item, i) => (
                <li key={`${item.symbol}-${i}`} id={`${listId}-${i}`} role="option" aria-selected={i === active}>
                  <button
                    onMouseDown={(e) => { e.preventDefault(); select(item); }}
                    onMouseEnter={() => setActive(i)}
                    className={`w-full text-left px-3 py-2 flex justify-between items-center gap-3 transition-colors ${i === active ? 'bg-surface-container' : ''}`}
                  >
                    <div className="flex flex-col min-w-0">
                      <span className="font-mono text-sm text-on-surface flex items-center gap-2">
                        <Highlight text={item.symbol} query={showingRecent ? '' : q} />
                        {item.matched_former && (
                          <span className="font-mono text-[9px] text-primary border border-primary/50 px-1 whitespace-nowrap">formerly {item.matched_former}</span>
                        )}
                      </span>
                      <span className="font-ui text-[10px] text-on-surface-variant truncate">
                        <Highlight text={item.name || ''} query={showingRecent ? '' : q} />
                      </span>
                    </div>
                    <span className="flex items-center gap-1.5 shrink-0">
                      {item.in_universe && <span className="text-[9px] text-secondary uppercase font-ui tracking-widest" title="In ARTHA's analysed universe">Artha</span>}
                      <span className="text-[9px] text-outline-variant uppercase font-ui tracking-widest">{item.type === 'INDEX' ? 'Index' : item.exchange || 'NSE'}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {!showingRecent && !noMatches && (
            <p className="px-3 py-1.5 border-t border-outline-variant/50 font-mono text-[9px] text-outline">↑↓ to move · Enter to open · Esc to close</p>
          )}
        </div>
      )}
    </div>
  );
}

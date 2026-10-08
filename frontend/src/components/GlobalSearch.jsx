import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';

export default function GlobalSearch() {
  const [search, setSearch] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [results, setResults] = useState([]);
  const navigate = useNavigate();
  const wrapperRef = useRef(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (search.trim().length >= 1) {
        try {
          const res = await fetch(`/api/search?q=${search.trim()}`);
          const data = await res.json();
          setResults(data.results || []);
        } catch (e) {
          console.error("Search fetch failed", e);
        }
      } else {
        setResults([]);
      }
    };
    
    // Simple debounce
    const timeoutId = setTimeout(fetchResults, 150);
    return () => clearTimeout(timeoutId);
  }, [search]);

  // Click outside listener
  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsFocused(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (sym) => {
    navigate(`/dashboard/${sym}`);
    setSearch('');
    setIsFocused(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (search.trim()) {
      navigate(`/dashboard/${search.trim().toUpperCase()}`);
      setSearch('');
      setIsFocused(false);
    }
  };

  return (
    <div ref={wrapperRef} className="relative w-full sm:w-64">
      <form onSubmit={handleSubmit} className="relative w-full">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
        <input 
          type="text" 
          placeholder="Search ticker (e.g. RELIANCE)..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onFocus={() => setIsFocused(true)}
          className="bg-surface border border-outline-variant pl-9 pr-4 py-1.5 text-xs font-mono text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary w-full rounded-none transition-colors duration-300"
        />
      </form>

      {/* Autocomplete Dropdown */}
      {isFocused && search.trim() && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-surface border border-outline-variant shadow-lg z-50 max-h-64 overflow-y-auto">
          <ul className="py-1">
            {results.map((item) => {
              const sym = item.symbol;
              const name = item.name;
              
              // Highlight matching part of symbol
              const matchIndex = sym.toLowerCase().indexOf(search.toLowerCase());
              
              return (
                <li key={sym}>
                  <button
                    onMouseDown={(e) => { e.preventDefault(); handleSelect(sym); }}
                    className="w-full text-left px-4 py-2 hover:bg-surface-container transition-colors focus:bg-surface-container focus:outline-none flex justify-between items-center"
                  >
                    <div className="flex flex-col">
                      <span className="font-mono text-sm text-on-surface">
                        {matchIndex >= 0 ? (
                          <>
                            {sym.substring(0, matchIndex)}
                            <span className="text-primary font-bold">{sym.substring(matchIndex, matchIndex + search.length)}</span>
                            {sym.substring(matchIndex + search.length)}
                          </>
                        ) : (
                          sym
                        )}
                      </span>
                      <span className="font-ui text-[10px] text-on-surface-variant truncate max-w-[150px]">{name}</span>
                    </div>
                    <span className="text-[9px] text-outline-variant uppercase font-ui tracking-widest">{item.exchange}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

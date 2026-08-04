import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';

export default function GlobalSearch() {
  const [search, setSearch] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [results, setResults] = useState([]);
  const navigate = useNavigate();
  const wrapperRef = useRef(null);

  // Mock list of symbols for autocomplete
  const popularSymbols = [
    'RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ITC', 
    'SBI', 'BHARTIARTL', 'KOTAKBANK', 'ICICIBANK',
    'L&T', 'BAJFINANCE', 'AXISBANK', 'ASIANPAINT',
    'MARUTI', 'HCLTECH', 'SUNPHARMA', 'TATASTEEL',
    'GOLD', 'SILVER', 'CRUDEOIL', 'COPPER', 'NATURALGAS'
  ];

  useEffect(() => {
    if (search.trim()) {
      const filtered = popularSymbols.filter(sym => 
        sym.toLowerCase().includes(search.toLowerCase())
      );
      setResults(filtered.slice(0, 5)); // show top 5
    } else {
      setResults([]);
    }
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
        <div className="absolute top-full left-0 right-0 mt-1 bg-surface border border-outline-variant shadow-lg z-50">
          <ul className="py-1">
            {results.map((sym) => (
              <li key={sym}>
                <button
                  onMouseDown={(e) => { e.preventDefault(); handleSelect(sym); }}
                  className="w-full text-left px-4 py-2 font-mono text-xs hover:bg-surface-container transition-colors focus:bg-surface-container focus:outline-none flex justify-between items-center"
                >
                  <span className="text-on-surface">
                    {sym.substring(0, sym.toLowerCase().indexOf(search.toLowerCase()))}
                    <span className="text-primary font-bold">{sym.substring(sym.toLowerCase().indexOf(search.toLowerCase()), sym.toLowerCase().indexOf(search.toLowerCase()) + search.length)}</span>
                    {sym.substring(sym.toLowerCase().indexOf(search.toLowerCase()) + search.length)}
                  </span>
                  <span className="text-[9px] text-outline-variant uppercase font-ui tracking-widest">NSE</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sun, Moon, Search, Settings as SettingsIcon, User, Menu, X } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import GlobalSearch from './GlobalSearch';
import { prefersReducedMotion, usePreferences } from '../lib/preferences';
import { apiFetch } from '../lib/api';

export default function TopBar({ onMenuClick, mobileOpen, desktopOpen }) {
  const { isDark, toggleTheme } = useTheme();
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [profileOpen, setProfileOpen] = useState(false);
  
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const isMenuOpen = isMobile ? mobileOpen : desktopOpen;

  // Live header data (replaces the old hard-coded marquee).
  const prefs = usePreferences();
  const [ticker, setTicker] = useState(null);
  const [movers, setMovers] = useState(null);
  const [regime, setRegime] = useState(null);
  const [validatedPairs, setValidatedPairs] = useState([]);
  const wantStocks = prefs.tickerEnabled && prefs.tickerContent === 'all';
  useEffect(() => {
    if (!prefs.tickerEnabled) return undefined;
    let stop = false;
    let timer;
    const load = async () => {
      try {
        const [t, m] = await Promise.all([
          apiFetch('/api/quant/ticker').then((r) => (r.ok ? r.json() : null)),
          wantStocks ? apiFetch('/api/quant/movers?top=10').then((r) => (r.ok ? r.json() : null)) : null,
        ]);
        if (stop) return;
        if (t) setTicker(t);
        setMovers(m);
        timer = setTimeout(load, t?.market_open ? 15000 : 300000);
      } catch {
        if (!stop) timer = setTimeout(load, 60000);
      }
    };
    load();
    return () => { stop = true; clearTimeout(timer); };
  }, [prefs.tickerEnabled, wantStocks]);
  useEffect(() => {
    let stop = false;
    apiFetch('/api/quant/regime').then((r) => (r.ok ? r.json() : null)).then((r) => { if (!stop && r) setRegime(r); }).catch(() => {});
    apiFetch('/api/quant/pairs').then((r) => (r.ok ? r.json() : null))
      .then((p) => { if (!stop && p?.pairs) setValidatedPairs(p.pairs.filter((x) => x.validated)); }).catch(() => {});
    return () => { stop = true; };
  }, []);

  // Constant pixel speed regardless of how many items are in the track.
  const trackRef = useRef(null);
  const [duration, setDuration] = useState(120);
  const pxPerSec = { slow: 30, normal: 55, fast: 95 }[prefs.tickerSpeed] ?? 55;
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return undefined;
    const measure = () => setDuration(Math.max(20, el.scrollWidth / 2 / pxPerSec));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [pxPerSec, ticker, movers, validatedPairs]);
  const scrolling = prefs.tickerScroll && !prefersReducedMotion();

  const tickerItems = [
    ...(ticker?.items ?? []).map((it) => ({ key: `i:${it.label}`, label: it.label, ltp: it.ltp, pct: it.change_pct, index: true })),
    ...validatedPairs.map((p) => ({ key: `p:${p.pair}`, pair: p })),
    ...(wantStocks ? (movers?.rows ?? []) : [])
      .slice().sort((a, b) => a.symbol.localeCompare(b.symbol))
      .map((r) => ({ key: `s:${r.symbol}`, label: r.symbol, ltp: r.ltp, pct: r.change_pct, symbol: r.symbol })),
  ];
  const renderItem = (it, dup) => {
    if (it.pair) {
      return (
        <span key={`${it.key}${dup}`} className="text-on-surface-variant" title="Pair that passed the out-of-sample gate; spread state, not a recommendation">
          <span className="text-primary font-bold">PAIR</span> {it.pair.pair} z {fmt(it.pair.z_score)}
        </span>
      );
    }
    const up = (it.pct ?? 0) >= 0;
    const tone = it.pct == null ? 'text-outline' : up ? 'text-secondary' : 'text-error';
    const body = (
      <>
        <span className={`${it.index ? 'text-on-surface font-semibold' : 'text-on-surface-variant'}`}>{it.label}</span>{' '}
        <span className="text-on-surface-variant">{fmt(it.ltp)}</span>{' '}
        <span className={tone}>{it.pct == null ? '—' : `${up ? '▲' : '▼'} ${up ? '+' : ''}${fmt(it.pct)}%`}</span>
      </>
    );
    return it.symbol ? (
      <button key={`${it.key}${dup}`} tabIndex={dup ? -1 : 0} onClick={() => navigate(`/dashboard/${it.symbol}`)}
        className="hover:underline focus:outline-none focus-visible:underline">{body}</button>
    ) : <span key={`${it.key}${dup}`}>{body}</span>;
  };
  const regimeTone = {
    BULL: { dot: 'bg-secondary' }, BEAR: { dot: 'bg-error' }, SIDEWAYS: { dot: 'bg-primary' },
  }[regime?.current_regime] ?? { dot: 'bg-outline' };

  return (
    <header className="h-14 border-b border-outline-variant bg-surface-container-low flex items-center px-4 md:px-6 sticky top-0 z-40 transition-colors duration-300 gap-4 md:gap-6">
      
      {/* LEFT COMPONENT */}
      <div className="flex items-center gap-4 md:gap-6 shrink-0">
        {/* Menu Button with smooth transition */}
        <button 
          onClick={onMenuClick}
          className="text-on-surface hover:text-primary transition-colors focus:outline-none relative w-6 h-6 flex items-center justify-center overflow-hidden"
          aria-label="Toggle Menu"
        >
          <div className={`absolute transition-all duration-300 ease-in-out ${isMenuOpen ? 'opacity-0 rotate-90 scale-50' : 'opacity-100 rotate-0 scale-100'}`}>
            <Menu size={20} />
          </div>
          <div className={`absolute transition-all duration-300 ease-in-out ${isMenuOpen ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-50'}`}>
            <X size={20} />
          </div>
        </button>

        <div className="hidden lg:flex items-center gap-2 shrink-0" title={regime?.as_of ? `HMM regime on NIFTY 50 · as of ${regime.as_of}` : ''}>
          <div className={`w-2 h-2 rounded-full ${regimeTone.dot}`}></div>
          <span className="font-ui text-xs uppercase tracking-widest text-on-surface-variant whitespace-nowrap">
            Regime: {regime?.current_regime ?? '—'}
          </span>
        </div>
      </div>

      {/* MIDDLE COMPONENT (TICKER) — real data only */}
      <div className="hidden md:flex flex-1 min-w-0 h-full items-center border-l border-r border-outline-variant/30 font-mono text-xs">
        {prefs.tickerEnabled ? (
          <>
            <div className="ticker-wrap flex-1 min-w-0 overflow-hidden h-full flex items-center mask-image-fade" aria-label="Live market ticker">
              <div
                ref={trackRef}
                className={`flex w-max whitespace-nowrap ${scrolling ? 'ticker-track' : ''}`}
                style={{ '--ticker-duration': `${duration}s` }}
              >
                {[0, 1].map((dup) => (
                  <div key={dup} className="flex items-center gap-6 pr-6" aria-hidden={dup === 1 || undefined}>
                    {tickerItems.map((it) => renderItem(it, dup ? ':dup' : ''))}
                  </div>
                )).slice(0, scrolling ? 2 : 1)}
              </div>
            </div>
            {ticker && (
              <span className="shrink-0 px-3 text-[10px] text-outline border-l border-outline-variant/30 h-full flex items-center" title={ticker.source ? `Source: ${ticker.source}` : ''}>
                {ticker.live ? 'KITE' : 'DELAYED'} · {ticker.market_open ? 'OPEN' : 'CLOSED'}
              </span>
            )}
          </>
        ) : <span className="flex-1" />}
      </div>

      {/* RIGHT COMPONENT */}
      <div className="flex items-center gap-4 md:gap-6 shrink-0 ml-auto">
        
        <GlobalSearch />
        
        {/* Premium Theme Toggle Switch */}
        <button 
          onClick={toggleTheme} 
          aria-label="Toggle theme"
          className="relative hidden sm:flex items-center justify-between w-12 h-6 rounded-full bg-surface-variant p-1 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary transition-colors shadow-inner shrink-0"
        >
          <div className={`absolute top-1 left-1 bg-primary w-4 h-4 rounded-full shadow-md transform transition-transform duration-300 ease-out ${isDark ? 'translate-x-6' : 'translate-x-0'}`}></div>
          <div className="w-full flex justify-between px-1 z-10 pointer-events-none text-on-surface-variant">
             <Sun size={10} className={`transition-colors duration-300 ${!isDark ? 'text-on-primary' : 'opacity-50'}`} />
             <Moon size={10} className={`transition-colors duration-300 ${isDark ? 'text-on-primary' : 'opacity-50'}`} />
          </div>
        </button>
        
        {/* User Profile / Auth */}
        {user ? (
          <div className="relative">
            <button 
              onClick={() => setProfileOpen(!profileOpen)}
              className="w-7 h-7 rounded-full bg-primary/20 border border-primary flex items-center justify-center text-primary hover:bg-primary/30 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              aria-label="User menu"
            >
              <User size={14} />
            </button>
            {profileOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-surface border border-outline-variant shadow-lg z-50">
                <div className="p-3 border-b border-outline-variant/50">
                  <p className="font-ui text-[10px] uppercase tracking-widest text-outline">Signed in as</p>
                  <p className="font-mono text-xs text-on-surface truncate">{user.email}</p>
                </div>
                <button 
                  onClick={() => { navigate('/settings'); setProfileOpen(false); }}
                  className="w-full flex items-center gap-2 text-left p-3 font-ui text-xs uppercase tracking-widest text-on-surface hover:bg-surface-dim transition-colors"
                >
                  <SettingsIcon size={12} /> Settings
                </button>
                <button 
                  onClick={async () => { setProfileOpen(false); await signOut(); navigate('/auth'); }}
                  className="w-full text-left p-3 font-ui text-xs uppercase tracking-widest text-error hover:bg-surface-dim transition-colors border-t border-outline-variant/50"
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
        ) : (
          <button 
            onClick={() => navigate('/auth')}
            className="hidden sm:block border border-primary px-3 py-1 font-ui text-[10px] uppercase tracking-wider text-primary hover:bg-primary/10 transition-colors shrink-0"
          >
            Sign In
          </button>
        )}
      </div>
    </header>
  );
}

function fmt(v) {
  return typeof v === 'number' && Number.isFinite(v)
    ? v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '—';
}

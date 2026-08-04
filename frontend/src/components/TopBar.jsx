import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sun, Moon, Search, Settings as SettingsIcon, User, Menu } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import GlobalSearch from './GlobalSearch';

export default function TopBar({ onMenuClick }) {
  const { isDark, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [profileOpen, setProfileOpen] = useState(false);

  return (
    <header className="h-14 border-b border-outline-variant bg-surface-container-low flex items-center justify-between px-4 md:px-6 sticky top-0 z-40 transition-colors duration-300">
      <div className="flex items-center gap-4 md:gap-6 w-full md:w-auto">
        {/* Mobile Menu Button */}
        <button 
          onClick={onMenuClick}
          className="md:hidden text-on-surface hover:text-primary transition-colors focus:outline-none"
        >
          <Menu size={20} />
        </button>

        <div className="hidden lg:flex items-center gap-2 w-32 shrink-0">
          <div className="w-2 h-2 rounded-full bg-secondary"></div>
          <span className="font-ui text-xs uppercase tracking-widest text-on-surface-variant whitespace-nowrap">Regime: Calm</span>
        </div>

        {/* Premium Scrolling Ticker Tape (Hidden on Mobile for Space) */}
        <div className="hidden md:flex flex-1 overflow-hidden h-full items-center border-l border-r border-outline-variant/30 mx-4 px-4 mask-image-fade" style={{ width: '400px' }}>
          <div className="whitespace-nowrap animate-marquee flex gap-12 font-mono text-xs items-center">
            <span className="text-on-surface-variant"><span className="text-secondary mr-1">▲</span>NIFTY 50 24,102.50 (+0.4%)</span>
            <span className="text-on-surface-variant"><span className="text-secondary mr-1">▲</span>SENSEX 79,340.20 (+0.3%)</span>
            <span className="text-on-surface-variant"><span className="text-error mr-1">▼</span>BANKNIFTY 51,200.15 (-0.1%)</span>
            <span className="text-on-surface-variant border-l border-outline-variant/50 pl-4"><span className="text-primary font-bold">SIGNAL:</span> LONG RELIANCE / SHORT INFY (Z: -2.1)</span>
            <span className="text-on-surface-variant border-l border-outline-variant/50 pl-4"><span className="text-primary font-bold">REGIME:</span> BULL VOLATILE</span>
            {/* Duplicate for infinite loop */}
            <span className="text-on-surface-variant"><span className="text-secondary mr-1">▲</span>NIFTY 50 24,102.50 (+0.4%)</span>
            <span className="text-on-surface-variant"><span className="text-secondary mr-1">▲</span>SENSEX 79,340.20 (+0.3%)</span>
            <span className="text-on-surface-variant"><span className="text-error mr-1">▼</span>BANKNIFTY 51,200.15 (-0.1%)</span>
            <span className="text-on-surface-variant border-l border-outline-variant/50 pl-4"><span className="text-primary font-bold">SIGNAL:</span> LONG RELIANCE / SHORT INFY (Z: -2.1)</span>
          </div>
        </div>
      </div>

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
                  onClick={() => { logout(); setProfileOpen(false); }}
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

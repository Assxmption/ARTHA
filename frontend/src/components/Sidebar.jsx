import { useNavigate, useLocation } from 'react-router-dom';
import { Home, Eye, Activity, Database, FileText } from 'lucide-react';

export default function Sidebar({ isOpen, setIsOpen }) {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { label: 'Overview', path: '/home', icon: <Home size={16} /> },
    { label: 'Watchlist', path: '/watchlist', icon: <Eye size={16} /> },
    { label: 'Quant Signals', path: '/signals', icon: <Activity size={16} /> },
    { label: 'Fundamentals', path: '/fundamentals/RELIANCE', icon: <FileText size={16} /> },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-background/80 z-40 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <aside className={`
        fixed md:static inset-y-0 left-0 z-50
        w-64 border-r border-outline-variant bg-surface-dim flex flex-col py-6 overflow-y-auto transition-transform duration-300
        ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <div className="px-6 mb-8 flex justify-between items-center">
          <h1 
            className="font-display text-3xl tracking-tighter text-primary cursor-pointer transition-colors" 
            onClick={() => { navigate('/home'); setIsOpen(false); }}
          >
            ARTHA
          </h1>
        </div>
        
        <nav className="px-4 flex-1">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const isActive = location.pathname.startsWith(item.path.split('/')[1] ? `/${item.path.split('/')[1]}` : item.path);
              return (
                <li key={item.label}>
                  <button 
                    onClick={() => { navigate(item.path); setIsOpen(false); }}
                    className={`
                      w-full flex items-center gap-3 px-4 py-3 text-sm font-ui uppercase tracking-widest transition-colors rounded-sm
                      ${isActive 
                        ? 'bg-primary/10 text-primary border-l-2 border-primary' 
                        : 'text-outline hover:bg-surface hover:text-on-surface border-l-2 border-transparent'
                      }
                    `}
                  >
                    {item.icon}
                    {item.label}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="px-6 mt-8 border-t border-outline-variant pt-6">
          <div className="flex items-center gap-2 text-outline-variant font-mono text-[10px] mb-2">
            <Database size={12} /> Live Engine Active
          </div>
        </div>
      </aside>
    </>
  );
}

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sun, Moon, ShieldCheck, Lock } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';

export default function Login() {
  const { isDark, toggleTheme } = useTheme();
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (email.trim()) {
      login(email);
      navigate('/dashboard');
    }
  };

  return (
    <div className={`min-h-screen ${isDark ? 'dark' : ''} bg-surface transition-colors duration-300 flex flex-col`}>
      {/* Top Bar for Mode Toggle */}
      <div className="flex justify-end p-6">
        {/* Premium Theme Toggle Switch */}
        <button 
          onClick={toggleTheme} 
          aria-label="Toggle theme"
          className="relative flex items-center justify-between w-14 h-7 rounded-full bg-surface-variant p-1 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary transition-colors shadow-inner"
        >
          <div className={`absolute top-1 left-1 bg-primary w-5 h-5 rounded-full shadow-md transform transition-transform duration-300 ease-out ${isDark ? 'translate-x-7' : 'translate-x-0'}`}></div>
          <div className="w-full flex justify-between px-1 z-10 pointer-events-none text-on-surface-variant">
             <Sun size={12} className={`transition-colors duration-300 ${!isDark ? 'text-on-primary' : 'opacity-50'}`} />
             <Moon size={12} className={`transition-colors duration-300 ${isDark ? 'text-on-primary' : 'opacity-50'}`} />
          </div>
        </button>
      </div>

      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="font-display text-4xl md:text-5xl text-on-surface mb-4">ARTHA</h1>
            <p className="font-ui text-on-surface-variant text-sm tracking-wide uppercase">Systematic Terminal</p>
          </div>

          {/* Login Card */}
          <div className="border border-outline-variant bg-surface-container-low p-8 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-[1px] bg-primary/20"></div>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="email" className="block font-ui text-sm text-on-surface">Email Address</label>
                <input 
                  type="email" 
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-surface border border-outline text-on-surface px-4 py-3 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary transition-shadow"
                  required
                />
              </div>

              <button 
                type="submit"
                className="w-full bg-primary text-surface font-ui py-3 px-4 hover:bg-primary/90 transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-surface"
              >
                Continue
              </button>
            </form>

            <div className="mt-8 border-t border-outline-variant pt-6">
              <div className="flex items-center justify-center gap-2 text-xs text-on-surface-variant font-mono">
                <ShieldCheck size={14} className="text-secondary" />
                <span>Zero-Trust Middleware Active</span>
              </div>
              <div className="flex items-center justify-center gap-2 text-xs text-on-surface-variant font-mono mt-2">
                <Lock size={14} className="text-secondary" />
                <span>Restricted Access</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

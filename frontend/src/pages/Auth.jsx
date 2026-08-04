import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Activity } from 'lucide-react';

export default function Auth() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLogin, setIsLogin] = useState(true);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    // Mock login for now
    if (email) {
      login(email);
      navigate('/dashboard/RELIANCE');
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md border border-outline-variant bg-surface p-8">
        <div className="flex justify-center mb-8">
          <h1 className="font-display text-4xl tracking-tighter text-primary">ARTHA</h1>
        </div>
        
        <div className="text-center mb-8">
          <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface mb-2">
            {isLogin ? 'Authentication Required' : 'Create Access Token'}
          </h2>
          <p className="font-mono text-[10px] text-outline">
            Secure connection to the quant engine
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block font-ui text-[10px] uppercase tracking-widest text-outline mb-2">
              Identifier (Email)
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full bg-surface-container border border-outline-variant p-3 font-mono text-sm text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
              placeholder="operator@artha.system"
            />
          </div>

          <div>
            <label className="block font-ui text-[10px] uppercase tracking-widest text-outline mb-2">
              Passkey
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full bg-surface-container border border-outline-variant p-3 font-mono text-sm text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            className="w-full flex items-center justify-center gap-2 bg-primary/10 border border-primary text-primary p-3 font-ui text-xs uppercase tracking-widest hover:bg-primary/20 transition-colors"
          >
            <Activity size={14} />
            {isLogin ? 'Establish Connection' : 'Initialize Account'}
          </button>
        </form>

        <div className="mt-8 text-center border-t border-outline-variant pt-6">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="font-mono text-[10px] text-outline hover:text-primary transition-colors"
          >
            {isLogin ? 'Request new access token?' : 'Already have connection credentials?'}
          </button>
        </div>
      </div>
      
      <div className="mt-12 text-center">
        <p className="font-mono text-[9px] text-outline-variant uppercase tracking-widest">
          End-to-End Encryption Enabled • ARTHA Node v2.1.0
        </p>
      </div>
    </div>
  );
}

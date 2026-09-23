import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Loader2, ArrowLeft } from 'lucide-react';

export default function Auth() {
  const { signIn, signUp, verifyOtp, resetPassword } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState('login'); // 'login' | 'signup' | 'otp' | 'forgot' | 'reset'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [otp, setOtp] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);

    try {
      if (mode === 'login') {
        const { error } = await signIn(email, password);
        if (error) throw error;
        navigate('/dashboard/overview');
      } 
      else if (mode === 'signup') {
        // Password Validation
        if (password.length < 12) {
          throw new Error("Password must be at least 12 characters long.");
        }
        if (!/[0-9]/.test(password)) {
          throw new Error("Password must contain at least one number.");
        }
        if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
          throw new Error("Password must contain at least one special character.");
        }

        const { error } = await signUp(email, password);
        if (error) throw error;
        setMode('otp');
        setMessage("Please check your email for the verification code.");
      }
      else if (mode === 'otp') {
        const { error } = await verifyOtp(email, otp, 'signup');
        if (error) throw error;
        navigate('/dashboard/overview');
      }
      else if (mode === 'forgot') {
        const { error } = await resetPassword(email);
        if (error) throw error;
        setMessage("Password reset link sent! Check your email.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-dim flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-surface border border-outline-variant p-8 shadow-xl shadow-black/50 relative overflow-hidden">
        
        {/* Decorative corner */}
        <div className="absolute top-0 right-0 w-12 h-12 border-l border-b border-outline-variant/50 flex items-start justify-end p-3 rounded-bl-2xl">
          <div className="w-1.5 h-1.5 rounded-full bg-primary/80 animate-pulse"></div>
        </div>

        <h1 className="font-display text-4xl text-primary mb-2 tracking-tight">ARTHA</h1>
        
        <h2 className="font-mono text-sm text-outline uppercase tracking-widest mb-8">
          {mode === 'login' && 'Secure Access Node'}
          {mode === 'signup' && 'Create Identity'}
          {mode === 'otp' && 'Verify Identity'}
          {mode === 'forgot' && 'Account Recovery'}
        </h2>

        {error && (
          <div className="mb-6 p-4 border border-error/50 bg-error/10 text-error font-mono text-sm">
            {error}
          </div>
        )}

        {message && (
          <div className="mb-6 p-4 border border-secondary/50 bg-secondary/10 text-secondary font-mono text-sm">
            {message}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          
          {(mode === 'login' || mode === 'signup' || mode === 'forgot' || mode === 'otp') && (
            <div className="flex flex-col gap-2">
              <label className="font-ui text-[10px] uppercase tracking-wider text-outline">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={mode === 'otp' || loading}
                className="bg-surface-container border border-outline-variant px-4 py-3 text-sm text-on-surface focus:border-primary focus:outline-none transition-colors disabled:opacity-50"
              />
            </div>
          )}

          {(mode === 'login' || mode === 'signup') && (
            <div className="flex flex-col gap-2">
              <div className="flex justify-between">
                <label className="font-ui text-[10px] uppercase tracking-wider text-outline">Password</label>
                {mode === 'login' && (
                  <button type="button" onClick={() => setMode('forgot')} className="font-ui text-[10px] uppercase tracking-wider text-primary hover:underline">
                    Forgot?
                  </button>
                )}
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                className="bg-surface-container border border-outline-variant px-4 py-3 text-sm text-on-surface focus:border-primary focus:outline-none transition-colors"
              />
            </div>
          )}

          {mode === 'otp' && (
            <div className="flex flex-col gap-2">
              <label className="font-ui text-[10px] uppercase tracking-wider text-outline">6-Digit Verification Code</label>
              <input
                type="text"
                required
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
                disabled={loading}
                placeholder="000000"
                className="bg-surface-container border border-outline-variant px-4 py-3 text-center tracking-[1em] text-lg text-on-surface focus:border-primary focus:outline-none transition-colors font-mono"
                maxLength={6}
              />
            </div>
          )}

          <div className="mt-4 flex flex-col gap-4">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary/10 border border-primary text-primary py-3 font-ui text-[11px] uppercase tracking-[0.2em] hover:bg-primary hover:text-surface-dim transition-all flex justify-center items-center gap-2 disabled:opacity-50"
            >
              {loading && <Loader2 size={16} className="animate-spin" />}
              {mode === 'login' && 'Authenticate'}
              {mode === 'signup' && 'Register'}
              {mode === 'otp' && 'Verify & Enter'}
              {mode === 'forgot' && 'Send Recovery Link'}
            </button>
            
            {mode !== 'login' && (
              <button 
                type="button" 
                onClick={() => setMode('login')}
                className="flex items-center justify-center gap-2 text-outline hover:text-on-surface font-ui text-[10px] uppercase tracking-wider transition-colors"
              >
                <ArrowLeft size={14} /> Back to Login
              </button>
            )}
          </div>
        </form>

        {mode === 'login' && (
          <div className="mt-8 pt-6 border-t border-outline-variant/50 text-center">
            <p className="font-mono text-xs text-outline">
              No access clearance?{' '}
              <button onClick={() => setMode('signup')} className="text-primary hover:underline">
                Request Access
              </button>
            </p>
          </div>
        )}
        
      </div>
    </div>
  );
}

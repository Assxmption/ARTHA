import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Check, Circle, Eye, EyeOff, KeyRound, Loader2, Mail, ShieldCheck } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { getPreferences } from '../lib/preferences';
import {
  A_CODE, OTP_LENGTH, RESEND_COOLDOWN_S, cleanCode, friendlyAuthError, passwordRules, passwordValid,
} from '../lib/authPolicy';

// Views: sign in (password or emailed code), create account (+ email code
// verification), and password recovery by emailed code. Codes and links come
// from Supabase Auth; the email templates must include {{ .Token }} for codes
// (see docs/DEPLOYMENT.md → Email & one-time codes).

const SUBTITLE = {
  login: 'Secure Access Node',
  signup: 'Create Identity',
  verify: 'Verify Identity',
  code: 'One-Time Access Code',
  forgot: 'Account Recovery',
  reset: 'Set New Password',
};

const inputCls =
  'w-full bg-surface-container border border-outline-variant px-4 py-3 text-sm text-on-surface focus:border-primary focus:outline-none transition-colors disabled:opacity-50';

export default function Auth() {
  const { signIn, signUp, verifyOtp, resetPassword, sendSignInCode, resendSignupCode, updatePassword, checkEmailHasAccount } = useAuth();
  const navigate = useNavigate();

  const [view, setView] = useState('login');
  const [method, setMethod] = useState('password'); // sign-in method: 'password' | 'code'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [cooldown, setCooldown] = useState(0);
  const [noAccount, setNoAccount] = useState(false); // offer "create an account" after a failed lookup

  useEffect(() => {
    if (cooldown <= 0) return undefined;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const go = (next, { keepMessage = false } = {}) => {
    setView(next);
    setError(null);
    setNoAccount(false);
    if (!keepMessage) setMessage(null);
    setCode('');
    if (next === 'login' || next === 'signup') { setPassword(''); setConfirm(''); }
  };

  const enter = () => navigate(getPreferences().landingPage);

  const run = async (fn) => {
    setLoading(true);
    setError(null);
    setNoAccount(false);
    try {
      await fn();
    } catch (err) {
      setError(friendlyAuthError(err));
    } finally {
      setLoading(false);
    }
  };

  const startCooldown = () => setCooldown(RESEND_COOLDOWN_S);

  // Tell the person when no account uses this email (instead of a silent "sent").
  const noAccountError = () => {
    setNoAccount(true);
    return new Error(`We couldn’t find an ARTHA account for ${email}. Check the spelling, or create an account.`);
  };

  // ── Actions ────────────────────────────────────────────────────────────────
  const submitLogin = () => run(async () => {
    if (method === 'code') {
      if ((await checkEmailHasAccount(email)) === false) throw noAccountError();
      const { error } = await sendSignInCode(email);
      if (error) {
        if (error.code === 'otp_disabled' || /signups not allowed for otp/i.test(error.message)) throw noAccountError();
        throw error;
      }
      startCooldown();
      go('code');
      setMessage(`We emailed ${A_CODE} code to ${email}.`);
      return;
    }
    const { error } = await signIn(email, password);
    if (error) {
      if (error.code === 'email_not_confirmed' || /not confirmed/i.test(error.message)) {
        go('verify');
        setMessage('Your email isn’t verified yet. Enter the code from your sign-up email, or send a new one.');
        return;
      }
      throw error;
    }
    enter();
  });

  const submitSignup = () => run(async () => {
    if (!passwordValid(password, confirm)) throw new Error('Your password doesn’t meet the requirements yet.');
    const { data, error } = await signUp(email, password);
    if (error) throw error;
    // With email confirmation on, Supabase hides whether the address exists by
    // returning a user with no identities instead of an error.
    if (data?.user && Array.isArray(data.user.identities) && data.user.identities.length === 0) {
      throw new Error('An account with this email already exists. Sign in instead.');
    }
    if (data?.session) { enter(); return; }
    startCooldown();
    go('verify');
    setMessage(`We emailed ${A_CODE} verification code to ${email}.`);
  });

  const submitVerify = (type) => run(async () => {
    const token = cleanCode(code);
    if (token.length < 6) throw new Error('Enter the full code from the email.');
    const { error } = await verifyOtp(email, token, type);
    if (error) throw error;
    enter();
  });

  const submitForgot = () => run(async () => {
    const exists = await checkEmailHasAccount(email); // true | false | null (unavailable/throttled)
    if (exists === false) throw noAccountError();
    const { error } = await resetPassword(email);
    if (error) throw error;
    startCooldown();
    go('reset');
    setMessage(exists
      ? `We emailed ${A_CODE} reset code (and a link) to ${email}.`
      : `If an account exists for ${email}, we emailed it ${A_CODE} reset code (and a link).`);
  });

  const submitReset = () => run(async () => {
    const token = cleanCode(code);
    if (token.length < 6) throw new Error('Enter the full code from the email.');
    if (!passwordValid(password, confirm)) throw new Error('Your new password doesn’t meet the requirements yet.');
    const { error: vErr } = await verifyOtp(email, token, 'recovery');
    if (vErr) throw vErr;
    const { error: uErr } = await updatePassword(password);
    if (uErr) throw uErr;
    enter();
  });

  const resend = () => run(async () => {
    const fn = view === 'verify' ? resendSignupCode : view === 'reset' ? resetPassword : sendSignInCode;
    const { error } = await fn(email);
    if (error) throw error;
    startCooldown();
    setMessage(`A new code is on its way to ${email}.`);
  });

  const onSubmit = (e) => {
    e.preventDefault();
    if (loading) return;
    ({
      login: submitLogin,
      signup: submitSignup,
      verify: () => submitVerify('signup'),
      code: () => submitVerify('email'),
      forgot: submitForgot,
      reset: submitReset,
    })[view]();
  };

  const codeView = view === 'verify' || view === 'code' || view === 'reset';
  const submitLabel = {
    login: method === 'code' ? 'Email Me a Code' : 'Authenticate',
    signup: 'Create Account',
    verify: 'Verify & Enter',
    code: 'Verify & Enter',
    forgot: 'Send Reset Code',
    reset: 'Set Password & Enter',
  }[view];

  return (
    <div className="min-h-screen bg-surface-dim flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-surface border border-outline-variant p-8 shadow-xl shadow-black/50 relative overflow-hidden">

        {/* Decorative corner */}
        <div className="absolute top-0 right-0 w-12 h-12 border-l border-b border-outline-variant/50 flex items-start justify-end p-3 rounded-bl-2xl">
          <div className="w-1.5 h-1.5 rounded-full bg-primary/80 animate-pulse"></div>
        </div>

        <h1 className="font-display text-4xl text-primary mb-2 tracking-tight">ARTHA</h1>
        <h2 className="font-mono text-sm text-outline uppercase tracking-widest mb-6">{SUBTITLE[view]}</h2>

        {(view === 'login' || view === 'signup') && (
          <div className="grid grid-cols-2 mb-6 border border-outline-variant" role="tablist" aria-label="Account">
            {[['login', 'Sign in'], ['signup', 'Create account']].map(([v, label]) => (
              <button key={v} type="button" role="tab" aria-selected={view === v} onClick={() => go(v)}
                className={`py-2.5 font-ui text-[10px] uppercase tracking-widest transition-colors ${view === v ? 'bg-primary/10 text-primary' : 'text-outline hover:text-on-surface'}`}>
                {label}
              </button>
            ))}
          </div>
        )}

        <div aria-live="polite">
          {error && (
            <div role="alert" className="mb-5 p-3.5 border border-error/50 bg-error/10 text-error font-mono text-xs leading-relaxed">
              {error}
              {noAccount && (
                <button type="button" onClick={() => go('signup')}
                  className="block mt-2 font-ui text-[10px] uppercase tracking-wider text-primary hover:underline">
                  Create an account with this email →
                </button>
              )}
            </div>
          )}
          {message && !error && (
            <div className="mb-5 p-3.5 border border-secondary/50 bg-secondary/10 text-secondary font-mono text-xs leading-relaxed">{message}</div>
          )}
        </div>

        <form onSubmit={onSubmit} className="flex flex-col gap-5">
          {view === 'login' && (
            <div className="flex gap-4 -mb-1" role="radiogroup" aria-label="Sign-in method">
              {[['password', 'Password', KeyRound], ['code', 'Email code', Mail]].map(([m, label, Icon]) => (
                <button key={m} type="button" role="radio" aria-checked={method === m} onClick={() => { setMethod(m); setError(null); }}
                  className={`flex items-center gap-1.5 font-ui text-[10px] uppercase tracking-wider pb-1 border-b transition-colors ${method === m ? 'border-primary text-primary' : 'border-transparent text-outline hover:text-on-surface'}`}>
                  <Icon size={12} /> {label}
                </button>
              ))}
            </div>
          )}

          <Field label="Email address" htmlFor="auth-email">
            <input id="auth-email" type="email" required autoComplete="email" inputMode="email" value={email}
              onChange={(e) => setEmail(e.target.value.trim())} disabled={codeView || loading} className={inputCls} />
          </Field>

          {view === 'login' && method === 'password' && (
            <PasswordField id="auth-password" label="Password" value={password} onChange={setPassword}
              autoComplete="current-password" disabled={loading}
              action={<button type="button" onClick={() => go('forgot')} className="font-ui text-[10px] uppercase tracking-wider text-primary hover:underline">Forgot?</button>} />
          )}

          {(view === 'signup' || view === 'reset') && (
            <>
              {view === 'reset' && <CodeField value={code} onChange={setCode} disabled={loading} />}
              <PasswordField id="auth-new-password" label={view === 'reset' ? 'New password' : 'Password'} value={password}
                onChange={setPassword} autoComplete="new-password" disabled={loading} />
              <PasswordField id="auth-confirm" label="Confirm password" value={confirm} onChange={setConfirm}
                autoComplete="new-password" disabled={loading} />
              <Rules rules={passwordRules(password, confirm)} />
            </>
          )}

          {(view === 'verify' || view === 'code') && (
            <CodeField value={code} onChange={setCode} disabled={loading}
              onComplete={() => submitVerify(view === 'verify' ? 'signup' : 'email')} />
          )}

          <div className="mt-2 flex flex-col gap-4">
            <button type="submit" disabled={loading}
              className="w-full bg-primary/10 border border-primary text-primary py-3 font-ui text-[11px] uppercase tracking-[0.2em] hover:bg-primary hover:text-surface-dim transition-all flex justify-center items-center gap-2 disabled:opacity-50">
              {loading && <Loader2 size={16} className="animate-spin" />}
              {submitLabel}
            </button>

            {codeView && (
              <div className="flex items-center justify-between font-ui text-[10px] uppercase tracking-wider">
                <span className="text-outline normal-case tracking-normal font-mono">Check spam if it hasn’t arrived.</span>
                <button type="button" onClick={resend} disabled={cooldown > 0 || loading}
                  className="text-primary hover:underline disabled:text-outline disabled:no-underline">
                  {cooldown > 0 ? `Resend in ${cooldown}s` : 'Resend code'}
                </button>
              </div>
            )}

            {view !== 'login' && view !== 'signup' && (
              <button type="button" onClick={() => go('login')}
                className="flex items-center justify-center gap-2 text-outline hover:text-on-surface font-ui text-[10px] uppercase tracking-wider transition-colors">
                <ArrowLeft size={14} /> Back to sign in
              </button>
            )}
          </div>
        </form>

        <div className="mt-8 pt-5 border-t border-outline-variant/50 flex items-start gap-2 font-mono text-[10px] text-outline leading-relaxed">
          <ShieldCheck size={12} className="shrink-0 mt-0.5 text-primary/70" />
          <span>Market research and simulated (paper) trading only — no real orders, not investment advice. Codes expire; never share them.</span>
        </div>
      </div>
    </div>
  );
}

function Field({ label, htmlFor, action, children }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between items-center">
        <label htmlFor={htmlFor} className="font-ui text-[10px] uppercase tracking-wider text-outline">{label}</label>
        {action}
      </div>
      {children}
    </div>
  );
}

function PasswordField({ id, label, value, onChange, autoComplete, disabled, action }) {
  const [show, setShow] = useState(false);
  const [caps, setCaps] = useState(false);
  const onKey = (e) => setCaps(!!e.getModifierState?.('CapsLock'));
  return (
    <Field label={label} htmlFor={id} action={action}>
      <div className="relative">
        <input id={id} type={show ? 'text' : 'password'} required value={value} autoComplete={autoComplete}
          onChange={(e) => onChange(e.target.value)} onKeyUp={onKey} onKeyDown={onKey} disabled={disabled}
          className={`${inputCls} pr-11`} />
        <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide password' : 'Show password'}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-outline hover:text-on-surface">
          {show ? <EyeOff size={15} /> : <Eye size={15} />}
        </button>
      </div>
      {caps && <span className="font-mono text-[10px] text-tertiary">Caps Lock is on</span>}
    </Field>
  );
}

function CodeField({ value, onChange, disabled, onComplete }) {
  const ref = useRef(null);
  const fired = useRef('');
  useEffect(() => { ref.current?.focus(); }, []);
  const set = (v) => {
    const c = cleanCode(v);
    onChange(c);
    if (onComplete && c.length === OTP_LENGTH && fired.current !== c) { fired.current = c; onComplete(); }
  };
  return (
    <Field label={`${OTP_LENGTH}-digit code from your email`} htmlFor="auth-code">
      <input id="auth-code" ref={ref} type="text" inputMode="numeric" autoComplete="one-time-code" required
        value={value} onChange={(e) => set(e.target.value)} disabled={disabled}
        placeholder={'0'.repeat(OTP_LENGTH)} maxLength={10}
        className={`${inputCls} text-center tracking-[0.6em] text-lg font-mono`} />
    </Field>
  );
}

function Rules({ rules }) {
  return (
    <ul className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[10px] -mt-1">
      {rules.map(([ok, text]) => (
        <li key={text} className={`flex items-center gap-1.5 ${ok ? 'text-secondary' : 'text-outline'}`}>
          {ok ? <Check size={11} /> : <Circle size={9} />} {text}
        </li>
      ))}
    </ul>
  );
}

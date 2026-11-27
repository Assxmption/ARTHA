import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  User, Link2, Palette, Rocket, LineChart, Server, ShieldCheck, HardDrive, Info,
  Check, AlertTriangle, RefreshCw, LogOut, KeyRound, Mail, Download, Upload, RotateCcw, Trash2, ExternalLink,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { supabase } from '../lib/supabase';
import {
  DEFAULT_PREFERENCES, importPreferences, prefersReducedMotion, resetPreferences, setPreference, usePreferences,
} from '../lib/preferences';
import { clearActiveSymbol, useActiveSymbol } from '../lib/activeSymbol';
import { INTERVALS, OVERLAYS, PANES } from '../components/CandlestickChart';
import { KEY_FORMATS, clearLlmKeys, getLlmKeys, isRemembered, maskKey, setLlmKeys } from '../lib/llmKeys';
import { A_CODE, OTP_LENGTH, cleanCode, friendlyAuthError, passwordRules, passwordValid } from '../lib/authPolicy';
import { apiFetch } from '../lib/api';

// Every control on this page does something real: display preferences are
// read by the components they name, account actions go to Supabase, broker
// actions go to /api/kite, and status comes from /api/system/status. Nothing
// here can change what the quant engine computes or loosen its validation gate.

const SECTIONS = [
  { id: 'account', label: 'Account', icon: User },
  { id: 'broker', label: 'Broker connection', icon: Link2 },
  { id: 'ai-keys', label: 'AI model keys', icon: KeyRound },
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'startup', label: 'Start-up & overview', icon: Rocket },
  { id: 'chart', label: 'Chart defaults', icon: LineChart },
  { id: 'system', label: 'Data & system status', icon: Server },
  { id: 'safeguards', label: 'Safeguards', icon: ShieldCheck },
  { id: 'data', label: 'Your data', icon: HardDrive },
  { id: 'about', label: 'About', icon: Info },
];

const LANDING = [
  ['/home', 'Market overview'], ['/dashboard', 'Chart (last instrument)'], ['/signals', 'Quant signals'],
  ['/watchlist', 'Watchlist'], ['/screener', 'Screener'], ['/paper-trading', 'Paper trading'],
];
const CHART_TYPES = [['candle', 'Candles'], ['ha', 'Heikin-Ashi'], ['line', 'Line'], ['area', 'Area']];
const plural = (n, unit) => `${n} ${unit}${n === 1 ? '' : 's'}`;
const RANGE_LABEL = (d) => (d < 30 ? plural(d, 'day') : d < 365 ? plural(Math.round(d / 30), 'month') : plural(Math.round(d / 365), 'year'));

const fmtDate = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? '—'
    : d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'short' }) + ' IST';
};
const fmtDuration = (s) => {
  if (typeof s !== 'number') return '';
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.round(s / 60)}m`;
  return `${Math.round(s)}s`;
};

// ── Small building blocks ──────────────────────────────────────────────────────

function Section({ id, title, description, icon: Icon, children }) {
  return (
    <section id={id} className="scroll-mt-20 border border-outline-variant bg-surface">
      <header className="border-b border-outline-variant bg-surface-container-low px-4 md:px-5 py-3 flex items-start gap-3">
        <Icon size={16} className="text-primary mt-0.5 shrink-0" />
        <div className="min-w-0">
          <h2 className="font-ui text-xs uppercase tracking-widest text-on-surface">{title}</h2>
          {description && <p className="font-mono text-[11px] text-outline mt-1">{description}</p>}
        </div>
      </header>
      <div className="divide-y divide-outline-variant/40">{children}</div>
    </section>
  );
}

function Row({ label, hint, children, htmlFor }) {
  return (
    <div className="px-4 md:px-5 py-3.5 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-6">
      <div className="sm:w-64 shrink-0 min-w-0">
        <label htmlFor={htmlFor} className="font-ui text-xs text-on-surface block">{label}</label>
        {hint && <p className="font-mono text-[10px] text-outline mt-0.5">{hint}</p>}
      </div>
      <div className="flex-1 min-w-0 flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}

function Toggle({ checked, onChange, label, disabled }) {
  return (
    <button
      type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative w-10 h-5 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-40 ${checked ? 'bg-primary' : 'bg-surface-variant'}`}
    >
      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-surface shadow transition-transform ${checked ? 'translate-x-5' : ''}`} />
    </button>
  );
}

function Segmented({ value, options, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex border border-outline-variant flex-wrap">
      {options.map(([v, text]) => (
        <button
          key={String(v)} type="button" role="radio" aria-checked={value === v} onClick={() => onChange(v)}
          className={`px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider transition-colors border-r border-outline-variant last:border-r-0 ${value === v ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'}`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function Select({ id, value, onChange, options }) {
  return (
    <select
      id={id} value={value} onChange={(e) => onChange(e.target.value)}
      className="bg-surface-container-low border border-outline-variant px-2 py-1.5 font-mono text-xs text-on-surface focus:outline-none focus:border-primary min-w-[12rem]"
    >
      {options.map(([v, text]) => <option key={v} value={v}>{text}</option>)}
    </select>
  );
}

function Button({ children, onClick, tone = 'default', disabled, type = 'button', title }) {
  const tones = {
    default: 'border-outline-variant text-on-surface hover:border-primary hover:text-primary',
    primary: 'border-primary text-primary hover:bg-primary/10',
    danger: 'border-error/60 text-error hover:bg-error/10',
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title}
      className={`inline-flex items-center gap-2 border px-3 py-1.5 font-ui text-[10px] uppercase tracking-wider transition-colors disabled:opacity-40 disabled:pointer-events-none ${tones[tone]}`}>
      {children}
    </button>
  );
}

function Pill({ ok, children, warn }) {
  const cls = ok ? 'border-secondary text-secondary' : warn ? 'border-primary text-primary' : 'border-error/70 text-error';
  return <span className={`px-1.5 py-0.5 border font-mono text-[10px] uppercase ${cls}`}>{children}</span>;
}

function Notice({ notice }) {
  if (!notice) return null;
  return (
    <p role="status" className={`font-mono text-[11px] ${notice.ok ? 'text-secondary' : 'text-error'}`}>
      {notice.ok ? <Check size={11} className="inline mr-1" /> : <AlertTriangle size={11} className="inline mr-1" />}
      {notice.text}
    </p>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function Settings() {
  const prefs = usePreferences();
  const { user, session, signOut, resetPassword, updatePassword, requestReauthCode } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const rememberedSymbol = useActiveSymbol();
  const [saved, setSaved] = useState(false);
  const savedTimer = useRef(null);

  const set = useCallback((key, value) => {
    setPreference(key, value);
    setSaved(true);
    clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => setSaved(false), 1500);
  }, []);
  useEffect(() => () => clearTimeout(savedTimer.current), []);

  const recovery = new URLSearchParams(location.search).get('recovery') === '1';

  useEffect(() => {
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }, [location.hash]);

  return (
    <div className="p-4 md:p-6 bg-surface-dim min-h-full">
      <div className="flex flex-wrap items-end justify-between gap-3 mb-6 border-b border-outline-variant pb-4">
        <div>
          <h1 className="font-display text-3xl mb-1 text-on-surface">Settings</h1>
          <p className="font-ui text-xs uppercase tracking-widest text-outline">Account, broker, display and system</p>
        </div>
        <span aria-live="polite" className={`font-mono text-[11px] text-secondary transition-opacity ${saved ? 'opacity-100' : 'opacity-0'}`}>
          <Check size={11} className="inline mr-1" />Saved on this device
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[13rem_1fr] gap-6">
        <nav className="hidden lg:block" aria-label="Settings sections">
          <ul className="sticky top-4 space-y-0.5">
            {SECTIONS.map(({ id, label, icon: Icon }) => (
              <li key={id}>
                <a href={`#${id}`} className={`flex items-center gap-2 px-3 py-2 font-ui text-[11px] uppercase tracking-wider border-l-2 transition-colors ${location.hash === `#${id}` ? 'border-primary text-primary bg-primary/5' : 'border-transparent text-outline hover:text-on-surface hover:bg-surface'}`}>
                  <Icon size={13} /> {label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="space-y-6 min-w-0 max-w-4xl">
          <AccountSection user={user} recovery={recovery} signOut={signOut} resetPassword={resetPassword}
            updatePassword={updatePassword} requestReauthCode={requestReauthCode} navigate={navigate} />
          <BrokerSection session={session} />
          <AiKeysSection session={session} />
          <AppearanceSection prefs={prefs} set={set} />
          <StartupSection prefs={prefs} set={set} rememberedSymbol={rememberedSymbol} />
          <ChartSection prefs={prefs} set={set} />
          <SystemSection />
          <SafeguardsSection />
          <DataSection prefs={prefs} />
          <AboutSection />
        </div>
      </div>
    </div>
  );
}

// ── Account ────────────────────────────────────────────────────────────────────

function AccountSection({ user, recovery, signOut, resetPassword, updatePassword, requestReauthCode, navigate }) {
  const [pw, setPw] = useState('');
  const [pw2, setPw2] = useState('');
  const [code, setCode] = useState('');
  const [needCode, setNeedCode] = useState(false);
  const [busy, setBusy] = useState(null);
  const [notice, setNotice] = useState(recovery ? { ok: true, text: 'Recovery link accepted — choose a new password below.' } : null);

  // Same policy as sign-up (lib/authPolicy.js).
  const rules = passwordRules(pw, pw2);
  const valid = passwordValid(pw, pw2) && (!needCode || cleanCode(code).length >= 6);

  const sendCode = async () => {
    setBusy('code');
    const { error } = await requestReauthCode();
    setBusy(null);
    if (error) setNotice({ ok: false, text: friendlyAuthError(error) });
    else { setNeedCode(true); setNotice({ ok: true, text: `We emailed ${A_CODE} confirmation code to ${user.email}.` }); }
  };

  const changePassword = async (e) => {
    e.preventDefault();
    if (!valid) return;
    setBusy('pw');
    const { error } = await updatePassword(pw, needCode ? cleanCode(code) : undefined);
    setBusy(null);
    if (error) {
      // With Supabase's "Secure password change" on, a session that isn't
      // fresh must confirm with an emailed code first.
      if (error.code === 'reauthentication_needed' || /reauthenticat/i.test(error.message)) {
        await sendCode();
        return;
      }
      setNotice({ ok: false, text: friendlyAuthError(error) });
      return;
    }
    setPw(''); setPw2(''); setCode(''); setNeedCode(false);
    setNotice({ ok: true, text: 'Password updated. Other devices stay signed in until you sign them out below.' });
  };

  const sendReset = async () => {
    setBusy('reset');
    const { error } = await resetPassword(user.email);
    setBusy(null);
    setNotice(error ? { ok: false, text: error.message } : { ok: true, text: `Reset link sent to ${user.email}. Check spam if it doesn't arrive within a few minutes.` });
  };

  const signOutHere = async () => { await signOut(); navigate('/auth'); };
  const signOutEverywhere = async () => {
    if (!window.confirm('Sign out of ARTHA on every device and browser?')) return;
    setBusy('global');
    const { error } = await supabase.auth.signOut({ scope: 'global' });
    setBusy(null);
    if (error) setNotice({ ok: false, text: error.message });
    else navigate('/auth');
  };

  if (!user) return null;
  const provider = user.app_metadata?.provider ?? 'email';
  return (
    <Section id="account" title="Account" icon={User} description="Your ARTHA sign-in, managed by Supabase Auth.">
      <Row label="Email">
        <span className="font-mono text-sm text-on-surface break-all">{user.email}</span>
        <Pill ok={!!user.email_confirmed_at} warn={!user.email_confirmed_at}>{user.email_confirmed_at ? 'Verified' : 'Unverified'}</Pill>
      </Row>
      <Row label="Sign-in method"><span className="font-mono text-xs text-on-surface-variant capitalize">{provider}</span></Row>
      <Row label="Member since"><span className="font-mono text-xs text-on-surface-variant">{fmtDate(user.created_at)}</span></Row>
      <Row label="Last sign-in"><span className="font-mono text-xs text-on-surface-variant">{fmtDate(user.last_sign_in_at)}</span></Row>
      <Row label="User ID" hint="Quote this if you report a problem.">
        <code className="font-mono text-[11px] text-on-surface-variant break-all">{user.id}</code>
      </Row>

      {provider === 'email' && (
        <form onSubmit={changePassword} className="px-4 md:px-5 py-4 space-y-3">
          <div className="flex items-center gap-2"><KeyRound size={13} className="text-primary" /><span className="font-ui text-xs text-on-surface">Change password</span></div>
          <div className="grid sm:grid-cols-2 gap-3 max-w-xl">
            <input type="password" autoComplete="new-password" placeholder="New password" value={pw} onChange={(e) => setPw(e.target.value)}
              aria-label="New password"
              className="bg-surface-container-low border border-outline-variant px-3 py-2 font-mono text-xs text-on-surface focus:outline-none focus:border-primary" />
            <input type="password" autoComplete="new-password" placeholder="Confirm new password" value={pw2} onChange={(e) => setPw2(e.target.value)}
              aria-label="Confirm new password"
              className="bg-surface-container-low border border-outline-variant px-3 py-2 font-mono text-xs text-on-surface focus:outline-none focus:border-primary" />
          </div>
          <ul className="font-mono text-[10px] space-y-0.5">
            {rules.map(([ok, text]) => (
              <li key={text} className={ok ? 'text-secondary' : 'text-outline'}>{ok ? '✓' : '○'} {text}</li>
            ))}
          </ul>
          {needCode && (
            <div className="max-w-xs space-y-1">
              <input type="text" inputMode="numeric" autoComplete="one-time-code" placeholder={'0'.repeat(OTP_LENGTH)}
                value={code} onChange={(e) => setCode(cleanCode(e.target.value))} aria-label="Confirmation code from email"
                className="w-full bg-surface-container-low border border-outline-variant px-3 py-2 font-mono text-sm tracking-[0.4em] text-center text-on-surface focus:outline-none focus:border-primary" />
              <button type="button" onClick={sendCode} disabled={busy === 'code'} className="font-mono text-[10px] text-primary hover:underline disabled:text-outline">
                {busy === 'code' ? 'Sending…' : 'Send a new code'}
              </button>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button type="submit" tone="primary" disabled={!valid || busy === 'pw'}>{busy === 'pw' ? 'Updating…' : needCode ? 'Confirm & update password' : 'Update password'}</Button>
            <Button onClick={sendReset} disabled={busy === 'reset'}><Mail size={11} /> {busy === 'reset' ? 'Sending…' : 'Email me a reset link'}</Button>
          </div>
        </form>
      )}
      <div className="px-4 md:px-5 py-3.5 flex flex-wrap items-center gap-2">
        <Button onClick={signOutHere}><LogOut size={11} /> Sign out</Button>
        <Button tone="danger" onClick={signOutEverywhere} disabled={busy === 'global'}><LogOut size={11} /> Sign out on all devices</Button>
        <div className="w-full"><Notice notice={notice} /></div>
      </div>
    </Section>
  );
}

// ── Broker (Kite Connect) ──────────────────────────────────────────────────────

function BrokerSection({ session }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(null);
  const [notice, setNotice] = useState(null);
  const authHeaders = useMemo(
    () => (session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {}),
    [session?.access_token],
  );

  const load = useCallback(async () => {
    try {
      const r = await apiFetch('/api/kite/status', { cache: 'no-store' });
      setStatus(r.ok ? await r.json() : { error: `HTTP ${r.status}` });
    } catch (e) {
      setStatus({ error: e.message });
    }
  }, []);

  useEffect(() => {
    load();
    // After the Kite login tab closes, coming back here re-checks the session.
    const onFocus = () => load();
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [load]);

  const connect = async () => {
    setBusy('connect');
    setNotice(null);
    try {
      const r = await apiFetch('/api/kite/login-url', { method: 'POST', headers: authHeaders });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
      (() => {
        // Only ever open Zerodha's own login page.
        let u;
        try { u = new URL(body.login_url); } catch { u = null; }
        if (!u || u.protocol !== 'https:' || !/(^|\.)zerodha\.com$/.test(u.hostname)) throw new Error('Unexpected login URL from server');
        return window.open(u.toString(), '_blank', 'noopener,noreferrer');
      })();
      setNotice({ ok: true, text: 'Kite login opened in a new tab. Approve it there, then come back — this page re-checks automatically.' });
    } catch (e) {
      setNotice({ ok: false, text: e.message });
    } finally {
      setBusy(null);
    }
  };

  const disconnect = async () => {
    if (!window.confirm('Disconnect Kite? Live quotes fall back to delayed Yahoo data until you reconnect.')) return;
    setBusy('disconnect');
    try {
      const r = await apiFetch('/api/kite/logout', { method: 'POST', headers: authHeaders });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setNotice({ ok: true, text: 'Disconnected. The token was revoked at Kite and removed from this server.' });
      await load();
    } catch (e) {
      setNotice({ ok: false, text: e.message });
    } finally {
      setBusy(null);
    }
  };

  const connected = status?.connected;
  return (
    <Section id="broker" title="Broker connection" icon={Link2}
      description="Zerodha Kite Connect powers live quotes, minute bars, market depth, the option chain and India VIX.">
      <Row label="Status">
        {!status ? <span className="font-mono text-xs text-outline">Checking…</span>
          : status.error ? <Pill>Unreachable</Pill>
          : !status.configured ? <Pill warn>Not configured</Pill>
          : connected ? <Pill ok>Connected</Pill>
          : <Pill warn>Not connected</Pill>}
        <button onClick={load} className="text-outline hover:text-primary" aria-label="Re-check Kite status"><RefreshCw size={12} /></button>
      </Row>
      {connected && (
        <>
          <Row label="Kite user"><span className="font-mono text-xs text-on-surface">{status.user_name} <span className="text-outline">({status.user_id})</span></span></Row>
          <Row label="Enabled segments">
            {(status.exchanges ?? []).map((x) => <span key={x} className="px-1.5 py-0.5 border border-outline-variant font-mono text-[10px] text-on-surface-variant">{x}</span>)}
          </Row>
          <Row label="Session" hint="Kite tokens expire daily at 06:00 IST; reconnect each trading day.">
            <span className="font-mono text-xs text-on-surface-variant">Issued {fmtDate(status.session_created_at)} · expires {fmtDate(status.session_expires_at)}</span>
          </Row>
        </>
      )}
      {status && !status.configured && !status.error && (
        <Row label="Set-up" hint="Server-side; never entered in the browser.">
          <span className="font-mono text-[11px] text-on-surface-variant">
            Add <code>KITE_API_KEY</code>, <code>KITE_API_SECRET</code> and <code>KITE_REDIRECT_URL</code> to the server's <code>.env</code>, then restart the API.
          </span>
        </Row>
      )}
      {status && !connected && status.reason && (
        <Row label="Reason"><span className="font-mono text-[11px] text-outline">{status.reason}</span></Row>
      )}
      <div className="px-4 md:px-5 py-3.5 flex flex-wrap items-center gap-2">
        <Button tone="primary" onClick={connect} disabled={!status?.configured || busy === 'connect'}>
          <ExternalLink size={11} /> {connected ? 'Reconnect' : 'Connect Kite'}
        </Button>
        {connected && <Button tone="danger" onClick={disconnect} disabled={busy === 'disconnect'}>Disconnect</Button>}
        <div className="w-full"><Notice notice={notice} /></div>
        <p className="w-full font-mono text-[10px] text-outline">
          Read-only: ARTHA has no order endpoints and never places trades. Kite's approval screen lists order permissions because they are part of its standard scope.
        </p>
      </div>
    </Section>
  );
}

// ── Appearance ─────────────────────────────────────────────────────────────────

function AppearanceSection({ prefs, set }) {
  const reduced = prefersReducedMotion();
  return (
    <Section id="appearance" title="Appearance" icon={Palette} description="Saved in this browser.">
      <Row label="Theme" hint="System follows your OS light/dark setting.">
        <Segmented label="Theme" value={prefs.theme} onChange={(v) => set('theme', v)}
          options={[['dark', 'Dark'], ['light', 'Light'], ['system', 'System']]} />
      </Row>
      <Row label="Top-bar ticker" hint="Live index levels, validated pairs and stocks.">
        <Toggle label="Show top-bar ticker" checked={prefs.tickerEnabled} onChange={(v) => set('tickerEnabled', v)} />
      </Row>
      <Row label="Ticker content">
        <Segmented label="Ticker content" value={prefs.tickerContent} onChange={(v) => set('tickerContent', v)}
          options={[['indices', 'Indices + pairs'], ['all', 'Indices + pairs + stocks']]} />
      </Row>
      <Row label="Scroll the ticker" hint={reduced ? 'Your OS asks for reduced motion, so it stays still regardless.' : 'Hover the ticker to pause it.'}>
        <Toggle label="Scroll the ticker" checked={prefs.tickerScroll} onChange={(v) => set('tickerScroll', v)} disabled={!prefs.tickerEnabled} />
      </Row>
      <Row label="Ticker speed">
        <Segmented label="Ticker speed" value={prefs.tickerSpeed} onChange={(v) => set('tickerSpeed', v)}
          options={[['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']]} />
      </Row>
    </Section>
  );
}

// ── Start-up & overview ────────────────────────────────────────────────────────

function StartupSection({ prefs, set, rememberedSymbol }) {
  const [sym, setSym] = useState(prefs.defaultSymbol);
  const [symNotice, setSymNotice] = useState(null);
  useEffect(() => setSym(prefs.defaultSymbol), [prefs.defaultSymbol]);

  const saveSymbol = async (e) => {
    e.preventDefault();
    const s = sym.trim().toUpperCase();
    if (!/^[A-Z0-9&^._-]{1,20}$/.test(s)) { setSymNotice({ ok: false, text: 'Use an NSE symbol such as INFY or HDFCBANK.' }); return; }
    // Check it actually quotes before saving it as the default.
    try {
      const r = await apiFetch(`/api/quant/quote/${encodeURIComponent(s)}`);
      const q = r.ok ? await r.json() : null;
      if (!q || typeof q.ltp !== 'number') throw new Error();
    } catch {
      setSymNotice({ ok: false, text: `No quote found for ${s}.` });
      return;
    }
    set('defaultSymbol', s);
    setSymNotice({ ok: true, text: `${s} saved.` });
  };

  return (
    <Section id="startup" title="Start-up & overview" icon={Rocket}>
      <Row label="After sign-in, open" htmlFor="landing">
        <Select id="landing" value={prefs.landingPage} onChange={(v) => set('landingPage', v)} options={LANDING} />
      </Row>
      <Row label="Default instrument" hint="Used by Chart, Fundamentals and Sentiment until you open another stock." htmlFor="defsym">
        <form onSubmit={saveSymbol} className="flex flex-wrap items-center gap-2">
          <input id="defsym" value={sym} onChange={(e) => { setSym(e.target.value.toUpperCase()); setSymNotice(null); }}
            maxLength={20} spellCheck={false}
            className="bg-surface-container-low border border-outline-variant px-2 py-1.5 font-mono text-xs text-on-surface w-36 focus:outline-none focus:border-primary" />
          <Button type="submit" disabled={sym.trim().toUpperCase() === prefs.defaultSymbol}>Save</Button>
        </form>
        <div className="w-full"><Notice notice={symNotice} /></div>
      </Row>
      <Row label="Remembered instrument" hint="The last stock you opened; the sidebar's symbol tabs follow it.">
        <span className="font-mono text-xs text-on-surface">{rememberedSymbol}</span>
        <Button onClick={clearActiveSymbol}>Forget</Button>
      </Row>
      <Row label="Movers shown per list" hint="Top gainers, top losers and most active on Market overview.">
        <Segmented label="Movers shown per list" value={prefs.moversCount} onChange={(v) => set('moversCount', v)}
          options={[[3, '3'], [5, '5'], [10, '10']]} />
      </Row>
    </Section>
  );
}

// ── Chart defaults ─────────────────────────────────────────────────────────────

function ChartSection({ prefs, set }) {
  const iv = INTERVALS.find((i) => i.id === prefs.chartInterval) ?? INTERVALS.find((i) => i.id === '1d');
  const rangeOk = iv.ranges.includes(prefs.chartDays);
  const toggleIn = (key, id) => {
    const cur = new Set(prefs[key]);
    if (cur.has(id)) cur.delete(id); else cur.add(id);
    set(key, [...cur]);
  };
  return (
    <Section id="chart" title="Chart defaults" icon={LineChart}
      description="How a chart opens. Changes made on the chart itself apply to that view only.">
      <Row label="Interval" htmlFor="civ">
        <Select id="civ" value={iv.id} onChange={(v) => {
          const next = INTERVALS.find((i) => i.id === v);
          set('chartInterval', v);
          if (!next.ranges.includes(prefs.chartDays)) set('chartDays', next.ranges[Math.min(1, next.ranges.length - 1)]);
        }} options={INTERVALS.map((i) => [i.id, i.label])} />
      </Row>
      <Row label="Range" htmlFor="crange" hint={rangeOk ? '' : 'Adjusted to a range this interval supports.'}>
        <Select id="crange" value={String(rangeOk ? prefs.chartDays : iv.ranges[0])} onChange={(v) => set('chartDays', Number(v))}
          options={iv.ranges.map((d) => [String(d), RANGE_LABEL(d)])} />
      </Row>
      <Row label="Chart type">
        <Segmented label="Chart type" value={prefs.chartType} onChange={(v) => set('chartType', v)} options={CHART_TYPES} />
      </Row>
      <Row label="Volume bars">
        <Toggle label="Show volume" checked={prefs.chartVolume} onChange={(v) => set('chartVolume', v)} />
      </Row>
      <Row label="Overlays" hint="Computed server-side and unit-tested; VWAP shows on intraday intervals only.">
        {OVERLAYS.map((o) => (
          <label key={o.id} className={`inline-flex items-center gap-1.5 px-2 py-1 border cursor-pointer font-mono text-[10px] ${prefs.chartOverlays.includes(o.id) ? 'border-primary text-on-surface' : 'border-outline-variant text-outline'}`}>
            <input type="checkbox" className="sr-only" checked={prefs.chartOverlays.includes(o.id)} onChange={() => toggleIn('chartOverlays', o.id)} />
            <span className="w-2 h-2 rounded-full" style={{ background: o.color }} /> {o.label}
          </label>
        ))}
      </Row>
      <Row label="Indicator panes">
        {PANES.map((p) => (
          <label key={p.id} className={`inline-flex items-center gap-1.5 px-2 py-1 border cursor-pointer font-mono text-[10px] ${prefs.chartPanes.includes(p.id) ? 'border-primary text-on-surface' : 'border-outline-variant text-outline'}`}>
            <input type="checkbox" className="sr-only" checked={prefs.chartPanes.includes(p.id)} onChange={() => toggleIn('chartPanes', p.id)} />
            {p.label}
          </label>
        ))}
      </Row>
      <Row label="Live tick" hint="Moves the last candle with each Kite quote while the market is open.">
        <Toggle label="Live tick" checked={prefs.chartLiveTick} onChange={(v) => set('chartLiveTick', v)} />
      </Row>
    </Section>
  );
}

// ── System status ──────────────────────────────────────────────────────────────

function SystemSection() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await apiFetch('/api/system/status', { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setS(await r.json());
      setErr(null);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  const llmLabel = { groq: 'Groq (gpt-oss)', gemini: 'Google Gemini', ollama: 'Ollama (local fallback)' };
  return (
    <Section id="system" title="Data & system status" icon={Server}
      description="What the server can reach right now. API keys stay on the server and are never shown.">
      {err && <Row label="Status"><Pill>Unreachable</Pill><span className="font-mono text-[11px] text-error">{err}</span></Row>}
      {s && (
        <>
          <Row label="Live market data">
            {s.kite.has_current_session ? <Pill ok>Kite · live</Pill> : <Pill warn>Yahoo · delayed</Pill>}
            <span className="font-mono text-[10px] text-outline">{s.kite.has_current_session ? 'Exchange-grade quotes and minute bars' : 'Connect Kite above for live data'}</span>
          </Row>
          <Row label="Live quant engine" hint="Out-of-sample portfolio and pair metrics.">
            <span className="font-mono text-xs text-on-surface-variant">
              {s.live_engine.version} · data as of {s.live_engine.as_of ?? '—'} · computed {fmtDate(s.live_engine.computed_at)}
            </span>
            {s.live_engine.last_error && <span className="w-full font-mono text-[10px] text-error">Last refresh failed: {s.live_engine.last_error}</span>}
          </Row>
          <Row label="Report-writing models" hint="Narrate validated facts only; they never compute numbers.">
            <div className="flex flex-col gap-1">
              {Object.entries(s.llm).map(([name, p]) => (
                <span key={name} className="font-mono text-[11px] text-on-surface-variant flex items-center gap-2">
                  {!p.configured ? <Pill warn>Off</Pill> : p.status === 'available' ? <Pill ok>Ready</Pill> : <Pill warn>Cooling</Pill>}
                  {llmLabel[name] ?? name}
                  {p.available_in_seconds != null && <span className="text-outline">· quota resets in {fmtDuration(p.available_in_seconds)}</span>}
                </span>
              ))}
            </div>
          </Row>
          <Row label="Fact store"><span className="font-mono text-xs text-on-surface-variant capitalize">{s.factstore_backend}</span></Row>
          <Row label="Environment">
            <span className="font-mono text-xs text-on-surface-variant">{s.environment}</span>
            <Pill ok={s.auth_required} warn={!s.auth_required}>{s.auth_required ? 'API auth enforced' : 'API auth not enforced'}</Pill>
          </Row>
        </>
      )}
      <div className="px-4 md:px-5 py-3">
        <Button onClick={load} disabled={loading}><RefreshCw size={11} className={loading ? 'animate-spin' : ''} /> Re-check</Button>
      </div>
    </Section>
  );
}

// ── AI model keys (bring your own key) ─────────────────────────────────────────

function AiKeysSection({ session }) {
  const [saved, setSaved] = useState(getLlmKeys);
  const [draft, setDraft] = useState({ groq: '', gemini: '' });
  const [remember, setRemember] = useState(isRemembered);
  const [policy, setPolicy] = useState(null);
  const [notice, setNotice] = useState(null);
  const [checking, setChecking] = useState(false);
  const auth = useMemo(
    () => (session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {}),
    [session?.access_token],
  );

  useEffect(() => {
    apiFetch('/api/system/llm-policy', { headers: auth, cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null)).then(setPolicy).catch(() => setPolicy(null));
  }, [auth]);

  const save = () => {
    try {
      const next = { groq: draft.groq.trim() || saved.groq || '', gemini: draft.gemini.trim() || saved.gemini || '' };
      setLlmKeys(next, remember);
      setSaved(getLlmKeys());
      setDraft({ groq: '', gemini: '' });
      setNotice({ ok: true, text: remember ? 'Saved in this browser (remembered on this device).' : 'Saved for this browser tab only.' });
    } catch (e) {
      setNotice({ ok: false, text: e.message });
    }
  };

  const forget = (provider) => {
    const next = { ...saved };
    if (provider) delete next[provider]; else { clearLlmKeys(); setSaved({}); setNotice({ ok: true, text: 'Keys removed from this browser.' }); return; }
    setLlmKeys(next, remember);
    setSaved(getLlmKeys());
  };

  const check = async () => {
    setChecking(true);
    setNotice(null);
    try {
      const headers = { ...auth };
      for (const [p, f] of Object.entries(KEY_FORMATS)) if (saved[p]) headers[f.header] = saved[p];
      const r = await apiFetch('/api/system/llm-key-check', { method: 'POST', headers });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body?.detail?.message || `HTTP ${r.status}`);
      const parts = Object.entries(body.results).map(([p, v]) => `${KEY_FORMATS[p].label}: ${v.message}`);
      setNotice({ ok: Object.values(body.results).every((v) => v.ok), text: parts.join(' · ') });
    } catch (e) {
      setNotice({ ok: false, text: `Check failed: ${e.message}` });
    } finally {
      setChecking(false);
    }
  };

  const demo = policy?.demo;
  const policyText = !policy ? 'Checking…'
    : policy.owner_keys === 'off' ? 'This deployment does not provide AI keys. Add your own below to use AI reports.'
    : policy.owner_keys === 'demo' ? `Demo: ${demo?.visitor_remaining ?? 0} of ${demo?.per_visitor ?? 0} free AI reports left for you today (shared limit ${demo?.day_remaining ?? 0} left). Your own key removes the limit.`
    : 'AI reports use the server’s keys. You can still use your own.';

  return (
    <Section id="ai-keys" title="AI model keys" icon={KeyRound}
      description="Optional. Use your own free Groq or Gemini key for AI-written reports, so you don’t depend on the shared demo quota.">
      <Row label="On this deployment"><span className="font-mono text-[11px] text-on-surface-variant">{policyText}</span></Row>
      {Object.entries(KEY_FORMATS).map(([p, f]) => (
        <Row key={p} label={`${f.label} key`} htmlFor={`key-${p}`}
          hint={<a href={f.help} target="_blank" rel="noopener noreferrer" className="underline">Get a free key</a>}>
          {saved[p] ? (
            <span className="flex items-center gap-2 font-mono text-xs text-on-surface-variant">
              <Pill ok>Saved</Pill>{maskKey(saved[p])}
              <Button onClick={() => forget(p)} tone="danger"><Trash2 size={11} /> Remove</Button>
            </span>
          ) : (
            <input id={`key-${p}`} type="password" autoComplete="off" spellCheck={false} value={draft[p]}
              onChange={(e) => setDraft((d) => ({ ...d, [p]: e.target.value }))}
              placeholder={p === 'groq' ? 'gsk_…' : 'AIza…'}
              className="bg-surface border border-outline-variant px-3 py-1.5 text-xs font-mono text-on-surface w-full max-w-sm focus:outline-none focus:border-primary" />
          )}
        </Row>
      ))}
      <Row label="Keep keys" hint="Off: forgotten when this tab closes. On: kept in this browser until you remove them.">
        <Toggle checked={remember} onChange={setRemember} label="Remember on this device" />
      </Row>
      <div className="px-4 md:px-5 py-3 flex flex-wrap gap-2 items-center">
        <Button onClick={save} disabled={!draft.groq.trim() && !draft.gemini.trim() && remember === isRemembered()}><Check size={11} /> Save</Button>
        <Button onClick={check} disabled={checking || !Object.keys(saved).length}><RefreshCw size={11} className={checking ? 'animate-spin' : ''} /> Test keys</Button>
        {Object.keys(saved).length > 0 && <Button onClick={() => forget()} tone="danger"><Trash2 size={11} /> Remove all</Button>}
        {notice && <span className={`font-mono text-[11px] ${notice.ok ? 'text-secondary' : 'text-error'}`}>{notice.text}</span>}
      </div>
      <p className="px-4 md:px-5 pb-3 font-mono text-[10px] text-outline">
        Keys are stored only in this browser and sent only to this ARTHA server, only when an AI model runs. The server uses them for that request and never saves or logs them. Anyone with access to this browser profile can read them.
      </p>
    </Section>
  );
}

// ── Safeguards (read-only) ─────────────────────────────────────────────────────

function SafeguardsSection() {
  const [gate, setGate] = useState(null);
  useEffect(() => {
    apiFetch('/api/system/status').then((r) => (r.ok ? r.json() : null)).then((s) => setGate(s)).catch(() => {});
  }, []);
  const g = gate?.validation_gate;
  return (
    <Section id="safeguards" title="Safeguards" icon={ShieldCheck}
      description="Fixed by design. They can't be changed from the UI, so a backtest can't be tuned until it looks good.">
      <Row label="Validation gate" hint="A signal is only reported after clearing all three on out-of-sample data.">
        {g ? (
          <div className="flex flex-wrap gap-2 font-mono text-[11px]">
            <span className="px-2 py-1 border border-outline-variant text-on-surface">Sharpe ≥ {g.min_oos_sharpe.toFixed(2)}</span>
            <span className="px-2 py-1 border border-outline-variant text-on-surface">Max drawdown ≤ {(g.max_oos_drawdown * 100).toFixed(0)}%</span>
            <span className="px-2 py-1 border border-outline-variant text-on-surface">≥ {g.min_oos_sessions} sessions</span>
          </div>
        ) : <span className="font-mono text-xs text-outline">—</span>}
      </Row>
      <Row label="Order placement" hint="Signals and simulations only. No code path sends orders to a broker.">
        <Pill warn>Disabled</Pill>
      </Row>
      <Row label="Report numbers" hint="Every figure in a generated report must cite a Fact Store entry, or the report is rejected.">
        <Pill ok>Citation check on</Pill>
      </Row>
    </Section>
  );
}

// ── Your data ──────────────────────────────────────────────────────────────────

function DataSection({ prefs }) {
  const fileRef = useRef(null);
  const [notice, setNotice] = useState(null);

  const exportPrefs = () => {
    const blob = new Blob([JSON.stringify({ artha_preferences: 1, exported_at: new Date().toISOString(), preferences: prefs }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'artha-preferences.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const importPrefs = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    try {
      if (file.size > 64 * 1024) throw new Error('File is too large to be an ARTHA preferences export.');
      const parsed = JSON.parse(await file.text());
      const rejected = importPreferences(parsed?.preferences ?? parsed);
      setNotice({ ok: true, text: rejected.length ? `Imported; ignored ${rejected.join(', ')}.` : 'Preferences imported.' });
    } catch (err) {
      setNotice({ ok: false, text: err.message || 'Not a valid preferences file.' });
    }
  };

  const reset = () => {
    if (!window.confirm('Reset every display preference to its default?')) return;
    resetPreferences();
    setNotice({ ok: true, text: 'Preferences reset to defaults.' });
  };

  const clearAll = () => {
    if (!window.confirm('Remove all ARTHA data stored in this browser (preferences and remembered instrument)? Your account, watchlists and the server are not affected.')) return;
    try {
      Object.keys(window.localStorage).filter((k) => k.startsWith('artha.')).forEach((k) => window.localStorage.removeItem(k));
    } catch { /* storage unavailable */ }
    resetPreferences();
    clearActiveSymbol();
    setNotice({ ok: true, text: 'Local ARTHA data cleared.' });
  };

  const changed = Object.keys(DEFAULT_PREFERENCES).filter((k) => JSON.stringify(prefs[k]) !== JSON.stringify(DEFAULT_PREFERENCES[k])).length;
  return (
    <Section id="data" title="Your data" icon={HardDrive}
      description="Preferences live in this browser only. Watchlists are stored with your account.">
      <Row label="Preferences" hint={`${changed} changed from default.`}>
        <Button onClick={exportPrefs}><Download size={11} /> Export</Button>
        <Button onClick={() => fileRef.current?.click()}><Upload size={11} /> Import</Button>
        <input ref={fileRef} type="file" accept="application/json,.json" className="hidden" onChange={importPrefs} />
        <Button onClick={reset} disabled={changed === 0}><RotateCcw size={11} /> Reset to defaults</Button>
      </Row>
      <Row label="Clear local data" hint="Signs nothing out and deletes nothing on the server.">
        <Button tone="danger" onClick={clearAll}><Trash2 size={11} /> Clear this browser</Button>
      </Row>
      {notice && <div className="px-4 md:px-5 py-3"><Notice notice={notice} /></div>}
    </Section>
  );
}

// ── About ──────────────────────────────────────────────────────────────────────

function AboutSection() {
  return (
    <Section id="about" title="About" icon={Info}>
      <div className="px-4 md:px-5 py-4 space-y-2 font-mono text-[11px] text-on-surface-variant">
        <p>
          ARTHA describes validated quantitative signals and their historical, out-of-sample performance.
          It is <span className="text-on-surface">not investment advice</span> and does not tell you what to buy or sell.
        </p>
        <p className="text-outline">
          Market data: Zerodha Kite Connect when connected, otherwise Yahoo Finance (delayed) and NSE archives.
          News: public RSS feeds, scored locally. Past performance does not predict future results.
        </p>
      </div>
    </Section>
  );
}

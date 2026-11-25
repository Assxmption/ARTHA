// The instrument the user last opened, shared by every symbol-based tab
// (chart, fundamentals, sentiment). A per-browser convenience, so it lives in
// localStorage; every access is guarded because storage can be unavailable
// (private mode, blocked site data) and the app must still work.
import { useEffect, useState } from 'react';
import { getPreferences } from './preferences';

const KEY = 'artha.activeSymbol';
const EVENT = 'artha:active-symbol';
const fallback = () => getPreferences().defaultSymbol || 'RELIANCE'; // Settings → Start-up

const clean = (s) => (typeof s === 'string' ? s.trim().toUpperCase().replace(/[^A-Z0-9&^._-]/g, '').slice(0, 20) : '');

export function getActiveSymbol() {
  try {
    return clean(window.localStorage.getItem(KEY)) || fallback();
  } catch {
    return fallback();
  }
}

export function setActiveSymbol(symbol) {
  const s = clean(symbol);
  if (!s) return;
  try {
    window.localStorage.setItem(KEY, s);
  } catch { /* storage unavailable: remember for this page view only */ }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: s }));
}

/** Forget the remembered instrument (tabs fall back to the default symbol). */
export function clearActiveSymbol() {
  try {
    window.localStorage.removeItem(KEY);
  } catch { /* nothing stored */ }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: '' }));
}

/** Current active symbol; re-renders when any page changes it. */
export function useActiveSymbol() {
  const [symbol, setSymbol] = useState(getActiveSymbol);
  useEffect(() => {
    const onChange = (e) => setSymbol(clean(e.detail) || getActiveSymbol());
    const onStorage = (e) => { if (e.key === KEY) setSymbol(getActiveSymbol()); };
    window.addEventListener(EVENT, onChange);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT, onChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);
  return symbol;
}

/** Call from pages that show one instrument so other tabs follow it. */
export function useRememberSymbol(symbol) {
  useEffect(() => {
    if (symbol) setActiveSymbol(symbol);
  }, [symbol]);
}

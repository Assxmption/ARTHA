// Per-browser display preferences. They change how ARTHA looks and what it
// opens with, never what the quant engine computes, so they live in
// localStorage (guarded: storage can be unavailable and the app must still work).
import { useSyncExternalStore } from 'react';

const KEY = 'artha.preferences.v1';
const EVENT = 'artha:preferences';

export const DEFAULT_PREFERENCES = Object.freeze({
  // Appearance
  theme: 'dark',              // 'dark' | 'light' | 'system'
  tickerEnabled: true,
  tickerScroll: true,         // false → static row (also forced off by OS reduce-motion)
  tickerSpeed: 'normal',      // 'slow' | 'normal' | 'fast'
  tickerContent: 'all',       // 'indices' | 'all' (indices + every stock in the universe)
  // Start-up
  landingPage: '/home',       // where sign-in lands
  defaultSymbol: 'RELIANCE',  // used until you open an instrument
  // Market overview
  moversCount: 5,
  // Chart defaults
  chartInterval: '1d',
  chartDays: 180,
  chartType: 'candle',
  chartVolume: true,
  chartOverlays: ['sma20'],
  chartPanes: [],
  chartLiveTick: true,
});

const ALLOWED = {
  theme: ['dark', 'light', 'system'],
  tickerSpeed: ['slow', 'normal', 'fast'],
  tickerContent: ['indices', 'all'],
  landingPage: ['/home', '/dashboard', '/signals', '/watchlist', '/screener', '/paper-trading'],
  moversCount: [3, 5, 10],
  chartType: ['candle', 'ha', 'line', 'area'],
};

const cleanSymbol = (s) =>
  typeof s === 'string' ? s.trim().toUpperCase().replace(/[^A-Z0-9&^._-]/g, '').slice(0, 20) : '';

// Validate everything read back from storage: it may be from an older version
// or hand-edited, and a bad value must fall back rather than break a page.
function sanitize(raw) {
  const out = { ...DEFAULT_PREFERENCES };
  if (!raw || typeof raw !== 'object') return out;
  for (const [k, def] of Object.entries(DEFAULT_PREFERENCES)) {
    const v = raw[k];
    if (v === undefined) continue;
    if (ALLOWED[k]) { if (ALLOWED[k].includes(v)) out[k] = v; continue; }
    if (typeof def === 'boolean') { if (typeof v === 'boolean') out[k] = v; continue; }
    if (k === 'defaultSymbol') { out[k] = cleanSymbol(v) || def; continue; }
    if (k === 'chartDays') { if (Number.isInteger(v) && v > 0 && v <= 3650) out[k] = v; continue; }
    if (k === 'chartInterval') { if (typeof v === 'string' && /^[0-9]+(m|d|wk|mo)$/.test(v)) out[k] = v; continue; }
    if (Array.isArray(def)) { if (Array.isArray(v)) out[k] = v.filter((x) => typeof x === 'string' && /^[a-z0-9]+$/.test(x)).slice(0, 12); continue; }
  }
  return out;
}

let cache = null;
function read() {
  if (cache) return cache;
  try {
    cache = sanitize(JSON.parse(window.localStorage.getItem(KEY) || 'null'));
  } catch {
    cache = { ...DEFAULT_PREFERENCES };
  }
  return cache;
}

function write(next) {
  cache = sanitize(next);
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cache));
  } catch { /* storage unavailable: keep for this page view */ }
  window.dispatchEvent(new Event(EVENT));
}

export const getPreferences = read;

export function setPreference(key, value) {
  write({ ...read(), [key]: value });
}

export function setPreferences(patch) {
  write({ ...read(), ...patch });
}

export function resetPreferences() {
  write({ ...DEFAULT_PREFERENCES });
}

/** Import a previously exported object; returns the keys that were rejected. */
export function importPreferences(obj) {
  const clean = sanitize(obj);
  const rejected = Object.keys(obj || {}).filter(
    (k) => !(k in DEFAULT_PREFERENCES) || JSON.stringify(clean[k]) !== JSON.stringify(obj[k]),
  );
  write(clean);
  return rejected;
}

function subscribe(cb) {
  const onStorage = (e) => { if (e.key === KEY) { cache = null; cb(); } };
  window.addEventListener(EVENT, cb);
  window.addEventListener('storage', onStorage);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener('storage', onStorage);
  };
}

export function usePreferences() {
  return useSyncExternalStore(subscribe, read, () => DEFAULT_PREFERENCES);
}

export function prefersReducedMotion() {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

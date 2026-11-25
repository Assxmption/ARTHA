// Visitor-supplied LLM keys (bring your own key).
//
// Keys stay in this browser: sessionStorage by default (gone when the tab
// closes), localStorage only if the visitor ticks "remember on this device".
// They are sent only to this ARTHA server, only on requests that run an AI
// model, in X-Groq-Api-Key / X-Gemini-Api-Key headers. The server uses them
// for that request and never stores or logs them.

const STORE_KEY = 'artha.llmKeys.v1';

export const KEY_FORMATS = {
  groq: { pattern: /^gsk_[A-Za-z0-9]{20,120}$/, header: 'X-Groq-Api-Key', label: 'Groq', help: 'https://console.groq.com/keys' },
  gemini: { pattern: /^AIza[0-9A-Za-z_-]{30,60}$/, header: 'X-Gemini-Api-Key', label: 'Google Gemini', help: 'https://aistudio.google.com/apikey' },
};

function read(storage) {
  try {
    const v = JSON.parse(storage.getItem(STORE_KEY) || 'null');
    if (!v || typeof v !== 'object') return null;
    const out = {};
    for (const p of Object.keys(KEY_FORMATS)) {
      if (typeof v[p] === 'string' && KEY_FORMATS[p].pattern.test(v[p])) out[p] = v[p];
    }
    return out;
  } catch {
    return null;
  }
}

export function getLlmKeys() {
  return read(window.sessionStorage) || read(window.localStorage) || {};
}

export function isRemembered() {
  try { return !!window.localStorage.getItem(STORE_KEY); } catch { return false; }
}

export function setLlmKeys(keys, remember) {
  const clean = {};
  for (const [p, f] of Object.entries(KEY_FORMATS)) {
    const v = (keys[p] || '').trim();
    if (v) {
      if (!f.pattern.test(v)) throw new Error(`${f.label} key is not in the expected format`);
      clean[p] = v;
    }
  }
  clearLlmKeys();
  if (!Object.keys(clean).length) return;
  try {
    (remember ? window.localStorage : window.sessionStorage).setItem(STORE_KEY, JSON.stringify(clean));
  } catch { /* storage unavailable: keys simply aren't kept */ }
}

export function clearLlmKeys() {
  for (const s of [window.sessionStorage, window.localStorage]) {
    try { s.removeItem(STORE_KEY); } catch { /* ignore */ }
  }
}

// Headers for a request that runs an AI model. Empty when no key is saved.
export function llmKeyHeaders() {
  const keys = getLlmKeys();
  const h = {};
  for (const [p, f] of Object.entries(KEY_FORMATS)) if (keys[p]) h[f.header] = keys[p];
  return h;
}

export const maskKey = (k) => (k ? `${k.slice(0, 4)}…${k.slice(-4)}` : '');

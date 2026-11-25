// One way to call the ARTHA backend: same-origin /api paths only, with the
// signed-in user's Supabase bearer token attached when there is one, so the
// backend can enforce auth uniformly (route guards in the UI are cosmetic).
import { supabase } from './supabase';
import { llmKeyHeaders } from './llmKeys';

export async function authHeaders() {
  try {
    const { data } = await supabase.auth.getSession();
    const token = data?.session?.access_token;
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}

// apiFetch('/api/x', { method, body, llm: true }) — `llm: true` adds the visitor's
// own AI keys (Settings → AI model keys) for requests that run a model.
export async function apiFetch(path, init = {}) {
  if (typeof path !== 'string' || !path.startsWith('/api/')) {
    throw new Error('apiFetch only calls this server’s /api endpoints');
  }
  const { llm, headers, ...rest } = init;
  const merged = new Headers(headers || {});
  if (!merged.has('Authorization')) {
    for (const [k, v] of Object.entries(await authHeaders())) merged.set(k, v);
  }
  if (llm) for (const [k, v] of Object.entries(llmKeyHeaders())) merged.set(k, v);
  return fetch(path, { ...rest, headers: merged });
}

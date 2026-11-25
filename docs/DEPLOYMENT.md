# Deploying ARTHA

ARTHA ships as one Docker image: the FastAPI backend serves the built React app
from the same origin, so there is no separate frontend host, no CORS between
them, and one set of security headers.

## Who pays for AI (your keys vs. visitors' keys)

AI-written reports and fundamentals narratives call Groq/Gemini. Two mechanisms
work together (app/llm/keys.py):

| `LLM_OWNER_KEYS` | Effect | Use it for |
|---|---|---|
| `demo` (production default) | Your keys, capped per visitor (`DEMO_JOBS_PER_VISITOR`) and per day (`DEMO_JOBS_PER_DAY`) | The showcase |
| `off` | Your keys are never used for visitors | Going fully public |
| `unlimited` | Your keys for everyone | Local development only |

Visitors can always add **their own** free Groq or Gemini key in
*Settings → AI model keys*. It stays in their browser (session-only unless they
tick "remember"), is sent only with requests that run a model, is never stored
or logged by the server, and never falls back to your quota.

Fundamentals narratives are cached per symbol per day and shared by all
visitors, so page views don't re-spend quota.

**Showcase → public:** set `LLM_OWNER_KEYS=off`, or delete `GROQ_API_KEY` /
`GEMINI_API_KEY` from the host. No code change or redeploy of code needed.

## Required settings

| Variable | Value |
|---|---|
| `APP_ENV` | `production` (the image sets it) |
| `AUTH_REQUIRED` | `true` (the image sets it; production refuses to start otherwise) |
| `SUPABASE_JWT_SECRET` | Supabase → Project settings → API → JWT secret (legacy HS256) |
| `SUPABASE_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` | Your Supabase project (the `VITE_*` two are public and must be available at **build** time) |
| `ADMIN_EMAILS` | Your sign-in email. Only admins can connect/disconnect Kite, run the paper engine, or rebuild caches |
| `ALLOWED_ORIGINS` | Your public URL, e.g. `https://artha.onrender.com` |
| `TRUSTED_PROXY_HOPS` | `1` on Render / Hugging Face / Railway; `2` with Cloudflare in front |
| `KITE_REDIRECT_URL` | `https://<your-host>/api/kite/callback` (and the same URL in the Kite developer console) |

## Before going live: checklist

1. **Rotate every key that was ever on a machine running an older build.**
   Builds before 2026-11-25 had a path-traversal bug that served any file
   (including `.env` and the Kite token) to anyone who could reach the server.
   The local dev server listened on 127.0.0.1 only, so exposure needed local
   access — but if any older build was ever reachable from a network, treat
   Groq, Gemini, OpenAI/Serper, the Kite API secret, and the Supabase JWT secret
   as compromised and rotate them.
2. Row-level security on `watchlists` (users see only their own lists) was
   verified on the live project on 2026-11-25. `supabase/rls_watchlists.sql`
   records the policies; run it only for a new Supabase project.
3. In Supabase → Authentication: require email confirmation; set the minimum
   password length to 12; enable "Secure password change". If this is a
   personal showcase, consider disabling public sign-ups.
4. Set the variables above. Keep secrets in the host's secret store, never in git.
5. After deploy, check: `/api/health` is 200, `/docs` and `/openapi.json` are 404,
   and *Settings → Data & system status* shows "API auth enforced".

## Choosing a free host (checked 2026-11-25 — verify current terms)

| Host | Free resources | Fit for ARTHA |
|---|---|---|
| [Render](https://render.com/docs/free) | 512 MB RAM, 0.1 CPU; sleeps after 15 min idle (≈1 min cold start); ephemeral disk | Works for a demo (`render.yaml` included). The paper engine stops while asleep and its ledger resets on restart. AI reports load crewai and may hit 512 MB. |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/en/spaces-overview) (Docker) | Docs: 2 vCPU, 16 GB RAM, 50 GB non-persistent disk; sleeps after 48 h idle | Best resources if your account still gets free CPU Basic — community reports from mid-2026 say free access was restricted. Set `app_port: 8000` in the Space README. |
| Oracle Cloud Always Free (VM) | Small always-on VM | The only $0 option where the paper engine runs 24/7; more setup (`docker compose up -d` behind Caddy/nginx for HTTPS). |

**State on free hosts:** the paper-trading ledger (`data_cache/paper/ledger.db`)
and the Kite session live on local disk. On hosts with ephemeral disks they are
lost on every restart or sleep. Use a persistent volume, or move the ledger to
Postgres (not yet supported) before relying on long-running paper results. The
fact store already supports Postgres via `DATABASE_URL`.

## Dependencies

`requirements.lock` is hash-pinned and the image installs it with
`--require-hashes`. After changing `requirements.txt`:

```bash
uv pip compile requirements.txt --universal --python-version 3.12 --generate-hashes -o requirements.lock
uvx pip-audit -r requirements.lock --no-deps --disable-pip
```

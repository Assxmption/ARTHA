-- Row-level security for ARTHA watchlists.
--
-- The browser talks to Supabase directly with the public anon key, so these
-- policies are the ONLY thing stopping one user from reading or deleting
-- another user's watchlists. Run once in the Supabase SQL editor; it is
-- idempotent. Verify afterwards: Table editor → watchlists → "RLS enabled".

alter table public.watchlists enable row level security;
alter table public.watchlists force row level security;

drop policy if exists "watchlists_select_own" on public.watchlists;
drop policy if exists "watchlists_insert_own" on public.watchlists;
drop policy if exists "watchlists_update_own" on public.watchlists;
drop policy if exists "watchlists_delete_own" on public.watchlists;

create policy "watchlists_select_own" on public.watchlists
  for select to authenticated using (user_id = auth.uid());

create policy "watchlists_insert_own" on public.watchlists
  for insert to authenticated with check (user_id = auth.uid());

create policy "watchlists_update_own" on public.watchlists
  for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

create policy "watchlists_delete_own" on public.watchlists
  for delete to authenticated using (user_id = auth.uid());

-- Anonymous (anon-key-only) requests get nothing.
revoke all on public.watchlists from anon;

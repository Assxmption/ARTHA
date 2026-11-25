-- Row-level security for ARTHA watchlists.
--
-- The browser talks to Supabase directly with the public anon key, so these
-- policies are the ONLY thing stopping one user from reading or deleting
-- another user's watchlists.
--
-- Status (checked in the live project on 2026-11-25): RLS is enabled and these
-- exact four policies already exist. This file records them so the setup is
-- reproducible; it is idempotent and uses the existing policy names, so running
-- it again replaces them instead of adding duplicates.
--
-- Note on UPDATE: with no WITH CHECK clause, Postgres applies the USING
-- expression to the new row too, so a user cannot reassign a list to someone
-- else. Anonymous requests have auth.uid() = NULL and match no policy.

alter table public.watchlists enable row level security;

drop policy if exists "Users can view their own watchlists" on public.watchlists;
drop policy if exists "Users can insert their own watchlists" on public.watchlists;
drop policy if exists "Users can update their own watchlists" on public.watchlists;
drop policy if exists "Users can delete their own watchlists" on public.watchlists;

create policy "Users can view their own watchlists" on public.watchlists
  for select using (auth.uid() = user_id);

create policy "Users can insert their own watchlists" on public.watchlists
  for insert with check (auth.uid() = user_id);

create policy "Users can update their own watchlists" on public.watchlists
  for update using (auth.uid() = user_id);

create policy "Users can delete their own watchlists" on public.watchlists
  for delete using (auth.uid() = user_id);

-- Table privileges for `anon` are left in place on purpose: the keep-alive
-- workflow pings this table with the anon key and expects HTTP 200 + [].

-- Verify: expect rls_enabled = true and the four policies above.
select relname, relrowsecurity as rls_enabled
from pg_class where oid = 'public.watchlists'::regclass;
select policyname, cmd, roles, qual, with_check
from pg_policies where schemaname = 'public' and tablename = 'watchlists' order by policyname;

-- "Is there an ARTHA account for this email?" — for friendlier Forgot-password
-- and email-code sign-in messages.
--
-- Trade-off: this lets a caller learn whether an email is registered (account
-- enumeration). Supabase's email-code sign-in already reveals the same thing, so
-- the new exposure is small, and it is throttled here:
--   * 10 checks per hour per client IP (Cloudflare's cf-connecting-ip, which a
--     client cannot spoof through Supabase's edge), and
--   * 300 checks per hour across everyone (caps distributed scraping).
-- Over the limit the function raises 'rate_limited' and the app falls back to
-- neutral wording. Attempt rows are kept for one day, then deleted.
--
-- Idempotent: safe to run again. Run in the Supabase SQL editor.

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

create table if not exists private.email_check_attempts (
  ip text not null,
  at timestamptz not null default now()
);
create index if not exists email_check_attempts_ip_at on private.email_check_attempts (ip, at);
create index if not exists email_check_attempts_at on private.email_check_attempts (at);
-- Defence in depth: the schema is not exposed to the API and only the
-- security-definer function (run as the table owner) touches this table.
alter table private.email_check_attempts enable row level security;

create or replace function public.email_has_account(p_email text)
returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_headers json := coalesce(nullif(current_setting('request.headers', true), ''), '{}')::json;
  v_ip text := coalesce(
    nullif(v_headers ->> 'cf-connecting-ip', ''),
    nullif(trim(split_part(coalesce(v_headers ->> 'x-forwarded-for', ''), ',', 1)), ''),
    'unknown'
  );
  v_email text := lower(trim(coalesce(p_email, '')));
begin
  if length(v_email) = 0 or length(v_email) > 254 or position('@' in v_email) = 0 then
    return false;
  end if;

  delete from private.email_check_attempts where at < now() - interval '1 day';

  if (select count(*) from private.email_check_attempts
        where ip = v_ip and at > now() - interval '1 hour') >= 10
     or (select count(*) from private.email_check_attempts
        where at > now() - interval '1 hour') >= 300 then
    raise exception 'rate_limited' using errcode = 'P0001';
  end if;

  insert into private.email_check_attempts (ip) values (v_ip);

  return exists (
    select 1 from auth.users
    where lower(email) = v_email and deleted_at is null
  );
end;
$$;

revoke all on function public.email_has_account(text) from public;
grant execute on function public.email_has_account(text) to anon, authenticated;

-- Verify (expect: true for a registered address, false for an unknown one):
-- select public.email_has_account('someone@example.com');

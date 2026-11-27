// Shared auth rules for the sign-in page and Settings.

// Supabase project setting "Email OTP length" (Authentication → Sign In / Providers → Email).
// Codes are accepted at any length 6–10; this only drives the placeholder and auto-submit.
export const OTP_LENGTH = Math.min(10, Math.max(6, Number(import.meta.env.VITE_AUTH_OTP_LENGTH) || 8));
export const RESEND_COOLDOWN_S = 60;
// "an 8-digit" / "a 6-digit" (8 is the only allowed length that takes "an").
export const A_CODE = `${OTP_LENGTH === 8 ? 'an' : 'a'} ${OTP_LENGTH}-digit`;

export function passwordRules(pw, confirm) {
  return [
    [pw.length >= 12, 'At least 12 characters'],
    [/[A-Za-z]/.test(pw) && /[0-9]/.test(pw), 'Letters and numbers'],
    [/[^A-Za-z0-9]/.test(pw), 'At least one symbol'],
    ...(confirm === undefined ? [] : [[pw.length > 0 && pw === confirm, 'Both entries match']]),
  ];
}

export const passwordValid = (pw, confirm) => passwordRules(pw, confirm).every(([ok]) => ok);

export const cleanCode = (v) => (v || '').replace(/\D/g, '').slice(0, 10);

// Supabase error → something a person can act on.
export function friendlyAuthError(err) {
  const msg = (err?.message || String(err || '')).toLowerCase();
  const code = err?.code || '';
  if (code === 'over_email_send_rate_limit' || msg.includes('rate limit') || err?.status === 429) {
    return 'Too many emails were requested. Wait a minute and try again.';
  }
  if (code === 'invalid_credentials' || msg.includes('invalid login credentials')) return 'Wrong email or password.';
  if (code === 'email_not_confirmed' || msg.includes('email not confirmed')) {
    return 'This email is not verified yet. Enter the code we sent, or request a new one.';
  }
  if (code === 'otp_expired' || msg.includes('expired') || msg.includes('token has expired or is invalid')) {
    return 'That code is wrong or has expired. Request a new one.';
  }
  if (code === 'user_already_exists' || msg.includes('already registered')) {
    return 'An account with this email already exists. Sign in instead.';
  }
  if (code === 'otp_disabled' || msg.includes('signups not allowed for otp')) {
    return 'No account uses this email. Create one first.';
  }
  if (code === 'weak_password' || msg.includes('password should')) return err.message;
  if (code === 'same_password' || msg.includes('should be different')) return 'Choose a password different from your current one.';
  if (code === 'reauthentication_needed' || msg.includes('reauthentication')) {
    return 'For security, confirm with the code we email you before changing your password.';
  }
  if (msg.includes('failed to fetch') || msg.includes('network')) return 'Could not reach the sign-in service. Check your connection.';
  return err?.message || 'Something went wrong. Please try again.';
}

import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'

const AuthContext = createContext({})

export const useAuth = () => useContext(AuthContext)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    })

    return () => subscription.unsubscribe()
  }, [])

  const signUp = async (email, password) => {
    return supabase.auth.signUp({
      email,
      password,
    })
  }

  const signIn = async (email, password) => {
    return supabase.auth.signInWithPassword({
      email,
      password,
    })
  }

  const signOut = async () => {
    return supabase.auth.signOut()
  }
  
  const resetPassword = async (email) => {
    return supabase.auth.resetPasswordForEmail(email, {
      redirectTo: window.location.origin + '/reset-password',
    })
  }
  
  const verifyOtp = async (email, token, type = 'signup') => {
    return supabase.auth.verifyOtp({
      email,
      token,
      type,
    })
  }

  // Passwordless sign-in: emails a one-time code (and link) to an existing account.
  const sendSignInCode = async (email) => {
    return supabase.auth.signInWithOtp({ email, options: { shouldCreateUser: false } })
  }

  // Re-send the sign-up verification email.
  const resendSignupCode = async (email) => {
    return supabase.auth.resend({ type: 'signup', email })
  }

  // Set a new password. `nonce` is the reauthentication code when Supabase's
  // "Secure password change" is on (see requestReauthCode).
  const updatePassword = async (password, nonce) => {
    return supabase.auth.updateUser(nonce ? { password, nonce } : { password })
  }

  // Does an ARTHA account use this email? Rate-limited database function
  // (supabase/auth_email_check.sql). Returns true/false, or null when the check
  // is unavailable or throttled — callers then fall back to neutral wording.
  const checkEmailHasAccount = async (email) => {
    try {
      const { data, error } = await supabase.rpc('email_has_account', { p_email: email })
      if (error || typeof data !== 'boolean') return null
      return data
    } catch {
      return null
    }
  }

  // Emails the signed-in user a one-time code that proves it's really them.
  const requestReauthCode = async () => {
    return supabase.auth.reauthenticate()
  }

  const value = {
    signUp,
    signIn,
    signOut,
    resetPassword,
    verifyOtp,
    sendSignInCode,
    resendSignupCode,
    updatePassword,
    requestReauthCode,
    checkEmailHasAccount,
    user,
    session,
    loading
  }

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  )
}

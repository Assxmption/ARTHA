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

  const value = {
    signUp,
    signIn,
    signOut,
    resetPassword,
    verifyOtp,
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

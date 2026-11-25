// Manual Supabase check. Run with VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY, TEST_EMAIL, TEST_PASSWORD set.
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(process.env.VITE_SUPABASE_URL, process.env.VITE_SUPABASE_ANON_KEY)

async function test() {
  const email = process.env.TEST_EMAIL
  const password = process.env.TEST_PASSWORD
  
  await supabase.auth.signUp({ email, password })
  const { data: { session }, error: authError } = await supabase.auth.signInWithPassword({ email, password })
  
  if (authError) {
    console.error('Auth error:', authError)
    return
  }
  
  console.log('Logged in as:', session.user.id)
  
  const { data, error } = await supabase.from('watchlists').select('*')
  console.log('Select result:', { data, error })
  
  if (!error) {
    const { data: iData, error: iError } = await supabase.from('watchlists').insert([{
      user_id: session.user.id,
      name: 'Test',
      symbols: []
    }]).select()
    console.log('Insert result:', { data: iData, error: iError })
  }
}

test()

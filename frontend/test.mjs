import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  'https://etajspdgktfyrrwywlkm.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV0YWpzcGRna3RmeXJyd3l3bGttIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk2MjAwMjAsImV4cCI6MjEwNTE5NjAyMH0.-USyhdVHeg1azUFxevtr9ZjdnYIOzqbNKRDF6s3Racw'
)

async function test() {
  const email = 'test_agent_123@example.com'
  const password = 'Password123!'
  
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

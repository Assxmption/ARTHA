// Manual Supabase check. Run with VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY, TEST_EMAIL, TEST_PASSWORD set.
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(process.env.VITE_SUPABASE_URL, process.env.VITE_SUPABASE_ANON_KEY)

async function test() {
  const { data, error } = await supabase.from('watchlists').select('*')
  console.log('Select result:', { data, error })
}

test()

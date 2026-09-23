import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  'https://etajspdgktfyrrwywlkm.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV0YWpzcGRna3RmeXJyd3l3bGttIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk2MjAwMjAsImV4cCI6MjEwNTE5NjAyMH0.-USyhdVHeg1azUFxevtr9ZjdnYIOzqbNKRDF6s3Racw'
)

async function test() {
  const { data, error } = await supabase.from('watchlists').select('*')
  console.log('Select result:', { data, error })
}

test()

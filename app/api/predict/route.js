import { NextResponse } from 'next/server'

export async function POST(req) {
  const { body } = await req.json()

  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body }),
    })

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('API Error:', error)
    return NextResponse.json({ error: 'Failed to fetch predictions' }, { status: 500 })
  }
}

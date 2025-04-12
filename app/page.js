'use client'
import { useState } from 'react'

export default function Home() {
    const [body, setBody] = useState('')
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)

    const handleSubmit = async () => {
        setLoading(true)
        try {
            const res = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ body }),
            })

            const data = await res.json()
            setResults(data)
        } catch (err) {
            console.error('Prediction error:', err)
            setResults({ error: 'Something went wrong' })
        }
        setLoading(false)
    }

    return (
        <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
            <div className="bg-white p-6 rounded-2xl shadow-lg w-full max-w-2xl space-y-4">
                <h1 className="text-2xl font-bold text-center">Rumor Dectection</h1>
                <textarea
                    placeholder="Enter Body"
                    value={body}
                    onChange={(e) => setBody(e.target.value)}
                    className="w-full p-3 border rounded-xl h-32"
                />

                <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl transition"
                >
                    {loading ? 'Predicting...' : 'Predict 🔮'}
                </button>

                {results && (
                    <div className="mt-6 space-y-2">
                        <h2 className="text-xl font-semibold">Predictions:</h2>
                        {results.error ? (
                            <div className="text-red-600">{results.error}</div>
                        ) : (
                            Object.entries(results).map(([model, prediction]) => (
                                <div key={model} className="p-3 bg-gray-50 border rounded-xl">
                                    <strong>{model}:</strong> {prediction}
                                </div>
                            ))
                        )}
                    </div>
                )}
                <div className='flex flex-col gap-2 bg-gray-200 rounded-lg p-2 w-32'>
                    <span>
                        <strong>1</strong> Not Rumor
                    </span>
                    <span>
                        <strong>0</strong> Rumor
                    </span>
                </div>
            </div>
        </main>
    )
}

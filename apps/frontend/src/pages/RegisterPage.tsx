import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

export default function RegisterPage() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!name || !email || !password) {
      setError('Please fill in all fields.')
      return
    }
    if (password !== confirm) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)
    // Mock registration — store a dev token and redirect
    await new Promise(r => setTimeout(r, 400))
    localStorage.setItem('auth_token', 'dev-token-facial-align')
    localStorage.setItem('auth_user', JSON.stringify({ email, name }))
    setLoading(false)
    navigate('/dashboard', { replace: true })
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950" data-testid="register-page">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <svg viewBox="0 0 28 28" fill="none" className="w-12 h-12 mb-3" aria-label="Facial Align">
            <rect width="28" height="28" rx="6" fill="#0f172a" />
            <path d="M14 3 C8.5 3, 5 8, 5 13 C5 19.5, 9 24.5, 14 26.5 C19 24.5, 23 19.5, 23 13 C23 8, 19.5 3, 14 3Z" stroke="#06b6d4" strokeWidth="1.4" fill="none" />
            <path d="M10.5 12.5 Q14 10, 17.5 12.5" stroke="#22d3ee" strokeWidth="1.2" fill="none" strokeLinecap="round" />
            <path d="M9.5 17 Q11.5 20.5, 14 21.5 Q16.5 20.5, 18.5 17" stroke="#22d3ee" strokeWidth="1.2" fill="none" strokeLinecap="round" />
            <circle cx="11.5" cy="11.5" r="0.9" fill="#06b6d4" />
            <circle cx="16.5" cy="11.5" r="0.9" fill="#06b6d4" />
          </svg>
          <h1 className="text-xl font-bold text-slate-100">Create Account</h1>
          <p className="text-sm text-slate-500 mt-1">Facial Align Platform</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4" data-testid="register-form">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-1.5">Full Name</label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Dr. Jane Smith"
              className="input-base w-full"
              data-testid="register-name"
              autoFocus
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1.5">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="surgeon@hospital.org"
              className="input-base w-full"
              data-testid="register-email"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-1.5">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Create password"
              className="input-base w-full"
              data-testid="register-password"
            />
          </div>

          <div>
            <label htmlFor="confirm" className="block text-sm font-medium text-slate-300 mb-1.5">Confirm Password</label>
            <input
              id="confirm"
              type="password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              placeholder="Confirm password"
              className="input-base w-full"
              data-testid="register-confirm"
            />
          </div>

          {error && (
            <p className="text-sm text-red-400" data-testid="register-error">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-2.5 disabled:opacity-50"
            data-testid="register-submit"
          >
            {loading ? 'Creating account...' : 'Create Account'}
          </button>

          <p className="text-center text-sm text-slate-500">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-400 hover:text-cyan-300 transition-colors">
              Sign In
            </Link>
          </p>
        </form>

        <p className="text-center text-xs text-slate-600 mt-4">
          Development mode — no backend authentication
        </p>
      </div>
    </div>
  )
}

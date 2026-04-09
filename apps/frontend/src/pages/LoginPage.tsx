import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { LogIn, AlertCircle, Loader2 } from 'lucide-react'
import { authApi } from '../lib/api'

export default function LoginPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await authApi.login(email, password)
      localStorage.setItem('auth_token', res.access_token)
      localStorage.setItem('refresh_token', res.refresh_token)
      navigate('/dashboard', { replace: true })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleDemoLogin() {
    setError(null)
    setLoading(true)
    try {
      const res = await authApi.login('surgeon@facialign.local', 'surgeon')
      localStorage.setItem('auth_token', res.access_token)
      localStorage.setItem('refresh_token', res.refresh_token)
      navigate('/dashboard', { replace: true })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Demo login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950" data-testid="login-page">
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
          <h1 className="text-xl font-bold text-slate-100">Facial Align</h1>
          <p className="text-sm text-slate-500 mt-1">Craniofacial Surgical Planning</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4" data-testid="login-form">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-950 border border-red-800 px-4 py-3 text-sm text-red-300" data-testid="login-error">
              <AlertCircle size={16} className="shrink-0" />
              {error}
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1.5">Email</label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="surgeon@hospital.org"
              className="input-base w-full"
              data-testid="login-email"
              autoFocus
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-1.5">Password</label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter password"
              className="input-base w-full"
              data-testid="login-password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-2.5 disabled:opacity-50 flex items-center justify-center gap-2"
            data-testid="login-submit"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <LogIn size={16} />}
            Sign In
          </button>

          <div className="relative">
            <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-slate-700" /></div>
            <div className="relative flex justify-center text-xs"><span className="bg-slate-800 px-2 text-slate-500">or</span></div>
          </div>

          <button
            type="button"
            onClick={handleDemoLogin}
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-4 py-2.5 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50 transition-colors"
          >
            Demo Login
          </button>

          <p className="text-center text-sm text-slate-500">
            Don't have an account?{' '}
            <Link to="/register" className="text-cyan-400 hover:text-cyan-300 transition-colors">
              Register
            </Link>
          </p>
        </form>
      </div>
    </div>
  )
}

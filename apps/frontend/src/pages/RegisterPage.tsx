import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { UserPlus, AlertCircle, Loader2 } from 'lucide-react'
import { authApi } from '../lib/api'

export default function RegisterPage() {
  const navigate = useNavigate()
  const [form, setForm] = useState({
    email: '',
    password: '',
    full_name: '',
    institution: '',
    specialty: '',
  })
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  function update(field: string, value: string) {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await authApi.register({
        email: form.email,
        password: form.password,
        full_name: form.full_name,
        role: 'surgeon',
        institution: form.institution || undefined,
        specialty: form.specialty || undefined,
      })
      localStorage.setItem('auth_token', res.access_token)
      localStorage.setItem('refresh_token', res.refresh_token)
      navigate('/dashboard', { replace: true })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
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
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-950 border border-red-800 px-4 py-3 text-sm text-red-300" data-testid="register-error">
              <AlertCircle size={16} className="shrink-0" />
              {error}
            </div>
          )}

          <div>
            <label htmlFor="full_name" className="block text-sm font-medium text-slate-300 mb-1.5">Full Name</label>
            <input
              id="full_name"
              type="text"
              required
              value={form.full_name}
              onChange={e => update('full_name', e.target.value)}
              placeholder="Dr. Jane Smith"
              className="input-base w-full"
              data-testid="register-name"
              autoFocus
            />
          </div>

          <div>
            <label htmlFor="reg-email" className="block text-sm font-medium text-slate-300 mb-1.5">Email</label>
            <input
              id="reg-email"
              type="email"
              required
              value={form.email}
              onChange={e => update('email', e.target.value)}
              placeholder="surgeon@hospital.org"
              className="input-base w-full"
              data-testid="register-email"
            />
          </div>

          <div>
            <label htmlFor="reg-password" className="block text-sm font-medium text-slate-300 mb-1.5">Password</label>
            <input
              id="reg-password"
              type="password"
              required
              minLength={6}
              value={form.password}
              onChange={e => update('password', e.target.value)}
              placeholder="Min. 6 characters"
              className="input-base w-full"
              data-testid="register-password"
            />
          </div>

          <div>
            <label htmlFor="institution" className="block text-sm font-medium text-slate-300 mb-1.5">Institution <span className="text-slate-500">(optional)</span></label>
            <input
              id="institution"
              type="text"
              value={form.institution}
              onChange={e => update('institution', e.target.value)}
              placeholder="University Hospital"
              className="input-base w-full"
              data-testid="register-institution"
            />
          </div>

          <div>
            <label htmlFor="specialty" className="block text-sm font-medium text-slate-300 mb-1.5">Specialty <span className="text-slate-500">(optional)</span></label>
            <input
              id="specialty"
              type="text"
              value={form.specialty}
              onChange={e => update('specialty', e.target.value)}
              placeholder="Oral and Maxillofacial Surgery"
              className="input-base w-full"
              data-testid="register-specialty"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-2.5 disabled:opacity-50 flex items-center justify-center gap-2"
            data-testid="register-submit"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <UserPlus size={16} />}
            Create Account
          </button>

          <p className="text-center text-sm text-slate-500">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-400 hover:text-cyan-300 transition-colors">
              Sign In
            </Link>
          </p>
        </form>
      </div>
    </div>
  )
}

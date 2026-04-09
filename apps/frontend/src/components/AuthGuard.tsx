import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { authApi } from '../lib/api'
import type { UserProfile } from '../lib/api'

export const AuthContext = {
  user: null as UserProfile | null,
}

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate()
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function validate() {
      const token = localStorage.getItem('auth_token')
      if (!token) {
        navigate('/login', { replace: true })
        return
      }

      try {
        const user = await authApi.me()
        if (!cancelled) {
          AuthContext.user = user
          setChecking(false)
        }
      } catch {
        // Token invalid — try refresh
        const refreshToken = localStorage.getItem('refresh_token')
        if (refreshToken) {
          try {
            const res = await authApi.refresh(refreshToken)
            localStorage.setItem('auth_token', res.access_token)
            localStorage.setItem('refresh_token', res.refresh_token)
            const user = await authApi.me()
            if (!cancelled) {
              AuthContext.user = user
              setChecking(false)
            }
            return
          } catch {
            // Refresh also failed
          }
        }
        localStorage.removeItem('auth_token')
        localStorage.removeItem('refresh_token')
        AuthContext.user = null
        if (!cancelled) navigate('/login', { replace: true })
      }
    }

    validate()
    return () => { cancelled = true }
  }, [navigate])

  if (checking) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950">
        <Loader2 size={32} className="animate-spin text-cyan-400" />
      </div>
    )
  }

  return <>{children}</>
}

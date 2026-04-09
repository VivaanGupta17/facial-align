import { Navigate, useLocation } from 'react-router-dom'

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const token = localStorage.getItem('auth_token')

  if (!token) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}

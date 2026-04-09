import { Settings, LogOut } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function SettingsPage() {
  const navigate = useNavigate()

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    navigate('/login', { replace: true })
  }

  return (
    <div className="p-6 animate-fade-in" data-testid="settings-page">
      <div className="flex items-center gap-3 mb-6">
        <Settings size={20} className="text-cyan-400" />
        <h1 className="text-xl font-bold text-slate-100">Settings</h1>
      </div>

      <div className="space-y-4 max-w-2xl">
        {/* Account */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-4">Account</h2>
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Email</span>
              <span className="text-slate-200 font-mono">
                {(() => { try { return JSON.parse(localStorage.getItem('auth_user') ?? '{}').email } catch { return 'N/A' } })()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Auth Token</span>
              <span className="text-slate-500 font-mono text-xs">{localStorage.getItem('auth_token') ? 'Active' : 'None'}</span>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 mt-4 btn-secondary text-red-400 border-red-800 hover:bg-red-950"
            data-testid="logout-btn"
          >
            <LogOut size={14} />
            Sign Out
          </button>
        </div>

        {/* Platform */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-5">
          <h2 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-4">Platform</h2>
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Version</span>
              <span className="text-slate-200 font-mono">v0.9.0-beta</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">API Base URL</span>
              <span className="text-slate-200 font-mono text-xs">/api/v1</span>
            </div>
          </div>
        </div>

        {/* Placeholder sections */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-12 text-center">
          <Settings size={48} className="text-slate-600 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-slate-300 mb-2">Additional Settings</h2>
          <p className="text-sm text-slate-500 max-w-md mx-auto">
            Notification preferences, AI model configuration, team management, and system administration options.
          </p>
          <p className="text-xs text-slate-600 mt-4">Coming soon</p>
        </div>
      </div>
    </div>
  )
}

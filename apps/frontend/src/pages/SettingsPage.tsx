import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { User, Palette, Bell, Info, Keyboard, Monitor, Moon, LogOut, Settings } from 'lucide-react'

type Section = 'profile' | 'appearance' | 'notifications' | 'about' | 'shortcuts'

const SECTIONS: Array<{ id: Section; label: string; icon: React.ReactNode }> = [
  { id: 'profile', label: 'Profile', icon: <User size={16} /> },
  { id: 'appearance', label: 'Appearance', icon: <Palette size={16} /> },
  { id: 'notifications', label: 'Notifications', icon: <Bell size={16} /> },
  { id: 'about', label: 'About', icon: <Info size={16} /> },
  { id: 'shortcuts', label: 'Keyboard Shortcuts', icon: <Keyboard size={16} /> },
]

const SHORTCUTS = [
  { keys: ['Ctrl', 'K'], description: 'Command palette' },
  { keys: ['Ctrl', 'N'], description: 'New case' },
  { keys: ['Ctrl', 'U'], description: 'Upload DICOM' },
  { keys: ['Ctrl', '/'], description: 'Toggle sidebar' },
  { keys: ['Esc'], description: 'Close modal / deselect' },
  { keys: ['1'], description: 'Switch to Overview tab' },
  { keys: ['2'], description: 'Switch to Segmentation tab' },
  { keys: ['3'], description: 'Switch to Planning tab' },
  { keys: ['4'], description: 'Switch to Occlusion tab' },
  { keys: ['5'], description: 'Switch to Review tab' },
  { keys: ['R'], description: 'Reset 3D camera' },
  { keys: ['G'], description: 'Toggle grid' },
  { keys: ['M'], description: 'Toggle measurements' },
]

function Kbd({ children }: { children: string }) {
  return (
    <kbd className="px-1.5 py-0.5 bg-slate-700 border border-slate-600 rounded text-xs font-mono text-slate-300">
      {children}
    </kbd>
  )
}

function getUserEmail(): string {
  try {
    return JSON.parse(localStorage.getItem('auth_user') ?? '{}').email ?? 'N/A'
  } catch {
    return 'N/A'
  }
}

export default function SettingsPage() {
  const navigate = useNavigate()
  const [section, setSection] = useState<Section>('profile')

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    navigate('/login', { replace: true })
  }

  return (
    <div className="p-6 max-w-5xl mx-auto animate-fade-in" data-testid="settings-page">
      <div className="flex items-center gap-3 mb-6">
        <Settings size={20} className="text-cyan-400" />
        <h1 className="text-xl font-bold text-slate-100">Settings</h1>
      </div>

      <div className="flex gap-6">
        {/* Sidebar nav */}
        <nav className="w-48 shrink-0 space-y-1" data-testid="settings-nav">
          {SECTIONS.map(s => (
            <button
              key={s.id}
              onClick={() => setSection(s.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                section === s.id
                  ? 'bg-cyan-950 text-cyan-400 border border-cyan-900/60'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              data-testid={`settings-nav-${s.id}`}
            >
              {s.icon}
              {s.label}
            </button>
          ))}
        </nav>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {section === 'profile' && (
            <div className="space-y-4" data-testid="settings-profile">
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5">
                <h2 className="text-base font-semibold text-slate-100 border-b border-slate-700 pb-3">Profile Information</h2>
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-600 to-cyan-800 flex items-center justify-center text-xl font-bold text-white">
                    EC
                  </div>
                  <div>
                    <p className="text-base font-semibold text-slate-100">Dr. Emily Chen</p>
                    <p className="text-sm text-slate-400">Craniomaxillofacial Surgeon</p>
                    <p className="text-xs text-slate-500 font-mono mt-0.5">{getUserEmail()}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {[
                    { label: 'Department', value: 'Oral & Maxillofacial Surgery' },
                    { label: 'Institution', value: 'Metro General Hospital' },
                    { label: 'Role', value: 'Primary Surgeon' },
                    { label: 'License #', value: 'MED-2019-4281' },
                  ].map(r => (
                    <div key={r.label}>
                      <p className="text-xs text-slate-500 mb-0.5">{r.label}</p>
                      <p className="text-slate-200">{r.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Account actions */}
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-5">
                <h2 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-4">Account</h2>
                <div className="space-y-3 text-sm">
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
            </div>
          )}

          {section === 'appearance' && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5" data-testid="settings-appearance">
              <h2 className="text-base font-semibold text-slate-100 border-b border-slate-700 pb-3">Appearance</h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Moon size={18} className="text-slate-400" />
                    <div>
                      <p className="text-sm font-medium text-slate-200">Dark Theme</p>
                      <p className="text-xs text-slate-500">Currently the only supported theme</p>
                    </div>
                  </div>
                  <div className="px-3 py-1 rounded bg-cyan-950 text-cyan-400 text-xs font-medium border border-cyan-900">
                    Active
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Monitor size={18} className="text-slate-400" />
                    <div>
                      <p className="text-sm font-medium text-slate-200">Compact Mode</p>
                      <p className="text-xs text-slate-500">Reduce padding and font sizes</p>
                    </div>
                  </div>
                  <button
                    className="px-3 py-1 rounded bg-slate-700 text-slate-400 text-xs font-medium border border-slate-600 cursor-not-allowed opacity-60"
                    disabled
                  >
                    Coming Soon
                  </button>
                </div>
              </div>
            </div>
          )}

          {section === 'notifications' && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5" data-testid="settings-notifications">
              <h2 className="text-base font-semibold text-slate-100 border-b border-slate-700 pb-3">Notification Preferences</h2>
              <div className="space-y-4">
                {[
                  { label: 'Segmentation Complete', description: 'When AI segmentation finishes for your cases', checked: true },
                  { label: 'Plan Generated', description: 'When a new reduction plan is ready for review', checked: true },
                  { label: 'Review Requested', description: 'When a colleague requests your review', checked: true },
                  { label: 'Case Status Change', description: 'When case status transitions occur', checked: false },
                  { label: 'System Alerts', description: 'GPU errors, queue delays, system maintenance', checked: true },
                ].map(pref => (
                  <label key={pref.label} className="flex items-start gap-3 cursor-pointer" data-testid={`notif-${pref.label.toLowerCase().replace(/ /g, '-')}`}>
                    <input
                      type="checkbox"
                      defaultChecked={pref.checked}
                      className="mt-1 rounded border-slate-600 bg-slate-800 text-cyan-500"
                    />
                    <div>
                      <p className="text-sm font-medium text-slate-200">{pref.label}</p>
                      <p className="text-xs text-slate-500">{pref.description}</p>
                    </div>
                  </label>
                ))}
              </div>
              <p className="text-xs text-slate-600 italic">Notification delivery methods coming in a future update.</p>
            </div>
          )}

          {section === 'about' && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5" data-testid="settings-about">
              <h2 className="text-base font-semibold text-slate-100 border-b border-slate-700 pb-3">About Facial Align</h2>
              <div className="space-y-3 text-sm">
                {[
                  { label: 'Version', value: '0.9.0-beta' },
                  { label: 'Environment', value: import.meta.env.MODE ?? 'development' },
                  { label: 'API Base URL', value: import.meta.env.VITE_API_BASE_URL ?? '/api/v1' },
                  { label: 'Build', value: import.meta.env.VITE_BUILD_SHA?.slice(0, 8) ?? 'dev' },
                  { label: 'Framework', value: 'React 18 + Vite + TypeScript' },
                  { label: 'License', value: 'Proprietary — Metro General Hospital' },
                ].map(r => (
                  <div key={r.label} className="flex items-center justify-between">
                    <span className="text-slate-500">{r.label}</span>
                    <span className="text-slate-200 font-mono text-xs">{r.value}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-600 mt-4">
                AI-assisted surgical planning for craniofacial fracture reduction and orthognathic surgery.
              </p>
            </div>
          )}

          {section === 'shortcuts' && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5" data-testid="settings-shortcuts">
              <h2 className="text-base font-semibold text-slate-100 border-b border-slate-700 pb-3">Keyboard Shortcuts</h2>
              <div className="space-y-2">
                {SHORTCUTS.map(s => (
                  <div key={s.description} className="flex items-center justify-between py-1.5">
                    <span className="text-sm text-slate-300">{s.description}</span>
                    <div className="flex items-center gap-1">
                      {s.keys.map((k, i) => (
                        <span key={i}>
                          <Kbd>{k}</Kbd>
                          {i < s.keys.length - 1 && <span className="text-slate-600 mx-0.5">+</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

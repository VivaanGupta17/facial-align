import { useLocation, useNavigate, Link } from 'react-router-dom'
import { ChevronRight, Cpu, AlertCircle, CheckCircle, RefreshCw, Bell, User, Settings, LogOut } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dashboardApi } from '../../lib/api'
import { useCaseStore } from '../../stores/caseStore'

function GpuStatusPill({ utilizationPct }: { utilizationPct: number }) {
  const isHigh = utilizationPct > 85
  const isMed = utilizationPct > 60
  const color = isHigh ? 'text-red-400 bg-red-950 border-red-800' : isMed ? 'text-amber-400 bg-amber-950 border-amber-800' : 'text-emerald-400 bg-emerald-950 border-emerald-800'
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs font-mono font-medium ${color}`} data-testid="gpu-status-pill">
      <Cpu size={12} />
      <span>GPU {utilizationPct}%</span>
    </div>
  )
}

function Breadcrumbs() {
  const location = useLocation()
  const navigate = useNavigate()
  const { activeCase } = useCaseStore()

  const segments = location.pathname.split('/').filter(Boolean)

  const crumbLabel = (seg: string): string => {
    if (seg === 'dashboard') return 'Dashboard'
    if (seg === 'cases') return 'Cases'
    if (seg === 'upload') return 'Upload DICOM'
    if (seg === 'studies') return 'Studies'
    if (seg === 'settings') return 'Settings'
    if (seg === 'compute') return 'Compute'
    if (seg === 'models') return '3D Models'
    if (activeCase && seg === activeCase.id) return activeCase.caseNumber
    return seg
  }

  return (
    <nav className="flex items-center gap-1 text-sm" aria-label="Breadcrumb" data-testid="breadcrumbs">
      <button
        onClick={() => navigate('/dashboard')}
        className="text-slate-500 hover:text-slate-300 transition-colors"
      >
        Facial Align
      </button>
      {segments.map((seg, i) => (
        <span key={seg} className="flex items-center gap-1">
          <ChevronRight size={14} className="text-slate-600" />
          <button
            onClick={() => navigate('/' + segments.slice(0, i + 1).join('/'))}
            className={i === segments.length - 1 ? 'text-slate-100 font-medium' : 'text-slate-500 hover:text-slate-300 transition-colors'}
          >
            {crumbLabel(seg)}
          </button>
        </span>
      ))}
    </nav>
  )
}

function CaseStatusBadge() {
  const { activeCase } = useCaseStore()
  if (!activeCase) return null

  const statusConfig: Record<string, { label: string; color: string }> = {
    planning: { label: 'Planning', color: 'bg-cyan-950 text-cyan-400 border-cyan-800' },
    review: { label: 'In Review', color: 'bg-amber-950 text-amber-400 border-amber-800' },
    approved: { label: 'Approved', color: 'bg-emerald-950 text-emerald-400 border-emerald-800' },
    segmentation_in_progress: { label: 'Segmenting', color: 'bg-blue-950 text-blue-400 border-blue-800' },
    segmentation_review: { label: 'Seg. Review', color: 'bg-purple-950 text-purple-400 border-purple-800' },
  }

  const config = statusConfig[activeCase.status] ?? { label: activeCase.status, color: 'bg-slate-800 text-slate-400 border-slate-700' }

  return (
    <div className={`px-2.5 py-1 rounded border text-xs font-semibold ${config.color}`} data-testid="case-status-badge">
      {config.label}
    </div>
  )
}

export default function TopBar() {
  const navigate = useNavigate()
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const { data: health } = useQuery({
    queryKey: ['system-health'],
    queryFn: dashboardApi.getSystemHealth,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })

  const maxGpuUtil = health?.gpus.reduce((m, g) => Math.max(m, g.utilizationPercent), 0) ?? 0
  const queueDepth = health?.queue.depth ?? 0

  // Close user menu on click outside
  useEffect(() => {
    if (!userMenuOpen) return
    const handler = () => setUserMenuOpen(false)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [userMenuOpen])

  return (
    <header
      className="h-[52px] flex items-center justify-between px-4 bg-slate-900 border-b border-slate-800 shrink-0"
      data-testid="topbar"
    >
      {/* Left: breadcrumbs + status */}
      <div className="flex items-center gap-3">
        <Breadcrumbs />
        <CaseStatusBadge />
      </div>

      {/* Right: system info + user */}
      <div className="flex items-center gap-2">
        {/* Queue indicator */}
        {queueDepth > 0 && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs font-mono text-blue-400 bg-blue-950 border-blue-800" data-testid="queue-indicator">
            <RefreshCw size={12} className="animate-spin" />
            <span>{queueDepth} queued</span>
          </div>
        )}

        {/* GPU status */}
        {health && <GpuStatusPill utilizationPct={maxGpuUtil} />}

        {/* API health */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs font-mono text-emerald-400 bg-emerald-950 border-emerald-800" data-testid="api-health">
          <CheckCircle size={12} />
          <span>API {health?.apiLatencyMs ?? '--'}ms</span>
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-slate-700" />

        {/* Notifications */}
        <button className="relative p-2 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors" data-testid="notifications-btn" aria-label="Notifications">
          <Bell size={16} />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-cyan-400 rounded-full" />
        </button>

        {/* User menu */}
        <div className="relative">
          <button
            onClick={(e) => { e.stopPropagation(); setUserMenuOpen(o => !o) }}
            className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-md hover:bg-slate-800 transition-colors"
            data-testid="user-menu-btn"
          >
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-600 to-cyan-800 flex items-center justify-center text-xs font-bold text-white shrink-0">
              EC
            </div>
            <div className="text-left hidden sm:block">
              <p className="text-xs font-semibold text-slate-200">Dr. Emily Chen</p>
              <p className="text-2xs text-slate-500">Oral & Maxillofacial</p>
            </div>
          </button>

          {userMenuOpen && (
            <div
              className="absolute right-0 top-full mt-1 w-52 bg-slate-800 border border-slate-700 rounded-lg shadow-panel py-1 z-50 animate-fade-in"
              data-testid="user-menu"
            >
              <div className="px-4 py-2.5 border-b border-slate-700">
                <p className="text-sm font-semibold text-slate-100">Dr. Emily Chen</p>
                <p className="text-xs text-slate-400">emily.chen@hospital.org</p>
              </div>
              <button className="flex items-center gap-2.5 w-full px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 transition-colors">
                <User size={14} /> Profile
              </button>
              <button className="flex items-center gap-2.5 w-full px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 transition-colors">
                <Settings size={14} /> Preferences
              </button>
              <div className="border-t border-slate-700 mt-1 pt-1">
                <button
                  onClick={() => { localStorage.removeItem('auth_token'); localStorage.removeItem('auth_user'); navigate('/login') }}
                  className="flex items-center gap-2.5 w-full px-4 py-2 text-sm text-red-400 hover:bg-slate-700 transition-colors"
                >
                  <LogOut size={14} /> Sign out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

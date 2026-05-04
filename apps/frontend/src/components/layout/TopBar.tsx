import { useLocation, useNavigate, Link } from 'react-router-dom'
import { ChevronRight, Cpu, CheckCircle, RefreshCw, Bell, User, Settings, LogOut, FlaskConical, ShieldCheck } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dashboardApi, authApi } from '../../lib/api'
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

function CapabilitySummaryPill({
  available,
  total,
}: {
  available: number
  total: number
}) {
  const healthy = total > 0 && available === total
  const mixed = available > 0 && available < total

  return (
    <div
      className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
        healthy
          ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
          : mixed
          ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
          : 'border-slate-600/40 bg-slate-800/60 text-slate-400'
      }`}
      data-testid="capability-summary-pill"
    >
      {healthy ? <ShieldCheck size={12} /> : <FlaskConical size={12} />}
      <span>{available}/{total} core capabilities ready</span>
    </div>
  )
}

export default function TopBar() {
  const navigate = useNavigate()
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const { data: currentUser } = useQuery({
    queryKey: ['current-user'],
    queryFn: authApi.me,
    staleTime: 60_000,
    retry: false,
  })
  const { data: health } = useQuery({
    queryKey: ['system-health'],
    queryFn: dashboardApi.getSystemHealth,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })

  const maxGpuUtil = health?.gpus.reduce((m, g) => Math.max(m, g.utilizationPercent), 0) ?? 0
  const queueDepth = health?.queue.depth ?? 0
  const coreCapabilities = health?.capabilities.filter((capability) =>
    ['segmentation', 'planning', 'landmarks', 'occlusion'].includes(capability.category)
  ) ?? []
  const availableCapabilities = coreCapabilities.filter((capability) => capability.status === 'available').length

  // Close user menu on click outside
  useEffect(() => {
    if (!userMenuOpen) return
    const handler = () => setUserMenuOpen(false)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [userMenuOpen])

  return (
    <header
      className="flex h-[72px] shrink-0 items-center justify-between border-b border-white/10 bg-[rgba(8,14,26,0.72)] px-5 backdrop-blur-xl"
      data-testid="topbar"
    >
      {/* Left: breadcrumbs + status */}
      <div className="flex items-center gap-3">
        <Breadcrumbs />
        <CaseStatusBadge />
      </div>

      {/* Right: system info + user */}
      <div className="flex items-center gap-2">
        {coreCapabilities.length > 0 && (
          <div className="hidden xl:block">
            <CapabilitySummaryPill available={availableCapabilities} total={coreCapabilities.length} />
          </div>
        )}

        {/* Queue indicator */}
        {queueDepth > 0 && (
          <div className="flex items-center gap-1.5 rounded-full border border-blue-500/20 bg-blue-500/10 px-2.5 py-1 text-xs font-mono text-blue-300" data-testid="queue-indicator">
            <RefreshCw size={12} className="animate-spin" />
            <span>{queueDepth} queued</span>
          </div>
        )}

        {/* GPU status */}
        {health && <GpuStatusPill utilizationPct={maxGpuUtil} />}

        {/* API health */}
        <div className="flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-mono text-emerald-300" data-testid="api-health">
          <CheckCircle size={12} />
          <span>API {health?.apiLatencyMs ?? '--'}ms</span>
        </div>

        {/* Divider */}
        <div className="h-5 w-px bg-white/10" />

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
              {(currentUser?.full_name ?? 'U').split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase()}
            </div>
            <div className="text-left hidden sm:block">
              <p className="text-xs font-semibold text-slate-200">{currentUser?.full_name ?? 'User'}</p>
              <p className="text-2xs text-slate-500">{currentUser?.specialty ?? currentUser?.role ?? ''}</p>
            </div>
          </button>

          {userMenuOpen && (
            <div
              className="absolute right-0 top-full z-50 mt-2 w-52 rounded-2xl border border-white/10 bg-[rgba(19,32,51,0.96)] py-1 shadow-2xl animate-fade-in"
              data-testid="user-menu"
            >
              <div className="border-b border-white/10 px-4 py-2.5">
                <p className="text-sm font-semibold text-slate-100">{currentUser?.full_name ?? 'User'}</p>
                <p className="text-xs text-slate-400">{currentUser?.email ?? ''}</p>
              </div>
              <button className="flex w-full items-center gap-2.5 px-4 py-2 text-sm text-slate-300 transition-colors hover:bg-white/5">
                <User size={14} /> Profile
              </button>
              <button className="flex w-full items-center gap-2.5 px-4 py-2 text-sm text-slate-300 transition-colors hover:bg-white/5">
                <Settings size={14} /> Preferences
              </button>
              <div className="mt-1 border-t border-white/10 pt-1">
                <button
                  onClick={() => {
                    localStorage.removeItem('auth_token')
                    localStorage.removeItem('refresh_token')
                    navigate('/login', { replace: true })
                  }}
                  className="flex w-full items-center gap-2.5 px-4 py-2 text-sm text-red-400 transition-colors hover:bg-white/5"
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

import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FolderOpen,
  Scan,
  BoxSelect,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  Upload,
  Cpu,
} from 'lucide-react'
import { useCaseStore } from '../../stores/caseStore'

interface NavItem {
  label: string
  to: string
  icon: React.ReactNode
  badge?: string | number
}

const navSections: Array<{ title: string; items: NavItem[] }> = [
  {
    title: 'Workflow',
    items: [
      { label: 'Dashboard', to: '/dashboard', icon: <LayoutDashboard size={16} /> },
      { label: 'Cases', to: '/cases', icon: <FolderOpen size={16} /> },
      { label: 'Upload DICOM', to: '/upload', icon: <Upload size={16} /> },
    ],
  },
  {
    title: 'Planning',
    items: [
      { label: 'Studies', to: '/studies', icon: <Scan size={16} /> },
      { label: '3D Models', to: '/models', icon: <BoxSelect size={16} /> },
    ],
  },
  {
    title: 'System',
    items: [
      { label: 'Compute', to: '/compute', icon: <Cpu size={16} /> },
      { label: 'Settings', to: '/settings', icon: <Settings size={16} /> },
    ],
  },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(() => window.innerWidth < 1024)
  const { activeCase } = useCaseStore()

  // Auto-collapse on narrow viewports
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1023px)')
    const handler = (e: MediaQueryListEvent) => setCollapsed(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  return (
    <aside
      className={`flex shrink-0 flex-col border-r border-white/10 bg-[rgba(8,14,26,0.86)] backdrop-blur-xl transition-all duration-300 ${collapsed ? 'w-16' : 'w-[260px]'}`}
      data-testid="sidebar"
    >
      {/* Logo */}
      <div className="flex h-[72px] items-center gap-3 border-b border-white/10 px-4">
        {/* SVG Logo — stylized facial bone outline */}
        <svg viewBox="0 0 28 28" fill="none" className="shrink-0 w-7 h-7" aria-label="Facial Align">
          <rect width="28" height="28" rx="6" fill="#0f172a" />
          <path
            d="M14 3 C8.5 3, 5 8, 5 13 C5 19.5, 9 24.5, 14 26.5 C19 24.5, 23 19.5, 23 13 C23 8, 19.5 3, 14 3Z"
            stroke="#06b6d4" strokeWidth="1.4" fill="none"
          />
          <path d="M10.5 12.5 Q14 10, 17.5 12.5" stroke="#22d3ee" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          <path d="M9.5 17 Q11.5 20.5, 14 21.5 Q16.5 20.5, 18.5 17" stroke="#22d3ee" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          <circle cx="11.5" cy="11.5" r="0.9" fill="#06b6d4" />
          <circle cx="16.5" cy="11.5" r="0.9" fill="#06b6d4" />
        </svg>
        {!collapsed && (
          <div className="min-w-0">
            <span className="text-sm font-semibold tracking-tight text-slate-100">Facial Align</span>
            <p className="text-[11px] text-slate-500">Fracture planning console</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-5 overflow-y-auto px-2 py-4" data-testid="sidebar-nav">
        {navSections.map(section => (
          <div key={section.title}>
            {!collapsed && (
              <p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">
                {section.title}
              </p>
            )}
            <ul className="space-y-0.5">
              {section.items.map(item => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    className={({ isActive }) =>
                      isActive
                        ? `flex items-center gap-2.5 rounded-xl border border-cyan-400/20 bg-[rgba(12,74,110,0.26)] px-3 py-2.5 text-sm font-medium text-cyan-300 shadow-[0_10px_30px_rgba(8,145,178,0.12)] ${collapsed ? 'justify-center' : ''}`
                        : `flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-400 transition-colors hover:bg-white/5 hover:text-slate-100 ${collapsed ? 'justify-center' : ''}`
                    }
                    data-testid={`nav-${item.label.toLowerCase().replace(' ', '-')}`}
                    title={collapsed ? item.label : undefined}
                  >
                    {item.icon}
                    {!collapsed && (
                      <>
                        <span className="flex-1">{item.label}</span>
                        {item.badge != null && (
                          <span className="bg-cyan-900 text-cyan-300 text-2xs font-mono font-semibold px-1.5 py-0.5 rounded">
                            {item.badge}
                          </span>
                        )}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Active case indicator */}
      {activeCase && !collapsed && (
        <div className="mx-2 mb-2 rounded-2xl border border-cyan-400/15 bg-[rgba(10,18,30,0.85)] p-3" data-testid="active-case-indicator">
          <div className="flex items-center gap-2 mb-1">
            <Activity size={12} className="text-cyan-400 animate-pulse" />
            <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-400">Active Case</span>
          </div>
          <p className="text-sm font-mono font-semibold text-slate-100">{activeCase.caseNumber}</p>
          <p className="text-xs text-slate-400 capitalize mt-0.5">{activeCase.caseType.replace(/_/g, ' ')}</p>
        </div>
      )}

      {!collapsed && (
        <div className="mx-2 mb-2 rounded-2xl border border-white/10 bg-[rgba(11,19,33,0.72)] px-3 py-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">Mode</p>
          <p className="mt-1 text-sm font-medium text-slate-200">Baseline-first workflow</p>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Upload, review, plan, and approve with explicit provenance on each step.
          </p>
        </div>
      )}

      {/* Collapse toggle */}
      <div className="border-t border-white/10 p-2">
        <button
          onClick={() => setCollapsed(c => !c)}
          className="flex w-full items-center justify-center gap-2 rounded-xl py-2 text-sm text-slate-500 transition-colors hover:bg-white/5 hover:text-slate-300"
          data-testid="sidebar-collapse-toggle"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span className="text-xs">Collapse</span></>}
        </button>
      </div>
    </aside>
  )
}

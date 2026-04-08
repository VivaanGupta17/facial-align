import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
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
    title: 'Main',
    items: [
      { label: 'Dashboard', to: '/dashboard', icon: <LayoutDashboard size={16} /> },
      { label: 'Cases', to: '/cases', icon: <FolderOpen size={16} />, badge: 4 },
      { label: 'Upload DICOM', to: '/upload', icon: <Upload size={16} /> },
    ],
  },
  {
    title: 'Tools',
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
  const [collapsed, setCollapsed] = useState(false)
  const { activeCase } = useCaseStore()
  const location = useLocation()

  return (
    <aside
      className={`flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-300 ${collapsed ? 'w-14' : 'w-[240px]'} shrink-0`}
      data-testid="sidebar"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-[52px] border-b border-slate-800">
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
            <span className="font-semibold text-sm text-slate-100 tracking-tight">Facial Align</span>
            <p className="text-2xs text-slate-500 font-mono">v0.9.0-beta</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-4" data-testid="sidebar-nav">
        {navSections.map(section => (
          <div key={section.title}>
            {!collapsed && (
              <p className="px-3 mb-1.5 text-2xs font-semibold uppercase tracking-wider text-slate-600">
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
                        ? `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium text-cyan-400 bg-cyan-950 border border-cyan-900/60 ${collapsed ? 'justify-center' : ''}`
                        : `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors ${collapsed ? 'justify-center' : ''}`
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
        <div className="mx-2 mb-2 p-3 rounded-md bg-slate-800 border border-slate-700" data-testid="active-case-indicator">
          <div className="flex items-center gap-2 mb-1">
            <Activity size={12} className="text-cyan-400 animate-pulse" />
            <span className="text-2xs font-semibold uppercase tracking-wider text-cyan-500">Active Case</span>
          </div>
          <p className="text-sm font-mono font-semibold text-slate-100">{activeCase.caseNumber}</p>
          <p className="text-xs text-slate-400 capitalize mt-0.5">{activeCase.type.replace(/_/g, ' ')}</p>
        </div>
      )}

      {/* Collapse toggle */}
      <div className="border-t border-slate-800 p-2">
        <button
          onClick={() => setCollapsed(c => !c)}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-md text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors text-sm"
          data-testid="sidebar-collapse-toggle"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span className="text-xs">Collapse</span></>}
        </button>
      </div>
    </aside>
  )
}

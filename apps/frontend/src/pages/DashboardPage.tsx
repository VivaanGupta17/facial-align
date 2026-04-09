import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  FolderOpen, Upload, BoxSelect, Plus, TrendingUp, TrendingDown,
  Clock, CheckCircle, Cpu, HardDrive, Activity, Layers, RefreshCw,
  X, AlertCircle, Sparkles,
} from 'lucide-react'
import { dashboardApi, casesApi } from '../lib/api'
import StatusBadge from '../components/common/StatusBadge'
import { Skeleton } from '../components/common/LoadingOverlay'
import type { CaseListItem, GpuStatus } from '../types/medical'

// ---------------------------
// Format helpers
// ---------------------------
function fmtTime(iso: string) {
  const d = new Date(iso)
  const now = Date.now()
  const diff = now - d.getTime()
  if (diff < 60_000) return 'Just now'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function fmtCaseType(type: string) {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function truncateId(id: string) {
  if (!id) return '—'
  return id.length > 8 ? `${id.slice(0, 8)}...` : id
}

// ---------------------------
// Dismissable Error Banner
// ---------------------------
function ErrorBanner({ message, onDismiss, onRetry }: { message: string; onDismiss: () => void; onRetry?: () => void }) {
  return (
    <div className="flex items-center gap-2 p-3 bg-red-950/60 border border-red-800 rounded-lg text-sm animate-fade-in" data-testid="error-banner">
      <AlertCircle size={15} className="text-red-400 shrink-0" />
      <span className="flex-1 text-red-300">{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition-colors shrink-0">
          <RefreshCw size={12} /> Retry
        </button>
      )}
      <button onClick={onDismiss} className="text-red-400/60 hover:text-red-300 transition-colors shrink-0" data-testid="error-banner-dismiss">
        <X size={14} />
      </button>
    </div>
  )
}

// ---------------------------
// Welcome Card (zero cases)
// ---------------------------
function WelcomeCard() {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center" data-testid="welcome-card">
      <div className="w-16 h-16 rounded-full bg-cyan-950 border border-cyan-800 flex items-center justify-center mb-4">
        <Sparkles size={28} className="text-cyan-400" />
      </div>
      <h3 className="text-lg font-semibold text-slate-100 mb-2">Welcome to Facial Align</h3>
      <p className="text-sm text-slate-400 max-w-md mb-6">
        Get started by uploading a DICOM study. The AI pipeline will automatically segment anatomical structures and generate a surgical reduction plan.
      </p>
      <div className="flex gap-3">
        <button onClick={() => navigate('/upload')} className="flex items-center gap-2 btn-primary" data-testid="welcome-upload-btn">
          <Upload size={15} /> Upload DICOM
        </button>
        <button onClick={() => navigate('/cases')} className="flex items-center gap-2 btn-secondary">
          <FolderOpen size={15} /> Browse Cases
        </button>
      </div>
    </div>
  )
}

// ---------------------------
// GPU Card
// ---------------------------
function GpuCard({ gpu }: { gpu: GpuStatus }) {
  const util = gpu.utilizationPercent
  const memPct = Math.round((gpu.memoryUsedGb / gpu.memoryTotalGb) * 100)
  const barColor = util > 85 ? 'bg-red-500' : util > 60 ? 'bg-amber-500' : 'bg-cyan-500'
  const statusDot = gpu.status === 'busy' ? 'bg-cyan-400 animate-pulse' : gpu.status === 'idle' ? 'bg-emerald-400' : gpu.status === 'error' ? 'bg-red-400' : 'bg-slate-600'

  return (
    <div className="p-3 rounded-md bg-slate-900 border border-slate-700" data-testid="gpu-card">
      <div className="flex items-center gap-2 mb-2">
        <span className={`w-2 h-2 rounded-full shrink-0 ${statusDot}`} />
        <span className="text-xs font-mono text-slate-300 truncate">{gpu.name.replace('NVIDIA ', '')}</span>
        <span className="ml-auto text-xs font-mono text-slate-500">{gpu.temperatureCelsius}°C</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-2xs text-slate-500">GPU</span>
            <span className="text-2xs font-mono text-slate-300">{util}%</span>
          </div>
          <div className="h-1 bg-slate-700 rounded-full">
            <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${util}%` }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-2xs text-slate-500">VRAM</span>
            <span className="text-2xs font-mono text-slate-300">{memPct}%</span>
          </div>
          <div className="h-1 bg-slate-700 rounded-full">
            <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${memPct}%` }} />
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------
// Recent Cases Table (using CaseListItem)
// ---------------------------
function RecentCasesTable({ cases }: { cases: CaseListItem[] }) {
  const navigate = useNavigate()

  return (
    <div className="overflow-x-auto" data-testid="recent-cases-table">
      <table className="data-table w-full">
        <thead>
          <tr>
            <th>Case #</th>
            <th>Patient ID</th>
            <th>Type</th>
            <th>Status</th>
            <th>Surgeon</th>
            <th>Updated</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {cases.map(c => (
            <tr
              key={c.id}
              className="cursor-pointer"
              onClick={() => navigate(`/cases/${c.id}`)}
              data-testid={`case-row-${c.id}`}
            >
              <td>
                <span className="font-mono text-sm text-slate-100 font-semibold">{c.caseNumber}</span>
              </td>
              <td>
                <span className="font-mono text-xs text-slate-400">{truncateId(c.patientId)}</span>
              </td>
              <td>
                <span className="text-xs text-slate-300">{fmtCaseType(c.caseType)}</span>
              </td>
              <td>
                <StatusBadge status={c.status} size="sm" />
              </td>
              <td>
                <span className="text-xs text-slate-400">{c.surgeonId ? truncateId(c.surgeonId) : 'Unassigned'}</span>
              </td>
              <td>
                <span className="text-xs font-mono text-slate-500">{fmtTime(c.updatedAt)}</span>
              </td>
              <td>
                <button
                  onClick={(e) => { e.stopPropagation(); navigate(`/cases/${c.id}`) }}
                  className="text-xs text-cyan-400 hover:text-cyan-300 font-medium transition-colors"
                >
                  Open →
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------
// Panel Refresh Button
// ---------------------------
function RefreshButton({ onClick, className = '' }: { onClick: () => void; className?: string }) {
  return (
    <button
      onClick={onClick}
      className={`p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-slate-300 transition-colors ${className}`}
      title="Refresh"
      data-testid="refresh-btn"
    >
      <RefreshCw size={13} />
    </button>
  )
}

// ---------------------------
// Main Dashboard
// ---------------------------
export default function DashboardPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [statsErrorDismissed, setStatsErrorDismissed] = useState(false)
  const [healthErrorDismissed, setHealthErrorDismissed] = useState(false)

  const { data: stats, isLoading: statsLoading, error: statsError, refetch: refetchStats } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: dashboardApi.getStats,
    refetchInterval: 30_000,
  })

  const { data: health, isLoading: healthLoading, error: healthError, refetch: refetchHealth } = useQuery({
    queryKey: ['system-health'],
    queryFn: dashboardApi.getSystemHealth,
    refetchInterval: 15_000,
  })

  const { data: recentCasesData, isLoading: casesLoading, refetch: refetchCases } = useQuery({
    queryKey: ['recent-cases'],
    queryFn: () => casesApi.getRecent(10),
    refetchInterval: 30_000,
  })

  const zeroCases = !casesLoading && recentCasesData && recentCasesData.length === 0

  return (
    <div className="p-6 space-y-6 max-w-screen-2xl mx-auto animate-fade-in" data-testid="dashboard-page">

      {/* Dismissable error banners */}
      {statsError && !statsErrorDismissed && (
        <ErrorBanner
          message="Failed to load dashboard statistics."
          onDismiss={() => setStatsErrorDismissed(true)}
          onRetry={() => { setStatsErrorDismissed(false); refetchStats() }}
        />
      )}
      {healthError && !healthErrorDismissed && (
        <ErrorBanner
          message="System health data unavailable."
          onDismiss={() => setHealthErrorDismissed(true)}
          onRetry={() => { setHealthErrorDismissed(false); refetchHealth() }}
        />
      )}

      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Surgical Command Center</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => navigate('/upload')}
            className="flex items-center gap-2 btn-secondary"
            data-testid="upload-dicom-btn"
          >
            <Upload size={15} />
            Upload DICOM
          </button>
          <button
            onClick={() => navigate('/cases')}
            className="flex items-center gap-2 btn-primary"
            data-testid="new-case-btn"
          >
            <Plus size={15} />
            New Case
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4" data-testid="stats-row">
        {statsLoading ? (
          Array(4).fill(0).map((_, i) => (
            <div key={i} className="bg-slate-800 border border-slate-700 rounded-lg p-4">
              <Skeleton className="h-3 w-24 mb-3" />
              <Skeleton className="h-8 w-16 mb-2" />
              <Skeleton className="h-3 w-20" />
            </div>
          ))
        ) : stats ? (
          <>
            <StatCard
              label="Active Cases"
              value={stats.activeCases}
              icon={<FolderOpen size={18} />}
              delta={stats.activeCasesDelta}
              color="cyan"
              onClick={() => navigate('/cases?status=active')}
            />
            <StatCard
              label="Pending Segmentation"
              value={stats.pendingSegmentation}
              icon={<Layers size={18} />}
              delta={stats.pendingSegmentationDelta}
              color="amber"
              onClick={() => navigate('/cases?status=segmentation')}
            />
            <StatCard
              label="Awaiting Review"
              value={stats.awaitingReview}
              icon={<Clock size={18} />}
              delta={stats.awaitingReviewDelta}
              color="purple"
              onClick={() => navigate('/cases?status=review')}
            />
            <StatCard
              label="Completed This Month"
              value={stats.completedThisMonth}
              icon={<CheckCircle size={18} />}
              delta={stats.completedThisMonthDelta}
              color="green"
            />
          </>
        ) : null}
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

        {/* Recent cases (2/3 width) */}
        <div className="xl:col-span-2 bg-slate-800 border border-slate-700 rounded-lg" data-testid="recent-cases-panel">
          <div className="panel-header">
            <div className="flex items-center gap-2">
              <FolderOpen size={15} className="text-cyan-400" />
              <h2 className="panel-title">Recent Cases</h2>
              {recentCasesData && (
                <span className="text-2xs font-mono text-slate-500 bg-slate-700 px-1.5 py-0.5 rounded">
                  {recentCasesData.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <RefreshButton onClick={() => refetchCases()} />
              <button
                onClick={() => navigate('/cases')}
                className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                View all →
              </button>
            </div>
          </div>

          {casesLoading ? (
            <div className="p-4">
              <table className="w-full">
                <tbody><TableSkeleton /></tbody>
              </table>
            </div>
          ) : zeroCases ? (
            <WelcomeCard />
          ) : recentCasesData && recentCasesData.length > 0 ? (
            <RecentCasesTable cases={recentCasesData} />
          ) : null}
        </div>

        {/* Right column */}
        <div className="space-y-4">

          {/* Quick actions */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg" data-testid="quick-actions-panel">
            <div className="panel-header">
              <h2 className="panel-title">Quick Actions</h2>
            </div>
            <div className="p-3 grid grid-cols-1 gap-2">
              {[
                { icon: <Plus size={15} />, label: 'New Case', sub: 'Create from existing study', to: '/cases' },
                { icon: <Upload size={15} />, label: 'Upload DICOM', sub: 'Import CT/CBCT series', to: '/upload' },
                { icon: <BoxSelect size={15} />, label: 'View 3D Models', sub: 'Browse segmented models', to: '/models' },
              ].map(a => (
                <button
                  key={a.label}
                  onClick={() => navigate(a.to)}
                  className="flex items-center gap-3 p-3 rounded-md bg-slate-900 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 transition-all text-left"
                  data-testid={`quick-action-${a.label.toLowerCase().replace(' ', '-')}`}
                >
                  <span className="p-2 rounded bg-slate-800 text-cyan-400">{a.icon}</span>
                  <div>
                    <p className="text-sm font-medium text-slate-200">{a.label}</p>
                    <p className="text-xs text-slate-500">{a.sub}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* System health */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg" data-testid="system-health-panel">
            <div className="panel-header">
              <div className="flex items-center gap-2">
                <Activity size={15} className="text-emerald-400" />
                <h2 className="panel-title">System Health</h2>
              </div>
              <div className="flex items-center gap-2">
                <RefreshButton onClick={() => refetchHealth()} />
                <span className="text-2xs font-mono text-slate-500">Live</span>
              </div>
            </div>
            <div className="p-3 space-y-3">
              {healthLoading ? (
                <div className="space-y-2">
                  {Array(3).fill(0).map((_, i) => <Skeleton key={i} className="h-14" />)}
                </div>
              ) : healthError ? (
                <div className="flex flex-col items-center justify-center py-6 text-center" data-testid="health-unavailable">
                  <Activity size={24} className="text-slate-600 mb-2" />
                  <p className="text-sm text-slate-500">System health unavailable</p>
                  <button onClick={() => refetchHealth()} className="mt-2 text-xs text-cyan-400 hover:text-cyan-300">
                    Retry
                  </button>
                </div>
              ) : health ? (
                <>
                  {/* GPU grid */}
                  <div>
                    <p className="label-xs mb-2">GPU Compute</p>
                    <div className="grid grid-cols-2 gap-2">
                      {health.gpus.map(g => <GpuCard key={g.id} gpu={g} />)}
                    </div>
                  </div>

                  {/* Queue */}
                  <div className="flex items-center justify-between p-2.5 rounded bg-slate-900 border border-slate-700">
                    <div className="flex items-center gap-2">
                      <Clock size={13} className="text-slate-400" />
                      <span className="text-xs text-slate-300">Inference Queue</span>
                    </div>
                    <div className="text-right">
                      <span className="font-mono text-sm font-bold text-slate-100">{health.queue.depth}</span>
                      <span className="text-2xs text-slate-500 ml-1">jobs</span>
                      {health.queue.estimatedWaitMinutes > 0 && (
                        <p className="text-2xs text-slate-500">~{health.queue.estimatedWaitMinutes}m wait</p>
                      )}
                    </div>
                  </div>

                  {/* Storage */}
                  <div className="p-2.5 rounded bg-slate-900 border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <HardDrive size={13} className="text-slate-400" />
                        <span className="text-xs text-slate-300">DICOM Storage</span>
                      </div>
                      <span className="text-xs font-mono text-slate-400">
                        {(health.storageUsedGb / 1024).toFixed(1)} / {(health.storageTotalGb / 1024).toFixed(0)} TB
                      </span>
                    </div>
                    <div className="h-1.5 bg-slate-700 rounded-full">
                      <div
                        className="h-full rounded-full bg-cyan-500"
                        style={{ width: `${(health.storageUsedGb / health.storageTotalGb) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Models */}
                  <div>
                    <p className="label-xs mb-2">AI Models</p>
                    <div className="space-y-1.5">
                      {health.models.map(m => (
                        <div key={m.name} className="flex items-center justify-between text-xs" data-testid={`model-row-${m.name}`}>
                          <div className="flex items-center gap-2">
                            <Cpu size={11} className="text-slate-500" />
                            <span className="text-slate-300">{m.name}</span>
                            <span className="font-mono text-slate-500">v{m.version}</span>
                          </div>
                          <span className="font-mono text-emerald-400">{Math.round(m.accuracy * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------
// Stat Card (internal)
// ---------------------------
type StatColor = 'cyan' | 'amber' | 'purple' | 'green'

function StatCard({
  label, value, icon, delta, color, onClick,
}: {
  label: string; value: number; icon: React.ReactNode; delta: number; color: StatColor; onClick?: () => void
}) {
  const colorMap: Record<StatColor, { bg: string; text: string; icon: string }> = {
    cyan: { bg: 'bg-cyan-950/50 border-cyan-900/50', text: 'text-cyan-300', icon: 'text-cyan-400' },
    amber: { bg: 'bg-amber-950/50 border-amber-900/50', text: 'text-amber-300', icon: 'text-amber-400' },
    purple: { bg: 'bg-purple-950/50 border-purple-900/50', text: 'text-purple-300', icon: 'text-purple-400' },
    green: { bg: 'bg-emerald-950/50 border-emerald-900/50', text: 'text-emerald-300', icon: 'text-emerald-400' },
  }
  const c = colorMap[color]
  const isUp = delta > 0
  const isDown = delta < 0

  return (
    <div
      className={`border rounded-lg p-4 ${c.bg} ${onClick ? 'cursor-pointer hover:brightness-110 transition-all' : ''}`}
      onClick={onClick}
      data-testid={`stat-card-${label.toLowerCase().replace(' ', '-')}`}
    >
      <div className="flex items-start justify-between mb-3">
        <span className={`text-xs font-semibold uppercase tracking-wider text-slate-400`}>{label}</span>
        <span className={c.icon}>{icon}</span>
      </div>
      <div className="flex items-end gap-2">
        <span className={`font-mono font-bold text-3xl ${c.text} leading-none`}>{value}</span>
        {delta !== 0 && (
          <span className={`flex items-center gap-0.5 text-xs pb-0.5 ${isUp ? 'text-emerald-400' : isDown ? 'text-red-400' : 'text-slate-500'}`}>
            {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {Math.abs(delta)}
          </span>
        )}
      </div>
      <p className="text-2xs text-slate-500 mt-1">vs. last week</p>
    </div>
  )
}

function TableSkeleton() {
  return (
    <>
      {Array(6).fill(0).map((_, i) => (
        <tr key={i} className="border-b border-slate-800">
          {[28, 20, 32, 24, 20, 12, 8].map((w, j) => (
            <td key={j} className="px-4 py-3">
              <div className={`h-4 bg-slate-700/60 rounded animate-pulse`} style={{ width: `${w * 4}px` }} />
            </td>
          ))}
        </tr>
      ))}
    </>
  )
}

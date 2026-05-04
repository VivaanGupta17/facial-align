import { useQuery } from '@tanstack/react-query'
import { Cpu, Clock, Activity, RefreshCw, HardDrive, Thermometer, Zap, AlertTriangle } from 'lucide-react'
import { dashboardApi } from '../lib/api'
import { Skeleton } from '../components/common/LoadingOverlay'
import PageHeader from '../components/common/PageHeader'
import { CapabilityBadge } from '../components/common/TrustIndicators'
import type { GpuStatus } from '../types/medical'

function GpuDetailCard({ gpu }: { gpu: GpuStatus }) {
  const util = gpu.utilizationPercent
  const memPct = Math.round((gpu.memoryUsedGb / gpu.memoryTotalGb) * 100)
  const barColor = util > 85 ? 'bg-red-500' : util > 60 ? 'bg-amber-500' : 'bg-cyan-500'
  const statusDot = gpu.status === 'busy' ? 'bg-cyan-400 animate-pulse' : gpu.status === 'idle' ? 'bg-emerald-400' : gpu.status === 'error' ? 'bg-red-400' : 'bg-slate-600'
  const statusLabel = gpu.status.charAt(0).toUpperCase() + gpu.status.slice(1)

  return (
    <div className="surface-card p-4" data-testid={`gpu-detail-${gpu.id}`}>
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[rgba(34,211,238,0.08)]">
          <Cpu size={20} className="text-cyan-300" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold text-slate-100">{gpu.name}</p>
          <div className="mt-0.5 flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${statusDot}`} />
            <span className="text-xs text-slate-400">{statusLabel}</span>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <div className="mb-1 flex justify-between">
            <div className="flex items-center gap-1.5">
              <Zap size={12} className="text-slate-500" />
              <span className="text-xs text-slate-400">GPU Utilization</span>
            </div>
            <span className="text-xs font-mono font-semibold text-slate-200">{util}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-700">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width: `${util}%` }} />
          </div>
        </div>

        <div>
          <div className="mb-1 flex justify-between">
            <div className="flex items-center gap-1.5">
              <HardDrive size={12} className="text-slate-500" />
              <span className="text-xs text-slate-400">VRAM</span>
            </div>
            <span className="text-xs font-mono text-slate-200">{gpu.memoryUsedGb.toFixed(1)} / {gpu.memoryTotalGb} GB ({memPct}%)</span>
          </div>
          <div className="h-2 rounded-full bg-slate-700">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width: `${memPct}%` }} />
          </div>
        </div>

        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-1.5">
            <Thermometer size={12} className="text-slate-500" />
            <span className="text-xs text-slate-400">Temperature</span>
          </div>
          <span className={`text-xs font-mono font-semibold ${gpu.temperatureCelsius > 80 ? 'text-red-400' : gpu.temperatureCelsius > 65 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {gpu.temperatureCelsius}°C
          </span>
        </div>
      </div>
    </div>
  )
}

function titleize(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export default function ComputePage() {
  const { data: health, isLoading, error, dataUpdatedAt } = useQuery({
    queryKey: ['system-health'],
    queryFn: dashboardApi.getSystemHealth,
    refetchInterval: 10_000,
  })

  const lastRefresh = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'

  return (
    <div className="mx-auto max-w-screen-2xl p-6 animate-fade-in" data-testid="compute-page">
      <PageHeader
        eyebrow="Operations"
        title="Compute And Runtime Status"
        description="Live infrastructure telemetry for the current deployment. This view focuses on resource readiness, artifact availability, and whether the workflow can keep moving when learned paths degrade."
        chips={[
          { label: `Last refresh ${lastRefresh}`, tone: 'neutral' },
          { label: 'Auto refresh every 10s', tone: 'info', icon: <Activity size={12} /> },
          { label: `${health?.capabilities.length ?? 0} subsystem checks`, tone: 'neutral', icon: <Cpu size={12} /> },
        ]}
      />

      {error && (
        <div className="mt-6 flex items-center gap-2 rounded-2xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300" data-testid="compute-error">
          <AlertTriangle size={15} />
          System health data is temporarily unavailable. The page will retry automatically.
        </div>
      )}

      <section className="mt-6 mb-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">GPU Compute</h2>
        {isLoading ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {[0, 1].map((index) => (
              <div key={index} className="surface-card p-4 space-y-3">
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-3/4" />
              </div>
            ))}
          </div>
        ) : health ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {health.gpus.map((gpu) => (
              <GpuDetailCard key={gpu.id} gpu={gpu} />
            ))}
          </div>
        ) : null}
      </section>

      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="hero-stat" data-testid="queue-panel">
          <p className="hero-stat-label">Inference Queue</p>
          {isLoading ? (
            <Skeleton className="mt-2 h-12 w-20" />
          ) : health ? (
            <>
              <p className="hero-stat-value">{health.queue.depth}</p>
              <p className="mt-2 text-sm text-slate-500">
                {health.queue.processingCount} processing
                {health.queue.estimatedWaitMinutes > 0 && ` · ~${health.queue.estimatedWaitMinutes}m wait`}
              </p>
            </>
          ) : null}
        </div>

        <div className="hero-stat" data-testid="latency-panel">
          <p className="hero-stat-label">API Latency</p>
          {isLoading ? (
            <Skeleton className="mt-2 h-12 w-20" />
          ) : health ? (
            <>
              <p className={`hero-stat-value ${health.apiLatencyMs < 100 ? 'text-emerald-300' : health.apiLatencyMs < 300 ? 'text-amber-300' : 'text-red-300'}`}>
                {health.apiLatencyMs}
              </p>
              <p className="mt-2 text-sm text-slate-500">milliseconds</p>
            </>
          ) : null}
        </div>

        <div className="hero-stat" data-testid="storage-panel">
          <p className="hero-stat-label">DICOM Storage</p>
          {isLoading ? (
            <Skeleton className="mt-2 h-12 w-20" />
          ) : health ? (
            <>
              <p className="hero-stat-value">
                {(health.storageUsedGb / 1024).toFixed(1)}<span className="text-base text-slate-500"> TB</span>
              </p>
              <div className="mt-3">
                <div className="h-1.5 rounded-full bg-slate-700">
                  <div className="h-full rounded-full bg-cyan-500" style={{ width: `${(health.storageUsedGb / health.storageTotalGb) * 100}%` }} />
                </div>
                <p className="mt-2 text-sm text-slate-500">of {(health.storageTotalGb / 1024).toFixed(0)} TB total</p>
              </div>
            </>
          ) : null}
        </div>
      </div>

      <section className="mb-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">Runtime Capability Posture</h2>
        <div className="surface-card p-5">
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-4 w-36" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : health ? (
            <>
              <div className="flex flex-wrap gap-2">
                {health.capabilities.map((capability) => (
                  <CapabilityBadge key={capability.name} capability={capability} compact />
                ))}
              </div>
              <div className="mt-5 overflow-x-auto">
                <table className="data-table w-full">
                  <thead>
                    <tr>
                      <th>Subsystem</th>
                      <th>Validation</th>
                      <th>Beta</th>
                      <th>Artifacts</th>
                      <th>Warnings</th>
                    </tr>
                  </thead>
                  <tbody>
                    {health.capabilities.map((capability) => (
                      <tr key={capability.name}>
                        <td>
                          <div>
                            <p className="text-sm font-medium text-slate-200">{titleize(capability.name)}</p>
                            <p className="text-xs text-slate-500">{titleize(capability.category)}</p>
                          </div>
                        </td>
                        <td><span className="text-xs text-slate-300">{titleize(capability.validationTier)}</span></td>
                        <td><span className="text-xs text-slate-300">{titleize(capability.betaStatus)}</span></td>
                        <td>
                          <span className="text-xs text-slate-400">
                            {capability.artifactRequired ? (capability.artifactReady ? 'Ready' : 'Missing') : 'Not required'}
                          </span>
                        </td>
                        <td>
                          <span className="text-xs text-slate-400">
                            {capability.warnings.length > 0 ? capability.warnings[0] : 'No warnings'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">Job Telemetry</h2>
        <div className="surface-card p-6">
          <div className="flex flex-col items-center justify-center py-6 text-center" data-testid="recent-jobs-empty">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-800">
              <RefreshCw size={20} className="text-slate-500" />
            </div>
            <p className="text-sm text-slate-300">Historical job telemetry is not yet surfaced in this UI.</p>
            <p className="mt-2 max-w-xl text-xs leading-5 text-slate-500">
              The deployment already exposes live queue depth and system readiness above. The next frontend slice for this page is a real job timeline rather than placeholder rows.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

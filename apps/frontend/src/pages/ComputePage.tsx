import { useQuery } from '@tanstack/react-query'
import { Cpu, Clock, Activity, RefreshCw, HardDrive, Thermometer, Zap } from 'lucide-react'
import { dashboardApi } from '../lib/api'
import { Skeleton } from '../components/common/LoadingOverlay'
import type { GpuStatus } from '../types/medical'

function GpuDetailCard({ gpu }: { gpu: GpuStatus }) {
  const util = gpu.utilizationPercent
  const memPct = Math.round((gpu.memoryUsedGb / gpu.memoryTotalGb) * 100)
  const barColor = util > 85 ? 'bg-red-500' : util > 60 ? 'bg-amber-500' : 'bg-cyan-500'
  const statusDot = gpu.status === 'busy' ? 'bg-cyan-400 animate-pulse' : gpu.status === 'idle' ? 'bg-emerald-400' : gpu.status === 'error' ? 'bg-red-400' : 'bg-slate-600'
  const statusLabel = gpu.status.charAt(0).toUpperCase() + gpu.status.slice(1)

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid={`gpu-detail-${gpu.id}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center">
          <Cpu size={20} className="text-cyan-400" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold text-slate-100">{gpu.name}</p>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`w-2 h-2 rounded-full ${statusDot}`} />
            <span className="text-xs text-slate-400">{statusLabel}</span>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <Zap size={12} className="text-slate-500" />
              <span className="text-xs text-slate-400">GPU Utilization</span>
            </div>
            <span className="text-xs font-mono font-semibold text-slate-200">{util}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width: `${util}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <HardDrive size={12} className="text-slate-500" />
              <span className="text-xs text-slate-400">VRAM</span>
            </div>
            <span className="text-xs font-mono text-slate-200">{gpu.memoryUsedGb.toFixed(1)} / {gpu.memoryTotalGb} GB ({memPct}%)</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full">
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

export default function ComputePage() {
  const { data: health, isLoading, error, dataUpdatedAt } = useQuery({
    queryKey: ['system-health'],
    queryFn: dashboardApi.getSystemHealth,
    refetchInterval: 10_000,
  })

  const lastRefresh = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'

  return (
    <div className="p-6 max-w-screen-xl mx-auto animate-fade-in" data-testid="compute-page">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Compute Resources</h1>
          <p className="text-sm text-slate-400 mt-0.5">GPU status, inference queue, and recent jobs</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 font-mono">Last refresh: {lastRefresh}</span>
          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
            <Activity size={12} className="animate-pulse" />
            Auto-refresh 10s
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-400 mb-6" data-testid="compute-error">
          System health data unavailable. Retrying automatically...
        </div>
      )}

      {/* GPU Cards */}
      <section className="mb-6">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">GPU Compute</h2>
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[0, 1].map(i => (
              <div key={i} className="bg-slate-800 border border-slate-700 rounded-lg p-4 space-y-3">
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-3/4" />
              </div>
            ))}
          </div>
        ) : health ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {health.gpus.map(gpu => (
              <GpuDetailCard key={gpu.id} gpu={gpu} />
            ))}
          </div>
        ) : null}
      </section>

      {/* Queue & Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="queue-panel">
          <div className="flex items-center gap-2 mb-3">
            <Clock size={16} className="text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-200">Inference Queue</h3>
          </div>
          {isLoading ? (
            <Skeleton className="h-12 w-20" />
          ) : health ? (
            <>
              <p className="font-mono text-3xl font-bold text-slate-100">{health.queue.depth}</p>
              <p className="text-xs text-slate-500 mt-1">
                {health.queue.processingCount} processing
                {health.queue.estimatedWaitMinutes > 0 && ` · ~${health.queue.estimatedWaitMinutes}m wait`}
              </p>
            </>
          ) : null}
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="latency-panel">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={16} className="text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-200">API Latency</h3>
          </div>
          {isLoading ? (
            <Skeleton className="h-12 w-20" />
          ) : health ? (
            <>
              <p className={`font-mono text-3xl font-bold ${health.apiLatencyMs < 100 ? 'text-emerald-400' : health.apiLatencyMs < 300 ? 'text-amber-400' : 'text-red-400'}`}>
                {health.apiLatencyMs}
              </p>
              <p className="text-xs text-slate-500 mt-1">milliseconds</p>
            </>
          ) : null}
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="storage-panel">
          <div className="flex items-center gap-2 mb-3">
            <HardDrive size={16} className="text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-200">DICOM Storage</h3>
          </div>
          {isLoading ? (
            <Skeleton className="h-12 w-20" />
          ) : health ? (
            <>
              <p className="font-mono text-3xl font-bold text-slate-100">
                {(health.storageUsedGb / 1024).toFixed(1)}<span className="text-base text-slate-500"> TB</span>
              </p>
              <div className="mt-2">
                <div className="h-1.5 bg-slate-700 rounded-full">
                  <div className="h-full rounded-full bg-cyan-500" style={{ width: `${(health.storageUsedGb / health.storageTotalGb) * 100}%` }} />
                </div>
                <p className="text-xs text-slate-500 mt-1">of {(health.storageTotalGb / 1024).toFixed(0)} TB total</p>
              </div>
            </>
          ) : null}
        </div>
      </div>

      {/* AI Models */}
      <section className="mb-6">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">AI Models</h2>
        <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
          <table className="data-table w-full">
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Version</th>
                <th>Accuracy</th>
                <th>Last Updated</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                Array.from({ length: 3 }).map((_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 5 }).map((_, j) => (
                      <td key={j} className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                    ))}
                  </tr>
                ))
              ) : health?.models.map(m => (
                <tr key={m.name} data-testid={`model-row-${m.name}`}>
                  <td><span className="text-sm font-medium text-slate-200">{m.name}</span></td>
                  <td><span className="text-xs text-slate-400 capitalize">{m.type}</span></td>
                  <td><span className="text-xs font-mono text-slate-400">v{m.version}</span></td>
                  <td><span className="font-mono text-sm font-semibold text-emerald-400">{Math.round(m.accuracy * 100)}%</span></td>
                  <td><span className="text-xs font-mono text-slate-500">{new Date(m.lastUpdated).toLocaleDateString()}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Recent Jobs */}
      <section>
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Recent Jobs</h2>
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
          <div className="flex flex-col items-center justify-center py-6 text-center" data-testid="recent-jobs-empty">
            <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center mb-3">
              <RefreshCw size={20} className="text-slate-500" />
            </div>
            <p className="text-sm text-slate-400">Recent jobs will appear here when available</p>
            <p className="text-xs text-slate-600 mt-1">Job history is fetched from the /jobs/ endpoint</p>
          </div>
        </div>
      </section>
    </div>
  )
}

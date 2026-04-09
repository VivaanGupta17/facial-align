import { useQuery } from '@tanstack/react-query'
import { BoxSelect, CheckCircle, AlertTriangle, XCircle, Clock } from 'lucide-react'
import { dashboardApi } from '../lib/api'
import { PageLoading, ErrorState } from '../components/common/LoadingOverlay'
import type { ModelInfo } from '../types/medical'

function StatusBadge({ model }: { model: ModelInfo }) {
  const accuracy = model.accuracy
  const status = accuracy >= 0.95 ? 'excellent' : accuracy >= 0.9 ? 'good' : accuracy >= 0.8 ? 'adequate' : 'degraded'
  const config = {
    excellent: { icon: <CheckCircle size={12} />, className: 'text-emerald-400 bg-emerald-950 border-emerald-800', label: 'Loaded' },
    good: { icon: <CheckCircle size={12} />, className: 'text-cyan-400 bg-cyan-950 border-cyan-800', label: 'Available' },
    adequate: { icon: <AlertTriangle size={12} />, className: 'text-amber-400 bg-amber-950 border-amber-800', label: 'Available' },
    degraded: { icon: <XCircle size={12} />, className: 'text-red-400 bg-red-950 border-red-800', label: 'Error' },
  }[status]
  return (
    <span className={`inline-flex items-center gap-1 text-2xs font-semibold px-1.5 py-0.5 rounded border ${config.className}`}>
      {config.icon} {config.label}
    </span>
  )
}

function TypeBadge({ type }: { type: ModelInfo['type'] }) {
  const config = {
    segmentation: 'text-blue-400 bg-blue-950 border-blue-800',
    planning: 'text-purple-400 bg-purple-950 border-purple-800',
    occlusion: 'text-amber-400 bg-amber-950 border-amber-800',
  }[type]
  return (
    <span className={`text-2xs font-semibold px-1.5 py-0.5 rounded border ${config}`}>
      {type}
    </span>
  )
}

export default function ModelsPage() {
  const { data: health, isLoading, error } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => dashboardApi.getSystemHealth(),
    staleTime: 30_000,
  })

  if (isLoading) return <PageLoading label="Loading model registry..." />
  if (error) return <ErrorState description="Failed to load model information" />

  const models = health?.models ?? []

  return (
    <div className="flex-1 flex flex-col p-6 animate-fade-in" data-testid="models-page">
      <div className="flex items-center gap-3 mb-6">
        <BoxSelect size={20} className="text-cyan-400" />
        <h1 className="text-lg font-semibold text-slate-100">ML Model Registry</h1>
        <span className="text-sm text-slate-500 font-mono">{models.length} models</span>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-6">
        {models.map(model => (
          <div
            key={model.name}
            className="bg-slate-800 border border-slate-700 rounded-lg p-5 hover:border-slate-600 transition-colors"
            data-testid={`model-card-${model.name}`}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">{model.name}</h3>
                <p className="text-xs text-slate-500 font-mono mt-0.5">v{model.version}</p>
              </div>
              <StatusBadge model={model} />
            </div>
            <div className="flex items-center gap-3 mb-3">
              <TypeBadge type={model.type} />
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <Clock size={11} />
                <span className="font-mono">{new Date(model.lastUpdated).toLocaleDateString()}</span>
              </div>
            </div>
            <div className="bg-slate-900 rounded-md p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-slate-400">Accuracy</span>
                <span className="text-sm font-mono font-bold text-slate-100">
                  {(model.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    model.accuracy >= 0.95 ? 'bg-emerald-500' :
                    model.accuracy >= 0.9 ? 'bg-cyan-500' :
                    model.accuracy >= 0.8 ? 'bg-amber-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${model.accuracy * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Detailed table */}
      <div className="flex-1 overflow-auto rounded-lg border border-slate-800">
        <table className="w-full text-sm" data-testid="models-table">
          <thead className="bg-slate-800/50 sticky top-0">
            <tr>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Name</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Version</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Type</th>
              <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Accuracy</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Last Updated</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {models.map(model => (
              <tr key={model.name} className="hover:bg-slate-800/30 transition-colors">
                <td className="px-4 py-3 text-sm font-medium text-slate-200">{model.name}</td>
                <td className="px-4 py-3 text-xs font-mono text-slate-400">v{model.version}</td>
                <td className="px-4 py-3"><TypeBadge type={model.type} /></td>
                <td className="px-4 py-3 text-right text-sm font-mono text-slate-200">
                  {(model.accuracy * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-xs font-mono text-slate-400">
                  {new Date(model.lastUpdated).toLocaleDateString()}
                </td>
                <td className="px-4 py-3 text-center"><StatusBadge model={model} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

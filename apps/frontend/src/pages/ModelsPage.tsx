import { useQuery } from '@tanstack/react-query'
import { BoxSelect, AlertTriangle, CheckCircle2, FlaskConical, ShieldCheck } from 'lucide-react'
import { dashboardApi } from '../lib/api'
import { PageLoading, ErrorState } from '../components/common/LoadingOverlay'
import PageHeader from '../components/common/PageHeader'
import { CapabilityBadge } from '../components/common/TrustIndicators'

function titleize(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export default function ModelsPage() {
  const { data: health, isLoading, error } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => dashboardApi.getSystemHealth(),
    staleTime: 30_000,
  })

  if (isLoading) {
    return (
      <div className="flex flex-1 flex-col p-6 animate-fade-in" data-testid="models-page">
        <PageLoading label="Loading capability registry..." />
      </div>
    )
  }

  if (error || !health) {
    return (
      <div className="flex flex-1 flex-col p-6 animate-fade-in" data-testid="models-page">
        <ErrorState description="Failed to load model and capability information" />
      </div>
    )
  }

  const capabilities = health.capabilities
  const readyCount = capabilities.filter((capability) => capability.status === 'available').length
  const betaCount = capabilities.filter((capability) => capability.betaStatus !== 'not_beta').length
  const warningCount = capabilities.reduce((count, capability) => count + capability.warnings.length, 0)

  return (
    <div className="flex flex-1 flex-col p-6 animate-fade-in" data-testid="models-page">
      <PageHeader
        eyebrow="Algorithm Registry"
        title="Models And Capability Status"
        description="This registry shows what the deployment can actually run today. It distinguishes deterministic baselines, learned beta paths, artifact readiness, and any warnings that affect research use."
        chips={[
          {
            label: `${readyCount}/${capabilities.length} capabilities ready`,
            tone: readyCount === capabilities.length ? 'success' : 'warning',
            icon: readyCount === capabilities.length ? <ShieldCheck size={12} /> : <FlaskConical size={12} />,
          },
          {
            label: `${betaCount} beta paths visible`,
            tone: betaCount > 0 ? 'warning' : 'neutral',
            icon: <BoxSelect size={12} />,
          },
          {
            label: `${warningCount} warnings`,
            tone: warningCount > 0 ? 'warning' : 'neutral',
            icon: <AlertTriangle size={12} />,
          },
        ]}
      />

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="hero-stat">
          <p className="hero-stat-label">Deterministic Baselines</p>
          <p className="hero-stat-value">
            {capabilities.filter((capability) => capability.validationTier === 'deterministic_baseline').length}
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Guaranteed paths the platform can fall back to when learned artifacts are unavailable.
          </p>
        </div>
        <div className="hero-stat">
          <p className="hero-stat-label">Learned Beta Paths</p>
          <p className="hero-stat-value">
            {capabilities.filter((capability) => capability.validationTier === 'learned_beta').length}
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Exposed to users, but explicitly labeled as beta and not clinically validated in this UI.
          </p>
        </div>
        <div className="hero-stat">
          <p className="hero-stat-label">Artifact Ready</p>
          <p className="hero-stat-value">
            {capabilities.filter((capability) => capability.artifactReady || !capability.artifactRequired).length}
          </p>
          <p className="mt-2 text-sm text-slate-500">
            Subsystems that have the assets needed to run without falling back to a manual or baseline path.
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 xl:grid-cols-[1.4fr_0.9fr]">
        <div className="surface-card overflow-hidden">
          <div className="panel-header">
            <div className="flex items-center gap-2">
              <BoxSelect size={15} className="text-cyan-400" />
              <h2 className="panel-title">Capability Registry</h2>
            </div>
            <span className="text-xs text-slate-500">{capabilities.length} subsystem entries</span>
          </div>

          <div className="overflow-x-auto">
            <table className="data-table w-full" data-testid="models-table">
              <thead>
                <tr>
                  <th>Subsystem</th>
                  <th>Category</th>
                  <th>Validation Tier</th>
                  <th>Beta Status</th>
                  <th>Artifacts</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {capabilities.map((capability) => (
                  <tr key={capability.name}>
                    <td>
                      <div>
                        <p className="text-sm font-medium text-slate-200">{titleize(capability.name)}</p>
                        <p className="text-xs font-mono text-slate-500">
                          {capability.modelVersion ?? 'No model version declared'}
                        </p>
                      </div>
                    </td>
                    <td>
                      <span className="text-xs text-slate-400">{titleize(capability.category)}</span>
                    </td>
                    <td>
                      <span className="text-xs text-slate-300">{titleize(capability.validationTier)}</span>
                    </td>
                    <td>
                      <span className="text-xs text-slate-300">{titleize(capability.betaStatus)}</span>
                    </td>
                    <td>
                      <span className="text-xs text-slate-400">
                        {capability.artifactRequired
                          ? capability.artifactReady
                            ? 'Ready'
                            : 'Missing artifact'
                          : 'Not required'}
                      </span>
                    </td>
                    <td>
                      <CapabilityBadge capability={capability} compact />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-4">
          <div className="surface-card p-5">
            <div className="flex items-center gap-2">
              <ShieldCheck size={16} className="text-emerald-400" />
              <h2 className="text-sm font-semibold text-slate-100">Operator Reading Guide</h2>
            </div>
            <div className="mt-4 space-y-3 text-sm text-slate-400">
              <p>
                <span className="text-slate-200">Available</span> means the platform can run that subsystem today with a declared path and the required assets.
              </p>
              <p>
                <span className="text-slate-200">Degraded</span> means the subsystem can still run, but it is using a fallback, a reduced path, or has warnings attached.
              </p>
              <p>
                <span className="text-slate-200">Unavailable</span> means the UI should not imply that this capability is ready for production use.
              </p>
            </div>
          </div>

          <div className="surface-card p-5">
            <div className="flex items-center gap-2">
              <CheckCircle2 size={16} className="text-cyan-400" />
              <h2 className="text-sm font-semibold text-slate-100">Current Warnings</h2>
            </div>
            <div className="mt-4 space-y-2">
              {capabilities.some((capability) => capability.warnings.length > 0) ? (
                capabilities
                  .filter((capability) => capability.warnings.length > 0)
                  .map((capability) => (
                    <div key={capability.name} className="surface-card-muted px-4 py-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                        {titleize(capability.name)}
                      </p>
                      <div className="mt-2 space-y-2">
                        {capability.warnings.map((warning) => (
                          <div key={warning} className="flex items-start gap-2 text-sm text-slate-400">
                            <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-400" />
                            <span>{warning}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
              ) : (
                <p className="text-sm text-slate-500">No active capability warnings reported by the backend.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

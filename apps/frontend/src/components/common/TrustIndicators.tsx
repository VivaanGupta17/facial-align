import { AlertTriangle, CheckCircle2, FlaskConical, ShieldCheck, Sparkles } from 'lucide-react'
import type { CapabilityInfo, ProvenanceInfo } from '../../types/medical'

function titleize(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function toneClass(status: string) {
  switch (status) {
    case 'available':
    case 'not_beta':
    case 'deterministic_baseline':
      return 'chip chip-success'
    case 'degraded':
    case 'fallback':
    case 'beta_available':
    case 'learned_beta':
      return 'chip chip-warning'
    case 'unavailable':
    case 'beta_unavailable':
      return 'chip chip-danger'
    case 'manual_override':
      return 'chip chip-info'
    default:
      return 'chip chip-neutral'
  }
}

function provenanceTone(provenance: ProvenanceInfo | null) {
  if (!provenance) return 'chip chip-neutral'
  if (provenance.fallbackReason) return 'chip chip-warning'
  if (provenance.validationTier === 'deterministic_baseline') return 'chip chip-success'
  if (provenance.validationTier === 'learned_beta') return 'chip chip-warning'
  return 'chip chip-info'
}

export function CapabilityBadge({
  capability,
  compact = false,
}: {
  capability: CapabilityInfo
  compact?: boolean
}) {
  return (
    <span
      className={`${toneClass(capability.status)} ${compact ? 'px-2 py-1 text-[11px]' : ''}`}
      title={capability.warnings[0] ?? undefined}
    >
      {capability.status === 'available' ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
      {titleize(compact ? capability.name.replace(/_(baseline|icp|v1)$/g, '') : capability.name)}
    </span>
  )
}

export function ProvenanceBadge({
  provenance,
  label = 'Algorithm Context',
}: {
  provenance: ProvenanceInfo | null
  label?: string
}) {
  if (!provenance) {
    return <span className="chip chip-neutral">{label}: unavailable</span>
  }

  return (
    <span className={provenanceTone(provenance)}>
      {provenance.validationTier === 'deterministic_baseline' ? <ShieldCheck size={12} /> : <FlaskConical size={12} />}
      {label}: {titleize(provenance.algorithmUsed)}
    </span>
  )
}

export function ProvenanceCard({
  provenance,
  title = 'Algorithm Provenance',
}: {
  provenance: ProvenanceInfo | null
  title?: string
}) {
  if (!provenance) {
    return (
      <div className="surface-card p-4">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-[var(--color-text-dim)]" />
          <p className="section-kicker">{title}</p>
        </div>
        <p className="mt-2 text-sm text-[var(--color-text-muted)]">
          No provenance metadata is available for this result yet.
        </p>
      </div>
    )
  }

  return (
    <div className="surface-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="section-kicker">{title}</p>
          <h3 className="mt-1 text-sm font-semibold text-[var(--color-text)]">
            {titleize(provenance.algorithmUsed)}
          </h3>
        </div>
        <ProvenanceBadge provenance={provenance} label="Path" />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="surface-card-muted px-3 py-3">
          <p className="micro-label">Validation Tier</p>
          <p className="mt-1 text-sm font-semibold text-[var(--color-text)]">
            {titleize(provenance.validationTier)}
          </p>
        </div>
        <div className="surface-card-muted px-3 py-3">
          <p className="micro-label">Beta Status</p>
          <p className="mt-1 text-sm font-semibold text-[var(--color-text)]">
            {titleize(provenance.betaStatus)}
          </p>
        </div>
        <div className="surface-card-muted px-3 py-3">
          <p className="micro-label">Model Version</p>
          <p className="mt-1 text-sm font-semibold text-[var(--color-text)]">
            {provenance.modelVersion ?? 'Not specified'}
          </p>
        </div>
      </div>

      {provenance.fallbackReason && (
        <div className="mt-4 rounded-2xl border border-[color:rgba(245,158,11,0.25)] bg-[rgba(245,158,11,0.08)] px-4 py-3 text-sm text-[var(--color-warning)]">
          <div className="flex items-start gap-2">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">Fallback Path Activated</p>
              <p className="mt-1 text-[color:rgba(251,191,36,0.9)]">{provenance.fallbackReason}</p>
            </div>
          </div>
        </div>
      )}

      {provenance.warnings.length > 0 && (
        <div className="mt-4">
          <p className="micro-label mb-2">Warnings</p>
          <div className="space-y-2">
            {provenance.warnings.map((warning) => (
              <div
                key={warning}
                className="flex items-start gap-2 rounded-2xl border border-[color:rgba(148,163,184,0.16)] bg-[rgba(148,163,184,0.06)] px-3 py-2 text-sm text-[var(--color-text-muted)]"
              >
                <AlertTriangle size={14} className="mt-0.5 shrink-0 text-[var(--color-warning)]" />
                <span>{warning}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

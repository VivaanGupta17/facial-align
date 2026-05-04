import { useEffect, useState } from 'react'
import { Cpu, Check, X, AlertTriangle, GitCompare } from 'lucide-react'
import { usePlan, useGeneratePlan, usePlanVersions } from '../../hooks/usePlanning'
import { useSegmentationResult } from '../../hooks/useSegmentation'
import { usePlanningStore } from '../../stores/planningStore'
import { useViewerStore } from '../../stores/viewerStore'
import Viewer3D from '../viewer/Viewer3D'
import FragmentControls from '../viewer/FragmentControls'
import RecommendationCard from '../common/RecommendationCard'
import { PageLoading, ErrorState, Spinner } from '../common/LoadingOverlay'
import { MetricInline } from '../common/MetricCard'
import { ProvenanceCard } from '../common/TrustIndicators'
import type { FragmentTransform, ConstraintValidation } from '../../types/medical'

function ValidationItem({ v }: { v: ConstraintValidation }) {
  return (
    <div
      className={`flex items-center gap-2 py-1.5 text-xs border-b border-slate-800 last:border-b-0 ${
        v.passed === false && v.severity === 'error' ? 'text-red-400' :
        v.passed === false && v.severity === 'warning' ? 'text-amber-400' :
        v.passed === true ? 'text-slate-300' : 'text-slate-500'
      }`}
      data-testid={`validation-${v.name.toLowerCase().replace(/ /g, '-')}`}
    >
      <span className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 ${
        v.passed === true ? 'bg-emerald-900 border border-emerald-700' :
        v.passed === false ? (v.severity === 'error' ? 'bg-red-900 border border-red-700' : 'bg-amber-900 border border-amber-700') :
        'bg-slate-700 border border-slate-600'
      }`}>
        {v.passed === true ? <Check size={9} className="text-emerald-400" /> :
         v.passed === false ? <X size={9} className={v.severity === 'error' ? 'text-red-400' : 'text-amber-400'} /> :
         <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />}
      </span>
      <span className="flex-1">{v.name}</span>
      {v.value != null && v.threshold != null && (
        <span className="font-mono text-2xs">
          {v.value.toFixed(1)} / {v.threshold.toFixed(1)}
        </span>
      )}
    </div>
  )
}

function FragmentStatusList({ fragments }: { fragments: FragmentTransform[] }) {
  const { selectFragment, selectedFragmentId } = usePlanningStore()
  const { setSelectedFragment } = useViewerStore()

  return (
    <div className="space-y-1" data-testid="fragment-list">
      {fragments.map(f => (
        <div
          key={f.fragmentId}
          onClick={() => {
            const nextId = f.fragmentId === selectedFragmentId ? null : f.fragmentId
            selectFragment(nextId)
            setSelectedFragment(nextId)
          }}
          className={`flex items-center gap-2 p-2.5 rounded-md border cursor-pointer transition-all ${
            selectedFragmentId === f.fragmentId
              ? 'border-cyan-700 bg-cyan-950/40'
              : 'border-slate-700 bg-slate-800 hover:border-slate-600'
          }`}
          data-testid={`fragment-item-${f.fragmentId}`}
        >
          <span className={`w-2 h-2 rounded-full shrink-0 ${f.isAligned ? 'bg-emerald-400' : 'bg-amber-400'}`} />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-slate-200 truncate">{f.displayName}</p>
            <p className="text-2xs font-mono text-slate-500">{f.volumeCm3.toFixed(1)} cm³</p>
          </div>
          <span className={`text-2xs font-semibold px-1.5 py-0.5 rounded border ${
            f.isAligned
              ? 'text-emerald-400 bg-emerald-950 border-emerald-800'
              : 'text-amber-400 bg-amber-950 border-amber-800'
          }`}>
            {f.isAligned ? 'Aligned' : 'Pending'}
          </span>
        </div>
      ))}
    </div>
  )
}

interface ReductionWorkspaceProps {
  caseId: string
  planId?: string
}

export default function ReductionWorkspace({ caseId, planId }: ReductionWorkspaceProps) {
  const [showFragmentControls, setShowFragmentControls] = useState(true)
  const [selectedVersionId, setSelectedVersionId] = useState(planId)

  const { data: plan, isLoading: planLoading } = usePlan(selectedVersionId ?? planId)
  const { data: segResult } = useSegmentationResult(caseId)
  const { data: versions } = usePlanVersions(caseId)
  const generatePlan = useGeneratePlan(caseId)
  const { setGenerating, setPlan } = usePlanningStore()

  useEffect(() => {
    if (plan) {
      setPlan(plan)
    }
  }, [plan, setPlan])

  if (planLoading) return <PageLoading label="Loading planning workspace..." />
  if (!plan && !planId) {
    return (
      <div className="flex items-center justify-center h-full" data-testid="no-plan-state">
        <div className="text-center max-w-sm space-y-4 p-8">
          <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mx-auto">
            <Cpu size={28} className="text-slate-500" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-200">No Reduction Plan</h3>
            <p className="text-sm text-slate-400 mt-1">Generate an AI-powered reduction plan to begin planning.</p>
          </div>
          <button
            onClick={async () => {
              setGenerating(true)
              await generatePlan.mutateAsync()
              setGenerating(false)
            }}
            disabled={generatePlan.isPending}
            className="flex items-center gap-2 btn-primary mx-auto"
            data-testid="generate-plan-btn"
          >
            {generatePlan.isPending ? <Spinner size={15} /> : <Cpu size={15} />}
            Generate AI Plan
          </button>
        </div>
      </div>
    )
  }

  if (!plan) return <ErrorState description="Plan not found" />

  const alignedCount = plan.fragmentTransforms.filter((f: FragmentTransform) => f.isAligned).length
  const totalCount = plan.fragmentTransforms.length
  const allValidationsPassed = plan.validations.every((v: ConstraintValidation) => v.passed !== false || v.severity !== 'error')

  return (
    <div className="flex h-full min-h-0 animate-fade-in" data-testid="reduction-workspace">
      {/* Left: 3D Viewer (60%) */}
      <div className="flex-1 min-w-0">
        <Viewer3D
          structures={segResult?.structures ?? []}
          fragments={plan.fragmentTransforms}
          height="100%"
        />
      </div>

      {/* Right: Planning panel (40%) */}
      <div className="w-[420px] shrink-0 flex flex-col overflow-hidden border-l border-white/10 bg-[rgba(8,14,26,0.84)] backdrop-blur-xl" data-testid="planning-panel">
        {/* Plan header */}
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">{plan.name}</h3>
              <p className="text-xs text-slate-500 mt-0.5">{plan.description}</p>
            </div>
            {/* Version selector */}
            {versions && versions.length > 1 && (
              <select
                value={selectedVersionId}
                onChange={e => setSelectedVersionId(e.target.value)}
                className="select-base w-28 text-xs"
                data-testid="version-selector"
              >
                {versions.map(v => (
                  <option key={v.id} value={v.id}>v{v.version}</option>
                ))}
              </select>
            )}
          </div>

          {/* Fragment alignment progress */}
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full">
              <div
                className="h-full rounded-full bg-emerald-500 transition-all"
                style={{ width: `${totalCount ? (alignedCount / totalCount) * 100 : 0}%` }}
              />
            </div>
            <span className="text-xs font-mono text-slate-300 shrink-0">{alignedCount}/{totalCount} aligned</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-4">
            <ProvenanceCard provenance={plan.provenance} title="Planning Provenance" />

            {/* AI Recommendation */}
            <RecommendationCard
              recommendation={plan.aiRecommendation}
              confidence={plan.aiConfidence}
              modelName="ReductionPlanner"
              acceptLabel="Accept All Suggestions"
              rejectLabel="Manual Override"
              onAccept={() => plan.fragmentTransforms.forEach((f: FragmentTransform) => console.log('accept', f.fragmentId))}
            />

            {/* Fragment list */}
            <div data-testid="fragment-section">
              <p className="label-xs mb-2">Fragment Status</p>
              <FragmentStatusList fragments={plan.fragmentTransforms} />
            </div>

            {/* Measurement readouts */}
            <div className="surface-card-muted p-3" data-testid="measurement-readouts">
              <p className="label-xs mb-2">Occlusal Measurements</p>
              <MetricInline
                label="Overjet"
                value={plan.occlusalMetrics.overjetMm.toFixed(1)}
                unit="mm"
                ideal="1–4 mm"
                withinRange={plan.occlusalMetrics.overjetMm >= plan.occlusalMetrics.overjetIdealMin && plan.occlusalMetrics.overjetMm <= plan.occlusalMetrics.overjetIdealMax}
              />
              <MetricInline
                label="Overbite"
                value={plan.occlusalMetrics.overbitePercent.toFixed(0)}
                unit="%"
                ideal="15–35%"
                withinRange={plan.occlusalMetrics.overbitePercent >= plan.occlusalMetrics.overbiteIdealMin && plan.occlusalMetrics.overbitePercent <= plan.occlusalMetrics.overbiteIdealMax}
              />
              <MetricInline
                label="Midline Deviation"
                value={plan.occlusalMetrics.midlineDeviationMm.toFixed(1)}
                unit="mm"
                ideal="< 2 mm"
                withinRange={plan.occlusalMetrics.midlineDeviationMm < plan.occlusalMetrics.midlineDeviationThreshold}
              />
              <MetricInline
                label="Occlusal Cant"
                value={plan.occlusalMetrics.occlusalCantDeg.toFixed(1)}
                unit="°"
                ideal="< 2°"
                withinRange={plan.occlusalMetrics.occlusalCantDeg < 2}
              />
            </div>

            {/* Constraint checklist */}
            <div className="surface-card-muted p-3" data-testid="constraint-checklist">
              <p className="label-xs mb-2">Constraint Validation</p>
              {plan.validations.map((v: ConstraintValidation) => (
                <ValidationItem key={v.name} v={v} />
              ))}
            </div>

            {/* Compare dropdown */}
            <div className="flex items-center gap-2">
              <GitCompare size={13} className="text-slate-500 shrink-0" />
              <select className="select-base text-xs flex-1" data-testid="compare-selector">
                <option value="">Compare with...</option>
                {versions?.filter(v => v.id !== selectedVersionId).map(v => (
                  <option key={v.id} value={v.id}>v{v.version} — {v.name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Fragment controls */}
        {showFragmentControls && (
          <div className="h-72 flex flex-col border-t border-white/10" data-testid="fragment-controls-section">
            <div className="panel-header py-2">
              <span className="text-xs font-semibold text-slate-300">Fragment Controls</span>
              <button onClick={() => setShowFragmentControls(false)} className="text-slate-500 hover:text-slate-300 text-xs">×</button>
            </div>
            <div className="flex-1 overflow-hidden">
              <FragmentControls />
            </div>
          </div>
        )}

        {/* Footer actions */}
        <div className="p-4 border-t border-white/10 space-y-2">
          <button
            onClick={async () => {
              setGenerating(true)
              await generatePlan.mutateAsync()
              setGenerating(false)
            }}
            disabled={generatePlan.isPending}
            className="w-full flex items-center justify-center gap-2 btn-secondary text-xs"
            data-testid="regenerate-plan-btn"
          >
            {generatePlan.isPending ? <><Spinner size={13} />Generating...</> : <><Cpu size={13} />Regenerate Plan</>}
          </button>
          <button
            disabled={!allValidationsPassed}
            className="w-full flex items-center justify-center gap-2 btn-primary disabled:opacity-40"
            data-testid="submit-review-btn"
          >
            Submit for Review →
          </button>
          {!allValidationsPassed && (
            <p className="text-center text-2xs text-red-400 flex items-center justify-center gap-1">
              <AlertTriangle size={10} /> Resolve all critical validations to submit
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

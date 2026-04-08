import { usePlan } from '../../hooks/usePlanning'
import { useSegmentationResult } from '../../hooks/useSegmentation'
import Viewer3D from '../viewer/Viewer3D'
import { MetricInline } from '../common/MetricCard'
import { PageLoading } from '../common/LoadingOverlay'
import type { AngleClass } from '../../types/medical'

function AngleClassBadge({ cls }: { cls: AngleClass }) {
  const config = {
    I: { label: 'Class I', color: 'text-emerald-400 bg-emerald-950 border-emerald-800' },
    II: { label: 'Class II', color: 'text-red-400 bg-red-950 border-red-800' },
    IIa: { label: 'Class IIa', color: 'text-amber-400 bg-amber-950 border-amber-800' },
    IIb: { label: 'Class IIb', color: 'text-amber-400 bg-amber-950 border-amber-800' },
    III: { label: 'Class III', color: 'text-blue-400 bg-blue-950 border-blue-800' },
  }[cls]

  return (
    <span className={`inline-flex items-center text-2xs font-bold px-1.5 py-0.5 rounded border ${config.color}`} data-testid={`angle-class-${cls}`}>
      {config.label}
    </span>
  )
}

function MeasurementGauge({
  label,
  value,
  min,
  max,
  idealMin,
  idealMax,
  unit,
}: {
  label: string
  value: number
  min: number
  max: number
  idealMin: number
  idealMax: number
  unit: string
}) {
  const range = max - min
  const valuePct = ((value - min) / range) * 100
  const idealMinPct = ((idealMin - min) / range) * 100
  const idealMaxPct = ((idealMax - min) / range) * 100
  const withinRange = value >= idealMin && value <= idealMax

  return (
    <div className="py-3 border-b border-slate-800 last:border-b-0" data-testid={`gauge-${label.toLowerCase().replace(' ', '-')}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-sm font-mono font-bold ${withinRange ? 'text-emerald-400' : 'text-red-400'}`}>
          {value.toFixed(1)} <span className="text-xs text-slate-500">{unit}</span>
        </span>
      </div>
      <div className="relative h-2 bg-slate-700 rounded-full">
        {/* Ideal range zone */}
        <div
          className="absolute inset-y-0 bg-emerald-900/60 rounded-full"
          style={{ left: `${idealMinPct}%`, width: `${idealMaxPct - idealMinPct}%` }}
        />
        {/* Current value marker */}
        <div
          className={`absolute inset-y-0 -translate-x-1/2 w-1 rounded-full ${withinRange ? 'bg-emerald-400' : 'bg-red-400'}`}
          style={{ left: `${Math.max(0, Math.min(100, valuePct))}%` }}
        />
      </div>
      <div className="flex justify-between mt-0.5">
        <span className="text-2xs font-mono text-slate-600">{min}{unit}</span>
        <span className="text-2xs text-slate-500">Ideal: {idealMin}–{idealMax} {unit}</span>
        <span className="text-2xs font-mono text-slate-600">{max}{unit}</span>
      </div>
    </div>
  )
}

interface OcclusionWorkspaceProps {
  caseId: string
  planId?: string
}

export default function OcclusionWorkspace({ caseId, planId }: OcclusionWorkspaceProps) {
  const { data: plan, isLoading } = usePlan(planId)
  const { data: segResult } = useSegmentationResult(caseId)

  if (isLoading) return <PageLoading label="Loading occlusion workspace..." />

  const m = plan?.occlusalMetrics
  const c = plan?.constraints

  // Filter to only dental structures for focused view
  const dentalStructures = segResult?.structures.filter(s =>
    ['teeth_upper', 'teeth_lower', 'maxilla', 'mandible'].includes(s.label)
  ) ?? []

  return (
    <div className="flex h-full min-h-0 animate-fade-in" data-testid="occlusion-workspace">
      {/* Left: 3D viewer focused on dental arches */}
      <div className="flex-1 min-w-0">
        <Viewer3D
          structures={dentalStructures}
          height="100%"
        />
      </div>

      {/* Right: Occlusal metrics panel */}
      <div className="w-[380px] shrink-0 flex flex-col border-l border-slate-800 bg-slate-900 overflow-y-auto" data-testid="occlusion-panel">
        <div className="panel-header border-b border-slate-800">
          <h3 className="panel-title">Occlusal Analysis</h3>
        </div>

        {m ? (
          <div className="p-4 space-y-5">
            {/* Molar relationship */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="molar-relationship">
              <p className="label-xs mb-3">Molar Relationship (Angle Classification)</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="text-center p-3 bg-slate-900 rounded-md border border-slate-700">
                  <p className="text-2xs text-slate-500 mb-2">Left</p>
                  <AngleClassBadge cls={m.molarRelationshipLeft} />
                  <p className="text-2xs text-slate-500 mt-2">Molar</p>
                </div>
                <div className="text-center p-3 bg-slate-900 rounded-md border border-slate-700">
                  <p className="text-2xs text-slate-500 mb-2">Right</p>
                  <AngleClassBadge cls={m.molarRelationshipRight} />
                  <p className="text-2xs text-slate-500 mt-2">Molar</p>
                </div>
                <div className="text-center p-3 bg-slate-900 rounded-md border border-slate-700">
                  <p className="text-2xs text-slate-500 mb-2">Left</p>
                  <AngleClassBadge cls={m.canineRelationshipLeft} />
                  <p className="text-2xs text-slate-500 mt-2">Canine</p>
                </div>
                <div className="text-center p-3 bg-slate-900 rounded-md border border-slate-700">
                  <p className="text-2xs text-slate-500 mb-2">Right</p>
                  <AngleClassBadge cls={m.canineRelationshipRight} />
                  <p className="text-2xs text-slate-500 mt-2">Canine</p>
                </div>
              </div>
            </div>

            {/* Measurement gauges */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="measurement-gauges">
              <p className="label-xs mb-1">Dimensional Analysis</p>
              <MeasurementGauge
                label="Overjet" value={m.overjetMm} min={-2} max={8}
                idealMin={m.overjetIdealMin} idealMax={m.overjetIdealMax} unit="mm"
              />
              <MeasurementGauge
                label="Overbite" value={m.overbitePercent} min={0} max={60}
                idealMin={m.overbiteIdealMin} idealMax={m.overbiteIdealMax} unit="%"
              />
              <MeasurementGauge
                label="Midline Deviation" value={m.midlineDeviationMm} min={0} max={6}
                idealMin={0} idealMax={m.midlineDeviationThreshold} unit="mm"
              />
              <MeasurementGauge
                label="Occlusal Cant" value={m.occlusalCantDeg} min={0} max={8}
                idealMin={0} idealMax={2} unit="°"
              />
            </div>

            {/* Dental constraints */}
            {c && (
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="dental-constraints">
                <p className="label-xs mb-3">Dental Constraints</p>
                <div className="space-y-2">
                  {[
                    { key: 'enforceOverjet', label: 'Enforce Overjet', value: c.enforceOverjet },
                    { key: 'enforceOverbite', label: 'Enforce Overbite', value: c.enforceOverbite },
                    { key: 'enforceMidline', label: 'Enforce Midline', value: c.enforceMidline },
                    { key: 'enforceSymmetry', label: 'Enforce Symmetry', value: c.enforceSymmetry },
                    { key: 'enforceCondylarSeating', label: 'Condylar Seating', value: c.enforceCondylarSeating },
                  ].map(opt => (
                    <label key={opt.key} className="flex items-center justify-between" data-testid={`constraint-${opt.key}`}>
                      <span className="text-xs text-slate-300">{opt.label}</span>
                      <div className={`relative w-8 h-4 rounded-full transition-colors ${opt.value ? 'bg-cyan-600' : 'bg-slate-600'}`}>
                        <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${opt.value ? 'translate-x-4' : 'translate-x-0.5'}`} />
                      </div>
                    </label>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-slate-700">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Max Condylar Deviation</span>
                    <span className="font-mono text-slate-300">{c.maxCondylarDeviationMm} mm</span>
                  </div>
                </div>
              </div>
            )}

            {/* Splint design preview */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="splint-preview">
              <div className="flex items-center justify-between mb-3">
                <p className="label-xs">Splint Design</p>
                <span className="text-2xs text-slate-500 bg-slate-700 px-1.5 py-0.5 rounded">Coming Soon</span>
              </div>
              <div className="h-24 rounded-md bg-slate-900 border border-dashed border-slate-700 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-slate-600 text-sm mb-1">⚙</div>
                  <p className="text-xs text-slate-600">Splint CAD preview will appear here</p>
                  <p className="text-2xs text-slate-700 mt-0.5">Requires approved occlusion plan</p>
                </div>
              </div>
              <button className="w-full mt-3 btn-secondary text-xs" disabled data-testid="export-splint-btn">
                Export Splint Design (STL)
              </button>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8 text-center">
            <div>
              <p className="text-sm font-medium text-slate-400">No occlusion data</p>
              <p className="text-xs text-slate-500 mt-1">Generate a reduction plan to see occlusal metrics</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

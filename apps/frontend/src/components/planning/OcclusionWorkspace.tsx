import { useState, useRef, useEffect } from 'react'
import { Pencil, Check, X, RotateCcw, Download } from 'lucide-react'
import { usePlan } from '../../hooks/usePlanning'
import { useSegmentationResult } from '../../hooks/useSegmentation'
import { usePlanningStore } from '../../stores/planningStore'
import { planningApi } from '../../lib/api'
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
    <div data-testid={`gauge-${label.toLowerCase().replace(' ', '-')}`}>
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

interface EditableMetricFieldProps {
  label: string
  value: number
  unit: string
  min: number
  max: number
  step: number
  idealMin: number
  idealMax: number
  gaugeMin: number
  gaugeMax: number
  onCommit: (newValue: number) => void
  isApplying?: boolean
}

function EditableMetricField({
  label,
  value,
  unit,
  min,
  max,
  step,
  idealMin,
  idealMax,
  gaugeMin,
  gaugeMax,
  onCommit,
  isApplying,
}: EditableMetricFieldProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const withinRange = value >= idealMin && value <= idealMax

  const handleEdit = () => {
    setEditValue(value.toFixed(1))
    setIsEditing(true)
  }

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.select()
    }
  }, [isEditing])

  const handleApply = () => {
    const parsed = parseFloat(editValue)
    if (!isNaN(parsed) && parsed >= min && parsed <= max) {
      onCommit(parsed)
      setIsEditing(false)
    }
  }

  const handleCancel = () => {
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleApply()
    if (e.key === 'Escape') handleCancel()
  }

  return (
    <div className="py-3 border-b border-slate-800 last:border-b-0" data-testid={`editable-metric-${label.toLowerCase().replace(' ', '-')}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400">{label}</span>
        <div className="flex items-center gap-2">
          {isEditing ? (
            <div className="flex items-center gap-1">
              <input
                ref={inputRef}
                type="number"
                value={editValue}
                onChange={e => setEditValue(e.target.value)}
                onKeyDown={handleKeyDown}
                step={step}
                min={min}
                max={max}
                className="w-16 text-sm font-mono text-right bg-slate-800 border border-cyan-500 rounded px-1 py-0.5 text-slate-100 outline-none"
                data-testid={`metric-input-${label.toLowerCase().replace(' ', '-')}`}
              />
              <span className="text-xs text-slate-500">{unit}</span>
              <button
                onClick={handleApply}
                className="p-0.5 rounded hover:bg-emerald-900 text-emerald-400"
                title="Apply"
                data-testid={`metric-apply-${label.toLowerCase().replace(' ', '-')}`}
              >
                <Check size={12} />
              </button>
              <button
                onClick={handleCancel}
                className="p-0.5 rounded hover:bg-red-900 text-red-400"
                title="Cancel"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 group">
              <span className={`text-sm font-mono font-bold ${
                isApplying ? 'text-amber-400 animate-pulse' : withinRange ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {value.toFixed(1)} <span className="text-xs text-slate-500">{unit}</span>
              </span>
              <button
                onClick={handleEdit}
                className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-slate-700 text-slate-400 hover:text-cyan-400 transition-opacity"
                title="Edit target value"
                data-testid={`metric-edit-${label.toLowerCase().replace(' ', '-')}`}
              >
                <Pencil size={11} />
              </button>
            </div>
          )}
        </div>
      </div>
      <MeasurementGauge
        label={label}
        value={value}
        min={gaugeMin}
        max={gaugeMax}
        idealMin={idealMin}
        idealMax={idealMax}
        unit={unit}
      />
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
  const { currentPlan, setPlan } = usePlanningStore()
  const [applyingMetric, setApplyingMetric] = useState<string | null>(null)
  const [molarOverride, setMolarOverride] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)

  if (isLoading) return <PageLoading label="Loading occlusion workspace..." />

  const m = plan?.occlusalMetrics
  const c = plan?.constraints

  // Filter to only dental structures for focused view
  const dentalStructures = segResult?.structures.filter(s =>
    ['teeth_upper', 'teeth_lower', 'maxilla', 'mandible'].includes(s.label)
  ) ?? []

  const handleMetricOverride = async (metricName: string, targetValue: number) => {
    if (!plan) return
    setApplyingMetric(metricName)
    try {
      const updatedPlan = await planningApi.overrideMetric(plan.id, metricName, targetValue)
      setPlan(updatedPlan)
    } catch {
      // Error handling — could add toast here
    } finally {
      setApplyingMetric(null)
    }
  }

  const handleResetToAi = async () => {
    if (!plan) return
    try {
      const freshPlan = await planningApi.getPlan(plan.id)
      setPlan(freshPlan)
    } catch {
      // Error handling
    }
  }

  const handleMolarOverride = async (angleClass: string) => {
    if (!plan) return
    setMolarOverride(angleClass)
    try {
      await planningApi.overrideMetric(plan.id, 'molar_class', 0, `Target: ${angleClass}`)
    } catch {
      // Error handling
    } finally {
      setMolarOverride(null)
    }
  }

  const handleExportSplint = async () => {
    if (!planId) return
    setExporting(true)
    try {
      const blob = await planningApi.exportSplint(planId, 'intermediate_splint')
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `splint-${planId}.stl`
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Splint export failed:', err)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 animate-fade-in" data-testid="occlusion-workspace">
      {/* Left: 3D viewer focused on dental arches */}
      <div className="flex-1 min-w-0">
        <Viewer3D
          structures={dentalStructures}
          fragments={currentPlan?.fragmentTransforms}
          height="100%"
        />
      </div>

      {/* Right: Occlusal metrics panel */}
      <div className="w-[380px] shrink-0 flex flex-col border-l border-slate-800 bg-slate-900 overflow-y-auto" data-testid="occlusion-panel">
        <div className="panel-header border-b border-slate-800">
          <div className="flex items-center justify-between w-full">
            <h3 className="panel-title">Occlusal Analysis</h3>
            <button
              onClick={handleResetToAi}
              className="flex items-center gap-1 text-2xs text-slate-400 hover:text-cyan-400 transition-colors"
              title="Reset all metrics to AI-predicted values"
              data-testid="reset-to-ai"
            >
              <RotateCcw size={11} /> Reset to AI
            </button>
          </div>
        </div>

        {m ? (
          <div className="p-4 space-y-5">
            {/* Molar relationship */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="molar-relationship">
              <div className="flex items-center justify-between mb-3">
                <p className="label-xs">Molar Relationship (Angle Classification)</p>
                <select
                  value={molarOverride ?? ''}
                  onChange={e => {
                    if (e.target.value) handleMolarOverride(e.target.value)
                  }}
                  className="text-2xs bg-slate-700 border border-slate-600 rounded px-1.5 py-0.5 text-slate-300 outline-none"
                  data-testid="molar-class-override"
                >
                  <option value="">Override...</option>
                  <option value="Class_I">Class I</option>
                  <option value="Class_II_div1">Class II div 1</option>
                  <option value="Class_II_div2">Class II div 2</option>
                  <option value="Class_III">Class III</option>
                </select>
              </div>
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

            {/* Editable measurement gauges */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="measurement-gauges">
              <p className="label-xs mb-1">Dimensional Analysis</p>
              <EditableMetricField
                label="Overjet"
                value={m.overjetMm}
                unit="mm"
                min={-4}
                max={10}
                step={0.1}
                idealMin={m.overjetIdealMin}
                idealMax={m.overjetIdealMax}
                gaugeMin={-2}
                gaugeMax={8}
                onCommit={(v) => handleMetricOverride('overjet_mm', v)}
                isApplying={applyingMetric === 'overjet_mm'}
              />
              <EditableMetricField
                label="Overbite"
                value={m.overbitePercent}
                unit="%"
                min={0}
                max={80}
                step={1}
                idealMin={m.overbiteIdealMin}
                idealMax={m.overbiteIdealMax}
                gaugeMin={0}
                gaugeMax={60}
                onCommit={(v) => handleMetricOverride('overbite_pct', v)}
                isApplying={applyingMetric === 'overbite_pct'}
              />
              <EditableMetricField
                label="Midline Deviation"
                value={m.midlineDeviationMm}
                unit="mm"
                min={0}
                max={10}
                step={0.1}
                idealMin={0}
                idealMax={m.midlineDeviationThreshold}
                gaugeMin={0}
                gaugeMax={6}
                onCommit={(v) => handleMetricOverride('midline_deviation_mm', v)}
                isApplying={applyingMetric === 'midline_deviation_mm'}
              />
              <EditableMetricField
                label="Occlusal Cant"
                value={m.occlusalCantDeg}
                unit="°"
                min={0}
                max={15}
                step={0.5}
                idealMin={0}
                idealMax={2}
                gaugeMin={0}
                gaugeMax={8}
                onCommit={(v) => handleMetricOverride('occlusal_cant_deg', v)}
                isApplying={applyingMetric === 'occlusal_cant_deg'}
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

            {/* Splint design */}
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="splint-preview">
              <div className="flex items-center justify-between mb-3">
                <p className="label-xs">Splint Design</p>
              </div>
              <div className="h-24 rounded-md bg-slate-900 border border-dashed border-slate-700 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-slate-600 text-sm mb-1">⚙</div>
                  <p className="text-xs text-slate-600">Splint CAD preview will appear here</p>
                  <p className="text-2xs text-slate-700 mt-0.5">Requires approved occlusion plan</p>
                </div>
              </div>
              <button
                onClick={handleExportSplint}
                disabled={!planId || exporting}
                className="w-full mt-3 flex items-center justify-center gap-2 btn-secondary text-xs disabled:opacity-40"
                data-testid="export-splint-btn"
              >
                <Download size={13} />
                {exporting ? 'Exporting...' : 'Export Splint Design (STL)'}
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

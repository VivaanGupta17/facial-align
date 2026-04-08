import { Check, X, RefreshCw, AlertTriangle, Cpu, Clock, Zap } from 'lucide-react'
import { useSegmentationResult, useApproveStructure, useRejectStructure } from '../../hooks/useSegmentation'
import ConfidenceBar, { ConfidenceRing } from '../common/ConfidenceBar'
import { PageLoading, ErrorState } from '../common/LoadingOverlay'
import Viewer3D from '../viewer/Viewer3D'
import { useViewerStore } from '../../stores/viewerStore'
import type { SegmentedStructure } from '../../types/medical'

function useRequestResegmentation(caseId: string) {
  // Placeholder — real implementation in useSegmentation.ts when backend is ready
  return { mutate: (_label: string) => console.log('Re-segment', caseId, _label), isPending: false }
}

interface StructureRowProps {
  structure: SegmentedStructure
  onAccept: () => void
  onReject: () => void
  onResegment: () => void
  isProcessing: boolean
}

function StructureRow({ structure, onAccept, onReject, onResegment, isProcessing }: StructureRowProps) {
  const { setStructureVisible, viewerState } = useViewerStore()
  const vis = viewerState.structureVisibility[structure.label]
  const passes = structure.confidence.passesClinicalThreshold

  const statusConfig = {
    accepted: { icon: <Check size={12} />, className: 'text-emerald-400 bg-emerald-950 border-emerald-800' },
    rejected: { icon: <X size={12} />, className: 'text-red-400 bg-red-950 border-red-800' },
    pending: { icon: null, className: 'text-slate-400 bg-slate-800 border-slate-700' },
    flagged: { icon: <AlertTriangle size={12} />, className: 'text-amber-400 bg-amber-950 border-amber-800' },
  }[structure.status]

  return (
    <div
      className={`px-4 py-3 border-b border-slate-800 last:border-b-0 hover:bg-slate-800/50 transition-colors ${
        structure.status === 'flagged' ? 'border-l-2 border-l-amber-600' : ''
      }`}
      data-testid={`structure-row-${structure.label}`}
    >
      <div className="flex items-center gap-3">
        {/* Visibility toggle + color */}
        <div className="flex items-center gap-2 w-4">
          <input
            type="checkbox"
            checked={vis?.visible ?? true}
            onChange={e => setStructureVisible(structure.label, e.target.checked)}
            className="rounded border-slate-600"
          />
        </div>
        <span
          className="w-3 h-3 rounded shrink-0"
          style={{ backgroundColor: structure.color }}
        />

        {/* Name + volume */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-200">{structure.displayName}</span>
            {!passes && (
              <AlertTriangle size={12} className="text-amber-400 shrink-0" />
            )}
            {structure.fragmentCount && structure.fragmentCount > 1 && (
              <span className="text-2xs bg-red-950 border border-red-800 text-red-400 px-1 rounded">
                {structure.fragmentCount} frags
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-xs text-slate-500 font-mono">{structure.volumeCm3.toFixed(1)} cm³</span>
            <span className="text-xs text-slate-500 font-mono">{structure.surfaceAreaCm2.toFixed(1)} cm²</span>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="w-32">
          <ConfidenceBar
            value={structure.confidence.value}
            threshold={structure.confidence.threshold}
            showValue
          />
        </div>

        {/* Status badge */}
        <span className={`flex items-center gap-1 text-2xs font-semibold px-1.5 py-0.5 rounded border ${statusConfig.className}`}>
          {statusConfig.icon}
          {structure.status}
        </span>

        {/* Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={onAccept}
            disabled={isProcessing || structure.status === 'accepted'}
            title="Accept structure"
            className={`w-7 h-7 rounded flex items-center justify-center transition-colors ${
              structure.status === 'accepted'
                ? 'bg-emerald-900 text-emerald-400 border border-emerald-800'
                : 'text-slate-500 hover:text-emerald-400 hover:bg-emerald-950'
            }`}
            data-testid={`accept-${structure.label}`}
          >
            <Check size={13} />
          </button>
          <button
            onClick={onReject}
            disabled={isProcessing || structure.status === 'rejected'}
            title="Reject structure"
            className="w-7 h-7 rounded flex items-center justify-center text-slate-500 hover:text-red-400 hover:bg-red-950 transition-colors"
            data-testid={`reject-${structure.label}`}
          >
            <X size={13} />
          </button>
          <button
            onClick={onResegment}
            disabled={isProcessing}
            title="Request re-segmentation"
            className="w-7 h-7 rounded flex items-center justify-center text-slate-500 hover:text-cyan-400 hover:bg-cyan-950 transition-colors"
            data-testid={`resegment-${structure.label}`}
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </div>
    </div>
  )
}

interface SegmentationReviewProps {
  caseId: string
}

export default function SegmentationReview({ caseId }: SegmentationReviewProps) {
  const { data: segResult, isLoading, error } = useSegmentationResult(caseId)
  const approveStructure = useApproveStructure(caseId)
  const rejectStructure = useRejectStructure(caseId)
  const requestResegmentation = useRequestResegmentation(caseId)

  if (isLoading) return <PageLoading label="Loading segmentation results..." />
  if (error || !segResult) return <ErrorState description="Failed to load segmentation results" />

  const acceptedCount = segResult.structures.filter(s => s.status === 'accepted').length
  const flaggedCount = segResult.structures.filter(s => s.status === 'flagged').length
  const allAccepted = acceptedCount === segResult.structures.length

  return (
    <div className="flex h-full min-h-0 animate-fade-in" data-testid="segmentation-review">
      {/* Left: 3D viewer */}
      <div className="flex-1 min-w-0">
        <Viewer3D
          structures={segResult.structures}
          height="100%"
        />
      </div>

      {/* Right: Review panel */}
      <div className="w-[400px] shrink-0 flex flex-col border-l border-slate-800 bg-slate-900" data-testid="segmentation-panel">
        {/* Model info header */}
        <div className="p-4 border-b border-slate-800 bg-slate-900">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Cpu size={15} className="text-cyan-400" />
              <span className="text-sm font-semibold text-slate-100">Segmentation Results</span>
            </div>
            <div className="ai-badge">AI</div>
          </div>

          {/* Overall confidence */}
          <div className="flex items-center gap-4 mb-3">
            <ConfidenceRing value={segResult.overallConfidence.value} size={64} />
            <div>
              <p className="text-xs text-slate-400 mb-1">Overall Confidence</p>
              <p className="text-2xl font-mono font-bold text-slate-100">
                {Math.round(segResult.overallConfidence.value * 100)}%
              </p>
              <p className={`text-xs font-semibold ${segResult.overallConfidence.passesClinicalThreshold ? 'text-emerald-400' : 'text-red-400'}`}>
                {segResult.overallConfidence.passesClinicalThreshold ? '✓ Passes clinical threshold' : '✗ Below clinical threshold'}
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-2 text-center">
            {[
              { label: 'Structures', value: segResult.structures.length },
              { label: 'Accepted', value: acceptedCount, color: 'text-emerald-400' },
              { label: 'Flagged', value: flaggedCount, color: 'text-amber-400' },
              { label: 'Pending', value: segResult.structures.length - acceptedCount - flaggedCount },
            ].map(s => (
              <div key={s.label} className="bg-slate-800 rounded p-2">
                <p className={`text-base font-mono font-bold ${s.color ?? 'text-slate-100'}`}>{s.value}</p>
                <p className="text-2xs text-slate-500">{s.label}</p>
              </div>
            ))}
          </div>

          {/* Model info */}
          <div className="flex items-center justify-between mt-3 text-xs text-slate-500">
            <div className="flex items-center gap-2">
              <span className="font-mono">{segResult.modelName} v{segResult.modelVersion}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock size={11} />
              <span className="font-mono">{segResult.inferenceTimeSeconds.toFixed(1)}s</span>
              <Zap size={11} className="ml-1" />
              <span className="font-mono text-2xs">{segResult.gpuUsed.replace('NVIDIA ', '')}</span>
            </div>
          </div>
        </div>

        {/* Warnings */}
        {segResult.warnings.length > 0 && (
          <div className="px-4 py-2 bg-amber-950/40 border-b border-amber-900/50" data-testid="segmentation-warnings">
            {segResult.warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-amber-400 py-1">
                <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                {w}
              </div>
            ))}
          </div>
        )}

        {/* Structures list */}
        <div className="flex-1 overflow-y-auto" data-testid="structures-list">
          {segResult.structures.map(s => (
            <StructureRow
              key={s.label}
              structure={s}
              onAccept={() => approveStructure.mutate(s.label)}
              onReject={() => rejectStructure.mutate(s.label)}
              onResegment={() => requestResegmentation.mutate(s.label)}
              isProcessing={approveStructure.isPending || rejectStructure.isPending}
            />
          ))}
        </div>

        {/* Footer actions */}
        <div className="p-4 border-t border-slate-800 space-y-2" data-testid="segmentation-actions">
          {!allAccepted && (
            <button
              className="w-full flex items-center justify-center gap-2 btn-secondary text-xs"
              data-testid="accept-all-btn"
            >
              <Check size={13} /> Accept All Structures
            </button>
          )}
          <button
            disabled={!allAccepted}
            className="w-full flex items-center justify-center gap-2 btn-primary disabled:opacity-40"
            data-testid="proceed-planning-btn"
          >
            Proceed to Planning →
          </button>
          {!allAccepted && (
            <p className="text-center text-2xs text-slate-500">
              Accept all {segResult.structures.length} structures to proceed
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

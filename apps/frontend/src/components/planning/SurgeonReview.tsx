import { useState } from 'react'
import { Check, X, Clock, MessageSquare, Pen, AlertTriangle, GitCompare, CheckSquare, Square } from 'lucide-react'
import { useSurgeonReview, useApproveReview, useRequestRevision, usePlan } from '../../hooks/usePlanning'
import { PageLoading, ErrorState } from '../common/LoadingOverlay'
import { MetricInline } from '../common/MetricCard'
import { ConfidenceRing } from '../common/ConfidenceBar'
import type { ReviewChecklist as ReviewChecklistItem } from '../../types/medical'

interface ChecklistSectionProps {
  category: string
  items: ReviewChecklistItem[]
  onToggle: (id: string, passed: boolean) => void
  disabled: boolean
}

function ChecklistSection({ category, items, onToggle, disabled }: ChecklistSectionProps) {
  const allPassed = items.every(i => i.passed === true)
  const anyFailed = items.some(i => i.passed === false)

  return (
    <div className="mb-4 last:mb-0" data-testid={`checklist-section-${category.toLowerCase()}`}>
      <div className="flex items-center gap-2 mb-2">
        <p className="label-xs">{category}</p>
        {allPassed && <span className="text-2xs text-emerald-400 font-semibold">✓ All passed</span>}
        {anyFailed && <span className="text-2xs text-red-400 font-semibold">Issues found</span>}
      </div>
      <div className="space-y-1">
        {items.map(item => (
          <div
            key={item.id}
            className={`flex items-start gap-2.5 p-2.5 rounded-md border cursor-pointer transition-all ${
              item.passed === true ? 'border-emerald-900/60 bg-emerald-950/20' :
              item.passed === false ? 'border-red-900/60 bg-red-950/20' :
              'border-slate-700 bg-slate-800 hover:border-slate-600'
            }`}
            onClick={() => !disabled && onToggle(item.id, item.passed !== true)}
            data-testid={`checklist-item-${item.id}`}
          >
            <span className={`mt-0.5 shrink-0 ${
              item.passed === true ? 'text-emerald-400' :
              item.passed === false ? 'text-red-400' : 'text-slate-500'
            }`}>
              {item.passed === true ? <CheckSquare size={15} /> : item.passed === false ? <X size={15} /> : <Square size={15} />}
            </span>
            <div className="flex-1 min-w-0">
              <span className={`text-xs ${
                item.passed === true ? 'text-emerald-300' :
                item.passed === false ? 'text-red-300' : 'text-slate-300'
              }`}>
                {item.label}
              </span>
              {item.severity === 'required' && (
                <span className="ml-1.5 text-2xs text-red-500 font-semibold">Required</span>
              )}
              {item.notes && (
                <p className="text-2xs text-slate-500 mt-0.5">{item.notes}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

interface SurgeonReviewProps {
  caseId: string
  planId?: string
}

export default function SurgeonReview({ caseId, planId }: SurgeonReviewProps) {
  const [notes, setNotes] = useState('')
  const [showComparison, setShowComparison] = useState(false)
  const [localChecklist, setLocalChecklist] = useState<ReviewChecklistItem[] | null>(null)

  const { data: review, isLoading: reviewLoading, error } = useSurgeonReview(caseId)
  const { data: plan } = usePlan(planId)
  const approveReview = useApproveReview(caseId)
  const requestRevision = useRequestRevision(caseId)

  if (reviewLoading) return <PageLoading label="Loading review panel..." />
  if (error || !review) return <ErrorState description="Failed to load review" />

  const checklist = localChecklist ?? review.checklist
  const categories = [...new Set(checklist.map(c => c.category))]
  const requiredItems = checklist.filter(c => c.severity === 'required')
  const allRequiredPassed = requiredItems.every(c => c.passed === true)
  const passedCount = checklist.filter(c => c.passed === true).length
  const failedCount = checklist.filter(c => c.passed === false).length

  const handleToggle = (id: string, passed: boolean) => {
    const base = localChecklist ?? review.checklist
    setLocalChecklist(base.map(c => c.id === id ? { ...c, passed: passed ? true : false } : c))
  }

  const handleApprove = async () => {
    await approveReview.mutateAsync({ reviewId: review.id, notes })
  }

  const handleRequestRevision = async () => {
    await requestRevision.mutateAsync({ reviewId: review.id, notes })
  }

  const isDecided = review.decision !== 'pending'

  return (
    <div className="flex h-full min-h-0 animate-fade-in" data-testid="surgeon-review">
      {/* Left: Plan summary */}
      <div className="flex-1 overflow-y-auto p-6 space-y-5" data-testid="review-content">
        {/* Plan summary card */}
        {plan && (
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-5" data-testid="plan-summary">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-base font-semibold text-slate-100">{plan.name}</h3>
                <p className="text-xs text-slate-400 mt-1">{plan.description}</p>
              </div>
              <ConfidenceRing value={plan.aiConfidence} size={72} />
            </div>

            <div className="grid grid-cols-2 gap-x-8 gap-y-0 border-t border-slate-700 pt-4">
              <MetricInline
                label="Overjet"
                value={plan.occlusalMetrics.overjetMm.toFixed(1)}
                unit="mm"
                withinRange={plan.occlusalMetrics.overjetMm <= plan.occlusalMetrics.overjetIdealMax}
              />
              <MetricInline
                label="Overbite"
                value={`${plan.occlusalMetrics.overbitePercent.toFixed(0)}%`}
                withinRange={plan.occlusalMetrics.overbitePercent <= plan.occlusalMetrics.overbiteIdealMax}
              />
              <MetricInline
                label="Midline"
                value={`${plan.occlusalMetrics.midlineDeviationMm.toFixed(1)} mm`}
                withinRange={plan.occlusalMetrics.midlineDeviationMm < 2}
              />
              <MetricInline
                label="Occlusal Cant"
                value={`${plan.occlusalMetrics.occlusalCantDeg.toFixed(1)}°`}
                withinRange={plan.occlusalMetrics.occlusalCantDeg < 2}
              />
            </div>

            {/* Before/after toggle (placeholder) */}
            <div className="mt-4 flex items-center gap-2">
              <button
                onClick={() => setShowComparison(c => !c)}
                className={`flex items-center gap-2 btn-secondary text-xs ${showComparison ? 'border-cyan-800 text-cyan-400' : ''}`}
                data-testid="comparison-toggle"
              >
                <GitCompare size={13} />
                {showComparison ? 'Hide Comparison' : 'Before / After Comparison'}
              </button>
            </div>

            {showComparison && (
              <div className="mt-3 p-4 bg-slate-900 border border-dashed border-slate-700 rounded-md text-center text-xs text-slate-500" data-testid="comparison-placeholder">
                Side-by-side 3D comparison will display here — requires both pre-op and planned positions to be computed
              </div>
            )}
          </div>
        )}

        {/* Validation summary */}
        <div className="grid grid-cols-3 gap-3" data-testid="validation-summary">
          {[
            { label: 'Passed', value: passedCount, color: 'text-emerald-400 bg-emerald-950/50 border-emerald-900' },
            { label: 'Failed', value: failedCount, color: 'text-red-400 bg-red-950/50 border-red-900' },
            { label: 'Pending', value: checklist.length - passedCount - failedCount, color: 'text-slate-400 bg-slate-800 border-slate-700' },
          ].map(s => (
            <div key={s.label} className={`border rounded-lg p-3 text-center ${s.color}`}>
              <p className="text-2xl font-mono font-bold">{s.value}</p>
              <p className="text-xs mt-0.5">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Checklist */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="review-checklist">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-200">Validation Checklist</h3>
            {!isDecided && (
              <span className="text-2xs text-slate-500">Click to toggle</span>
            )}
          </div>
          {categories.map(cat => (
            <ChecklistSection
              key={cat}
              category={cat}
              items={checklist.filter(c => c.category === cat)}
              onToggle={handleToggle}
              disabled={isDecided}
            />
          ))}
        </div>

        {/* Edit history timeline (compact) */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="edit-history">
          <h3 className="text-sm font-semibold text-slate-200 mb-3">Review History</h3>
          <div className="space-y-2 text-xs">
            {[
              { event: 'Review created', by: 'System', at: review.createdAt, icon: <Clock size={11} /> },
              ...(review.signedAt ? [{ event: 'Signed', by: review.reviewerName, at: review.signedAt, icon: <Pen size={11} /> }] : []),
            ].map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-slate-400">
                <span className="text-slate-600">{e.icon}</span>
                <span>{e.event}</span>
                <span className="text-slate-500">by {e.by}</span>
                <span className="ml-auto font-mono text-slate-600">
                  {new Date(e.at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Approval actions */}
      <div className="w-80 shrink-0 flex flex-col border-l border-slate-800 bg-slate-900" data-testid="approval-panel">
        <div className="panel-header border-b border-slate-800">
          <h3 className="panel-title">Surgical Approval</h3>
          {isDecided && (
            <span className={`text-2xs font-bold ${
              review.decision === 'approved' ? 'text-emerald-400' :
              review.decision === 'rejected' ? 'text-red-400' : 'text-amber-400'
            }`}>
              {review.decision.replace('_', ' ').toUpperCase()}
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Decision status */}
          {isDecided && (
            <div className={`p-4 rounded-lg border text-center ${
              review.decision === 'approved' ? 'bg-emerald-950/50 border-emerald-800' :
              review.decision === 'rejected' ? 'bg-red-950/50 border-red-800' :
              'bg-amber-950/50 border-amber-800'
            }`} data-testid="decision-status">
              <div className="text-2xl mb-1">
                {review.decision === 'approved' ? '✓' : review.decision === 'rejected' ? '✗' : '⟳'}
              </div>
              <p className={`text-sm font-semibold ${
                review.decision === 'approved' ? 'text-emerald-400' :
                review.decision === 'rejected' ? 'text-red-400' : 'text-amber-400'
              }`}>
                {review.decision === 'approved' ? 'Plan Approved' :
                 review.decision === 'rejected' ? 'Plan Rejected' : 'Revision Requested'}
              </p>
              {review.signedAt && (
                <p className="text-xs text-slate-400 mt-1">
                  Signed by {review.reviewerName} · {new Date(review.signedAt).toLocaleString()}
                </p>
              )}
            </div>
          )}

          {/* Requirement status */}
          {!isDecided && !allRequiredPassed && (
            <div className="flex items-start gap-2 p-3 bg-red-950/40 border border-red-900/50 rounded-lg text-xs text-red-400" data-testid="requirements-warning">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" />
              <div>
                <p className="font-semibold mb-0.5">Required items incomplete</p>
                <p>Complete all required checklist items before approving.</p>
              </div>
            </div>
          )}

          {/* Reviewer info */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-3">
            <p className="label-xs mb-2">Reviewing Surgeon</p>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-sm font-bold text-slate-100">
                {review.reviewerName.split(' ').map(n => n[0]).join('').slice(0, 2)}
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">{review.reviewerName}</p>
                <p className="text-xs text-slate-500">Oral & Maxillofacial Surgery</p>
              </div>
            </div>
          </div>

          {/* Notes textarea */}
          <div>
            <label className="label-sm block mb-1.5">Surgeon Notes</label>
            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              rows={5}
              placeholder="Clinical observations, concerns, recommendations..."
              className="input-base resize-none text-sm"
              disabled={isDecided}
              data-testid="surgeon-notes"
            />
          </div>

          {/* Digital signature placeholder */}
          {!isDecided && (
            <div className="border border-dashed border-slate-700 rounded-lg p-3 text-center" data-testid="signature-placeholder">
              <Pen size={16} className="text-slate-600 mx-auto mb-1" />
              <p className="text-xs text-slate-500">Digital signature will be applied on approval</p>
            </div>
          )}
        </div>

        {/* Action buttons */}
        {!isDecided && (
          <div className="p-4 border-t border-slate-800 space-y-2">
            <button
              onClick={handleApprove}
              disabled={!allRequiredPassed || approveReview.isPending}
              className="w-full flex items-center justify-center gap-2 btn-success disabled:opacity-40"
              data-testid="approve-btn"
            >
              <Check size={15} />
              Approve Plan
            </button>
            <button
              onClick={handleRequestRevision}
              disabled={requestRevision.isPending}
              className="w-full flex items-center justify-center gap-2 btn-secondary text-amber-400 border-amber-800 hover:bg-amber-950/30"
              data-testid="request-revision-btn"
            >
              <MessageSquare size={15} />
              Request Revision
            </button>
            <button
              className="w-full flex items-center justify-center gap-2 btn-ghost text-red-400 hover:bg-red-950/30 text-sm"
              data-testid="reject-btn"
            >
              <X size={15} />
              Reject Plan
            </button>

            {!allRequiredPassed && (
              <p className="text-center text-2xs text-slate-500">
                {requiredItems.filter(c => c.passed !== true).length} required items not confirmed
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

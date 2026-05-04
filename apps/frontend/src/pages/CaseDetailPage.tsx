import { lazy, Suspense, type ReactNode, useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Clock, User, FileText, Layers, Target, AlignCenter,
  ClipboardCheck, History, AlertTriangle, Play, CheckCircle, XCircle, Download,
  Plus, Star, Trash2, Link2,
} from 'lucide-react'
import { useCase } from '../hooks/useCases'
import { useCaseStore } from '../stores/caseStore'
import { useToastStore } from '../stores/toastStore'
import StatusBadge from '../components/common/StatusBadge'
import { PageLoading } from '../components/common/LoadingOverlay'
import JobProgressBar from '../components/common/JobProgressBar'
import { casesApi, caseStudiesApi } from '../lib/api'
import { useWebSocket, type WsMessage } from '../hooks/useWebSocket'
import type { SurgicalCase } from '../types/medical'

const SegmentationReview = lazy(() => import('../components/planning/SegmentationReview'))
const ReductionWorkspace = lazy(() => import('../components/planning/ReductionWorkspace'))
const OcclusionWorkspace = lazy(() => import('../components/planning/OcclusionWorkspace'))
const SurgeonReview = lazy(() => import('../components/planning/SurgeonReview'))
const ExportPanel = lazy(() => import('../components/planning/ExportPanel'))

type Tab = 'overview' | 'segmentation' | 'planning' | 'occlusion' | 'review' | 'history'

const TABS: Array<{ id: Tab; label: string; icon: ReactNode }> = [
  { id: 'overview', label: 'Overview', icon: <FileText size={14} /> },
  { id: 'segmentation', label: 'Segmentation', icon: <Layers size={14} /> },
  { id: 'planning', label: 'Planning', icon: <Target size={14} /> },
  { id: 'occlusion', label: 'Occlusion', icon: <AlignCenter size={14} /> },
  { id: 'review', label: 'Review', icon: <ClipboardCheck size={14} /> },
  { id: 'history', label: 'History', icon: <History size={14} /> },
]

// Irreversible transitions that need confirmation
const IRREVERSIBLE_TRANSITIONS = ['approved', 'rejected', 'completed', 'archived']

const TRANSITION_STYLES: Record<string, { bg: string; text: string; icon: React.ReactNode }> = {
  approved: { bg: 'bg-emerald-950 border-emerald-800 hover:bg-emerald-900', text: 'text-emerald-400', icon: <CheckCircle size={14} /> },
  rejected: { bg: 'bg-red-950 border-red-800 hover:bg-red-900', text: 'text-red-400', icon: <XCircle size={14} /> },
  completed: { bg: 'bg-emerald-950 border-emerald-800 hover:bg-emerald-900', text: 'text-emerald-400', icon: <CheckCircle size={14} /> },
  default: { bg: 'bg-slate-800 border-slate-700 hover:bg-slate-700', text: 'text-cyan-400', icon: <Play size={14} /> },
}

function fmtDateTime(iso: string | null | undefined) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function fmtCaseType(type: string) {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function truncateId(id: string | null) {
  if (!id) return '—'
  return id.length > 12 ? `${id.slice(0, 8)}...` : id
}

function fmtTransition(t: string) {
  return t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function WorkspaceFallback({ label }: { label: string }) {
  return (
    <div className="flex h-full min-h-[420px] items-center justify-center px-6 py-10" data-testid="case-workspace-loading">
      <PageLoading label={label} />
    </div>
  )
}

function renderWorkspace(element: ReactNode, label: string) {
  return <Suspense fallback={<WorkspaceFallback label={label} />}>{element}</Suspense>
}

// ---------------------------
// 404 Not Found
// ---------------------------
function CaseNotFound({ caseId }: { caseId?: string }) {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[400px] py-12 px-6 text-center animate-fade-in" data-testid="case-not-found">
      <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-4">
        <AlertTriangle size={28} className="text-amber-400" />
      </div>
      <h2 className="text-lg font-semibold text-slate-100 mb-2">Case Not Found</h2>
      <p className="text-sm text-slate-400 max-w-sm mb-6">
        {caseId ? `Case "${caseId}" does not exist or you don't have permission to view it.` : 'No case ID was provided.'}
      </p>
      <button
        onClick={() => navigate('/cases')}
        className="flex items-center gap-2 btn-primary"
        data-testid="back-to-cases-btn"
      >
        <ArrowLeft size={15} /> Back to Cases
      </button>
    </div>
  )
}

// ---------------------------
// Status Transition Buttons
// ---------------------------
function StatusTransitions({
  caseData,
  onTransition,
}: {
  caseData: SurgicalCase
  onTransition: (status: string) => void
}) {
  const [confirmingTransition, setConfirmingTransition] = useState<string | null>(null)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const { addToast } = useToastStore()

  const handleClick = async (transition: string) => {
    if (IRREVERSIBLE_TRANSITIONS.includes(transition) && confirmingTransition !== transition) {
      setConfirmingTransition(transition)
      return
    }

    setIsTransitioning(true)
    try {
      await casesApi.transitionStatus(caseData.id, transition)
      addToast({ type: 'success', message: `Case status changed to ${fmtTransition(transition)}` })
      onTransition(transition)
    } catch (err) {
      addToast({ type: 'error', message: `Failed to transition: ${err instanceof Error ? err.message : 'Unknown error'}` })
    } finally {
      setIsTransitioning(false)
      setConfirmingTransition(null)
    }
  }

  if (caseData.allowedTransitions.length === 0) return null

  return (
    <div className="flex items-center gap-2" data-testid="status-transitions">
      {caseData.allowedTransitions.map(t => {
        const style = TRANSITION_STYLES[t] ?? TRANSITION_STYLES.default
        const isConfirming = confirmingTransition === t

        return (
          <div key={t} className="flex items-center gap-1">
            {isConfirming && (
              <button
                onClick={() => setConfirmingTransition(null)}
                className="text-xs text-slate-500 hover:text-slate-300 px-2 py-1"
                data-testid={`cancel-confirm-${t}`}
              >
                Cancel
              </button>
            )}
            <button
              onClick={() => handleClick(t)}
              disabled={isTransitioning}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-xs font-medium transition-colors disabled:opacity-50 ${style.bg} ${style.text}`}
              data-testid={`transition-${t}`}
            >
              {style.icon}
              {isConfirming ? `Confirm ${fmtTransition(t)}` : fmtTransition(t)}
            </button>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------
// Overview Tab
// ---------------------------
function OverviewTab({ caseData }: { caseData: SurgicalCase }) {
  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in" data-testid="overview-tab">
      {/* Left: Case info */}
      <div className="lg:col-span-2 space-y-4">
        {/* Case details card */}
        <div className="card p-4 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2">Case Details</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {[
              { label: 'Case Number', value: caseData.caseNumber, mono: true },
              { label: 'Case Type', value: fmtCaseType(caseData.caseType) },
              { label: 'Status', value: <StatusBadge status={caseData.status} /> },
              { label: 'Fracture Classification', value: caseData.fractureClassification ?? 'Not classified' },
              { label: 'Created', value: fmtDateTime(caseData.createdAt) },
              { label: 'Last Updated', value: fmtDateTime(caseData.updatedAt) },
              { label: 'Target Surgery Date', value: caseData.targetSurgeryDate ? fmtDateTime(caseData.targetSurgeryDate) : 'Not scheduled' },
              { label: 'Study ID', value: truncateId(caseData.studyId), mono: true },
              { label: 'Planned Procedure', value: caseData.plannedProcedure ?? 'Not specified' },
              { label: 'Approved At', value: caseData.approvedAt ? fmtDateTime(caseData.approvedAt) : 'Not yet approved' },
            ].map(row => (
              <div key={row.label}>
                <p className="text-xs text-slate-500 mb-0.5">{row.label}</p>
                {typeof row.value === 'string' ? (
                  <p className={`text-slate-200 ${row.mono ? 'font-mono text-xs' : ''}`}>{row.value}</p>
                ) : row.value}
              </div>
            ))}
          </div>
        </div>

        {/* Studies section (multi-study) */}
        <div className="card p-4" data-testid="studies-section">
          <div className="flex items-center justify-between border-b border-slate-700 pb-2 mb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Link2 size={14} className="text-slate-400" />
              Attached Studies
            </h3>
            <a
              href={`/upload?caseId=${caseData.id}`}
              className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
              data-testid="add-study-link"
            >
              <Plus size={12} /> Add Study
            </a>
          </div>
          {(!caseData.studies || caseData.studies.length === 0) ? (
            <div className="text-sm text-slate-500 italic py-2">
              No additional studies attached. Primary study: <span className="font-mono text-xs text-slate-400">{truncateId(caseData.studyId)}</span>
            </div>
          ) : (
            <div className="space-y-2">
              {caseData.studies.map(cs => {
                const roleBg: Record<string, string> = {
                  pre_op: 'bg-blue-950 text-blue-400 border-blue-800',
                  post_op: 'bg-emerald-950 text-emerald-400 border-emerald-800',
                  follow_up: 'bg-amber-950 text-amber-400 border-amber-800',
                  intra_op: 'bg-violet-950 text-violet-400 border-violet-800',
                }
                return (
                  <div key={cs.id} className="flex items-center gap-3 p-2 bg-slate-900 rounded-md border border-slate-700" data-testid={`case-study-${cs.id}`}>
                    {cs.isPrimary && <Star size={14} className="text-amber-400 shrink-0" />}
                    <span className={`text-2xs px-1.5 py-0.5 rounded border ${roleBg[cs.studyRole] ?? 'bg-slate-800 text-slate-400 border-slate-700'}`}>
                      {cs.studyRole.replace(/_/g, ' ')}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-slate-200 truncate">{cs.studyLabel ?? cs.studyUid ?? truncateId(cs.studyId)}</p>
                      <p className="text-2xs text-slate-500">{cs.modality ?? 'Unknown'} — {cs.ingestionStatus ?? 'pending'}</p>
                    </div>
                    <button
                      onClick={async () => {
                        await caseStudiesApi.update(caseData.id, cs.studyId, { isPrimary: true })
                        window.location.reload()
                      }}
                      className="text-2xs text-slate-500 hover:text-amber-400 transition-colors"
                      title="Set as primary"
                      data-testid={`set-primary-${cs.id}`}
                    >
                      <Star size={12} />
                    </button>
                    <button
                      onClick={async () => {
                        await caseStudiesApi.detach(caseData.id, cs.studyId)
                        window.location.reload()
                      }}
                      className="text-2xs text-slate-500 hover:text-red-400 transition-colors"
                      title="Remove study"
                      data-testid={`remove-study-${cs.id}`}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Segmentation status */}
        {!caseData.latestSegmentation && (
          <div className="card p-4 border-slate-700" data-testid="no-segmentation">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center">
                <Layers size={20} className="text-slate-500" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-slate-300">No segmentation run yet</p>
                <p className="text-xs text-slate-500">Run AI segmentation to identify anatomical structures</p>
              </div>
              <button className="btn-primary text-xs" data-testid="run-segmentation-btn">
                Run Segmentation
              </button>
            </div>
          </div>
        )}

        {/* Plan status */}
        {!caseData.latestPlan && caseData.latestSegmentation && (
          <div className="card p-4 border-slate-700" data-testid="no-plan">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center">
                <Target size={20} className="text-slate-500" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-300">No reduction plan generated yet</p>
                <p className="text-xs text-slate-500">Generate an AI-powered surgical reduction plan</p>
              </div>
            </div>
          </div>
        )}

        {/* Diagnosis codes */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">Diagnosis Codes</h3>
          {!caseData.diagnosisCodes || caseData.diagnosisCodes.length === 0 ? (
            <p className="text-sm text-slate-500 italic">No diagnosis codes assigned</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {caseData.diagnosisCodes.map(code => (
                <span key={code} className="text-xs font-mono bg-slate-700 text-slate-300 px-2 py-1 rounded">
                  {code}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Error display */}
        {caseData.lastError && (
          <div className="card p-4 border-red-800 bg-red-950/30" data-testid="case-error">
            <h3 className="text-sm font-semibold text-red-400 border-b border-red-800 pb-2 mb-3">Last Error</h3>
            <p className="text-sm text-red-300 font-mono">{caseData.lastError}</p>
          </div>
        )}
      </div>

      {/* Right: Team + Processing info */}
      <div className="space-y-4">
        {/* Surgical Team */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">Surgical Team</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3 py-2" data-testid="surgeon-info">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-xs font-bold text-slate-100 shrink-0">
                <User size={14} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {caseData.surgeonId ? truncateId(caseData.surgeonId) : 'Unassigned'}
                </p>
                <p className="text-xs text-slate-500">Primary Surgeon</p>
              </div>
            </div>
            <div className="flex items-center gap-3 py-2" data-testid="reviewer-info">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-xs font-bold text-slate-100 shrink-0">
                <ClipboardCheck size={14} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {caseData.reviewerId ? truncateId(caseData.reviewerId) : 'Unassigned'}
                </p>
                <p className="text-xs text-slate-500">Reviewer</p>
              </div>
            </div>
          </div>
        </div>

        {/* Processing status */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">AI Processing</h3>
          <div className="space-y-2 text-xs">
            {[
              { label: 'Segmentations', value: String(caseData.segmentationCount), done: caseData.segmentationCount > 0 },
              { label: 'Latest Segmentation', value: caseData.latestSegmentation ? truncateId(caseData.latestSegmentation) : 'None', done: !!caseData.latestSegmentation },
              { label: 'Plans', value: String(caseData.planCount), done: caseData.planCount > 0 },
              { label: 'Latest Plan', value: caseData.latestPlan ? truncateId(caseData.latestPlan) : 'None', done: !!caseData.latestPlan },
              { label: 'Patient ID', value: truncateId(caseData.patientId), mono: true },
            ].map(r => (
              <div key={r.label} className="flex items-center justify-between">
                <span className="text-slate-500">{r.label}</span>
                <span className={`font-mono ${r.mono ? 'text-slate-400' : r.done ? 'text-emerald-400' : 'text-slate-500'}`}>{r.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------
// History Tab
// ---------------------------
function HistoryTab({ caseData }: { caseData: SurgicalCase }) {
  return (
    <div className="p-6 animate-fade-in" data-testid="history-tab">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">Audit Log & Timeline</h3>
      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-px bg-slate-700" />
        <div className="space-y-4">
          {[
            { id: 'created', event: 'Case Created', description: `Case ${caseData.caseNumber} created`, performedAt: caseData.createdAt, performedBy: caseData.createdBy ?? 'System' },
            ...(caseData.updatedAt !== caseData.createdAt ? [{ id: 'updated', event: 'Case Updated', description: `Status: ${caseData.status}`, performedAt: caseData.updatedAt, performedBy: 'System' }] : []),
            ...(caseData.approvedAt ? [{ id: 'approved', event: 'Case Approved', description: 'Case has been approved', performedAt: caseData.approvedAt, performedBy: caseData.reviewerId ?? 'Reviewer' }] : []),
          ].reverse().map((event) => (
            <div key={event.id} className="relative flex items-start gap-4 pl-10" data-testid={`timeline-event-${event.id}`}>
              <div className="absolute left-2.5 w-3 h-3 rounded-full border-2 border-slate-600 bg-slate-800 mt-0.5" />
              <div className="flex-1 bg-slate-800 border border-slate-700 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-slate-200">{event.event}</span>
                  <span className="text-2xs text-slate-500 ml-auto font-mono">{fmtDateTime(event.performedAt)}</span>
                </div>
                <p className="text-xs text-slate-400">{event.description}</p>
                <p className="text-2xs text-slate-600 mt-1">by {event.performedBy}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ---------------------------
// Main Detail Page
// ---------------------------
export default function CaseDetailPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const { data: caseData, isLoading, error, refetch } = useCase(caseId ?? '')
  const { setActiveCase } = useCaseStore()
  const { addToast } = useToastStore()
  const [showExportPanel, setShowExportPanel] = useState(false)

  // Job progress state for WebSocket
  const [jobProgress, setJobProgress] = useState<{ stage: string; progress: number } | null>(null)

  const handleWsMessage = useCallback((msg: WsMessage) => {
    if (msg.caseId !== caseId) return

    if (msg.type === 'SEGMENTATION_PROGRESS') {
      const m = msg as { stage: string; progress: number }
      setJobProgress({ stage: m.stage, progress: m.progress })
    } else if (msg.type === 'SEGMENTATION_COMPLETE' || msg.type === 'REDUCTION_COMPLETE') {
      setJobProgress(null)
      addToast({ type: 'success', message: msg.type === 'SEGMENTATION_COMPLETE' ? 'Segmentation complete' : 'Reduction plan ready' })
      refetch()
    } else if (msg.type === 'SEGMENTATION_FAILED' || msg.type === 'REDUCTION_FAILED' || msg.type === 'JOB_FAILED') {
      setJobProgress(null)
      const errMsg = (msg as { errorMessage?: string }).errorMessage ?? 'Job failed'
      addToast({ type: 'error', message: errMsg })
      refetch()
    } else if (msg.type === 'CASE_STATUS_CHANGED') {
      refetch()
    }
  }, [caseId, addToast, refetch])

  useWebSocket({ onMessage: handleWsMessage })

  useEffect(() => {
    if (caseData) setActiveCase(caseData)
    return () => setActiveCase(null)
  }, [caseData, setActiveCase])

  if (isLoading) return <PageLoading label="Loading case..." />
  if (error || !caseData) return <CaseNotFound caseId={caseId} />

  return (
    <div className="flex flex-col h-full animate-fade-in" data-testid="case-detail-page">
      {/* Job progress bar */}
      {jobProgress && (
        <div className="px-6 pt-3" data-testid="case-job-progress">
          <JobProgressBar stage={jobProgress.stage} progress={jobProgress.progress} />
        </div>
      )}

      {/* Case header */}
      <div className="border-b border-white/10 bg-[rgba(8,14,26,0.72)] px-6 py-4 backdrop-blur-xl">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => navigate('/cases')}
            className="btn-icon"
            data-testid="back-btn"
          >
            <ArrowLeft size={16} />
          </button>

          <div className="flex items-center gap-3 flex-1">
            <div>
              <h1 className="text-lg font-bold text-slate-100 font-mono">{caseData.caseNumber}</h1>
              <p className="text-xs text-slate-400">{fmtCaseType(caseData.caseType)}</p>
            </div>
            <StatusBadge status={caseData.status} />
          </div>

          {/* Status transition buttons */}
          <StatusTransitions caseData={caseData} onTransition={() => refetch()} />

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <User size={12} />
            {caseData.surgeonId ? truncateId(caseData.surgeonId) : 'Unassigned'}
            <Clock size={12} className="ml-2" />
            {fmtDateTime(caseData.updatedAt)}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="surface-card-muted px-4 py-3">
            <p className="micro-label">Segmentations</p>
            <p className="mt-1 text-lg font-semibold text-slate-100">{caseData.segmentationCount}</p>
          </div>
          <div className="surface-card-muted px-4 py-3">
            <p className="micro-label">Plans</p>
            <p className="mt-1 text-lg font-semibold text-slate-100">{caseData.planCount}</p>
          </div>
          <div className="surface-card-muted px-4 py-3">
            <p className="micro-label">Fracture Class</p>
            <p className="mt-1 text-sm font-medium text-slate-200">{caseData.fractureClassification ?? 'Not classified'}</p>
          </div>
          <div className="surface-card-muted px-4 py-3">
            <p className="micro-label">Latest Plan</p>
            <p className="mt-1 text-sm font-medium text-slate-200">{caseData.latestPlan ? truncateId(caseData.latestPlan) : 'Not generated'}</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="mt-4 flex flex-wrap gap-2" data-testid="case-tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border border-cyan-400/20 bg-[rgba(12,74,110,0.26)] text-cyan-300'
                  : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
              }`}
              data-testid={`tab-${tab.id}`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'overview' && <OverviewTab caseData={caseData} />}
        {activeTab === 'segmentation' && renderWorkspace(
          <SegmentationReview caseId={caseData.id} />,
          'Loading segmentation review...',
        )}
        {activeTab === 'planning' && (
          <div className="relative h-full">
            {caseData.latestPlan && (
              <div className="absolute top-4 right-4 z-10">
                {showExportPanel ? (
                  renderWorkspace(
                    <ExportPanel planId={caseData.latestPlan} onClose={() => setShowExportPanel(false)} />,
                    'Loading export tools...',
                  )
                ) : (
                  <button
                    onClick={() => setShowExportPanel(true)}
                    className="flex items-center gap-2 px-4 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium shadow-lg transition-colors"
                    data-testid="export-stl-btn"
                  >
                    <Download size={16} />
                    Export STL
                  </button>
                )}
              </div>
            )}
            {renderWorkspace(
              <ReductionWorkspace caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />,
              'Loading reduction workspace...',
            )}
          </div>
        )}
        {activeTab === 'occlusion' && renderWorkspace(
          <OcclusionWorkspace caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />,
          'Loading occlusion workspace...',
        )}
        {activeTab === 'review' && renderWorkspace(
          <SurgeonReview caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />,
          'Loading surgeon review...',
        )}
        {activeTab === 'history' && <HistoryTab caseData={caseData} />}
      </div>
    </div>
  )
}

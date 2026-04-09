import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Clock, User, FileText, Layers, Target, AlignCenter, ClipboardCheck, History } from 'lucide-react'
import { useCase } from '../hooks/useCases'
import { useCaseStore } from '../stores/caseStore'
import StatusBadge from '../components/common/StatusBadge'
import { PageLoading, ErrorState } from '../components/common/LoadingOverlay'
import SegmentationReview from '../components/planning/SegmentationReview'
import ReductionWorkspace from '../components/planning/ReductionWorkspace'
import OcclusionWorkspace from '../components/planning/OcclusionWorkspace'
import SurgeonReview from '../components/planning/SurgeonReview'

type Tab = 'overview' | 'segmentation' | 'planning' | 'occlusion' | 'review' | 'history'

const TABS: Array<{ id: Tab; label: string; icon: React.ReactNode }> = [
  { id: 'overview', label: 'Overview', icon: <FileText size={14} /> },
  { id: 'segmentation', label: 'Segmentation', icon: <Layers size={14} /> },
  { id: 'planning', label: 'Planning', icon: <Target size={14} /> },
  { id: 'occlusion', label: 'Occlusion', icon: <AlignCenter size={14} /> },
  { id: 'review', label: 'Review', icon: <ClipboardCheck size={14} /> },
  { id: 'history', label: 'History', icon: <History size={14} /> },
]

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

// ---------------------------
// Overview Tab
// ---------------------------
function OverviewTab({ caseData }: { caseData: NonNullable<ReturnType<typeof useCase>['data']> }) {
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

        {/* Allowed Transitions */}
        {caseData.allowedTransitions.length > 0 && (
          <div className="card p-4">
            <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">Available Actions</h3>
            <div className="flex flex-wrap gap-2">
              {caseData.allowedTransitions.map(t => (
                <span key={t} className="text-xs bg-cyan-950 text-cyan-400 border border-cyan-900 px-2 py-1 rounded capitalize">
                  {t.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error display */}
        {caseData.lastError && (
          <div className="card p-4 border-red-800 bg-red-950/30">
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
function HistoryTab({ caseData }: { caseData: NonNullable<ReturnType<typeof useCase>['data']> }) {
  return (
    <div className="p-6 animate-fade-in" data-testid="history-tab">
      <h3 className="text-sm font-semibold text-slate-200 mb-4">Audit Log & Timeline</h3>
      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-px bg-slate-700" />
        <div className="space-y-4">
          {/* Build a basic timeline from available case data */}
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
  const { data: caseData, isLoading, error } = useCase(caseId ?? '')
  const { setActiveCase } = useCaseStore()

  useEffect(() => {
    if (caseData) setActiveCase(caseData)
    return () => setActiveCase(null)
  }, [caseData, setActiveCase])

  if (isLoading) return <PageLoading label="Loading case..." />
  if (error || !caseData) return <ErrorState description={`Case ${caseId} not found`} onRetry={() => navigate('/cases')} />

  return (
    <div className="flex flex-col h-full animate-fade-in" data-testid="case-detail-page">
      {/* Case header */}
      <div className="px-6 py-4 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-4">
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

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <User size={12} />
            {caseData.surgeonId ? truncateId(caseData.surgeonId) : 'Unassigned'}
            <Clock size={12} className="ml-2" />
            {fmtDateTime(caseData.updatedAt)}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-4" data-testid="case-tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-cyan-950 text-cyan-400 border border-cyan-900'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
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
        {activeTab === 'segmentation' && <SegmentationReview caseId={caseData.id} />}
        {activeTab === 'planning' && <ReductionWorkspace caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />}
        {activeTab === 'occlusion' && <OcclusionWorkspace caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />}
        {activeTab === 'review' && <SurgeonReview caseId={caseData.id} planId={caseData.latestPlan ?? undefined} />}
        {activeTab === 'history' && <HistoryTab caseData={caseData} />}
      </div>
    </div>
  )
}

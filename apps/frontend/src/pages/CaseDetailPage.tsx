import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Clock, User, FileText, Layers, Target, AlignCenter, ClipboardCheck, History } from 'lucide-react'
import { useCase } from '../hooks/useCases'
import { useCaseStore } from '../stores/caseStore'
import StatusBadge, { PriorityBadge } from '../components/common/StatusBadge'
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

function fmtDateTime(iso: string) {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function fmtCaseType(type: string) {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

// ---------------------------
// Overview Tab
// ---------------------------
function OverviewTab({ caseData }: { caseData: NonNullable<ReturnType<typeof useCase>['data']> }) {
  const primarySurgeon = caseData.assignments.find(a => a.role === 'primary')
  const otherAssignees = caseData.assignments.filter(a => a.role !== 'primary')

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
              { label: 'Case Type', value: fmtCaseType(caseData.type) },
              { label: 'Priority', value: <PriorityBadge priority={caseData.priority} /> },
              { label: 'Status', value: <StatusBadge status={caseData.status} /> },
              { label: 'Created', value: fmtDateTime(caseData.createdAt) },
              { label: 'Last Updated', value: fmtDateTime(caseData.updatedAt) },
              { label: 'Scheduled', value: caseData.scheduledDate ? fmtDateTime(caseData.scheduledDate) : 'Not scheduled' },
              { label: 'Study ID', value: caseData.studyId, mono: true },
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

        {/* Notes */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">Clinical Notes</h3>
          {caseData.notes.length === 0 ? (
            <p className="text-sm text-slate-500 italic">No clinical notes</p>
          ) : caseData.notes.map(note => (
            <div key={note.id} className="border-l-2 border-cyan-800 pl-3 mb-3 last:mb-0" data-testid={`note-${note.id}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-300">{note.authorName}</span>
                <span className="text-xs text-slate-500">{fmtDateTime(note.createdAt)}</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{note.content}</p>
              {note.tags.length > 0 && (
                <div className="flex gap-1 mt-1.5">
                  {note.tags.map(t => (
                    <span key={t} className="text-2xs bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded">{t}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right: Assignments + Study info */}
      <div className="space-y-4">
        {/* Assignments */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">Surgical Team</h3>
          {caseData.assignments.map(a => (
            <div key={a.surgeonId} className="flex items-center gap-3 py-2" data-testid={`assignment-${a.surgeonId}`}>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-xs font-bold text-slate-100 shrink-0">
                {a.surgeonName.split(' ').map(n => n[0]).join('').slice(0, 2)}
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">{a.surgeonName}</p>
                <p className="text-xs text-slate-500 capitalize">{a.role}</p>
              </div>
              {a.role === 'primary' && (
                <span className="ml-auto text-2xs text-cyan-400 bg-cyan-950 border border-cyan-900 px-1.5 py-0.5 rounded">Primary</span>
              )}
            </div>
          ))}
        </div>

        {/* Processing status */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-700 pb-2 mb-3">AI Processing</h3>
          <div className="space-y-2 text-xs">
            {[
              { label: 'Segmentation Job', value: caseData.segmentationJobId ?? 'Not started', done: !!caseData.segmentationJobId },
              { label: 'Current Plan', value: caseData.currentPlanId ?? 'No plan', done: !!caseData.currentPlanId },
              { label: 'Patient ID', value: caseData.patientId, mono: true },
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
          {caseData.timeline.length === 0 ? (
            <p className="text-sm text-slate-500 pl-10">No events recorded yet</p>
          ) : [...caseData.timeline].reverse().map((event, i) => (
            <div key={event.id} className="relative flex items-start gap-4 pl-10" data-testid={`timeline-event-${event.id}`}>
              <div className="absolute left-2.5 w-3 h-3 rounded-full border-2 border-slate-600 bg-slate-800 mt-0.5" />
              <div className="flex-1 bg-slate-800 border border-slate-700 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-slate-200">
                    {event.event.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
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
              <p className="text-xs text-slate-400">{fmtCaseType(caseData.type)}</p>
            </div>
            <StatusBadge status={caseData.status} />
            <PriorityBadge priority={caseData.priority} />
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <User size={12} />
            {caseData.assignments.find(a => a.role === 'primary')?.surgeonName ?? 'Unassigned'}
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
        {activeTab === 'planning' && <ReductionWorkspace caseId={caseData.id} planId={caseData.currentPlanId} />}
        {activeTab === 'occlusion' && <OcclusionWorkspace caseId={caseData.id} planId={caseData.currentPlanId} />}
        {activeTab === 'review' && <SurgeonReview caseId={caseData.id} planId={caseData.currentPlanId} />}
        {activeTab === 'history' && <HistoryTab caseData={caseData} />}
      </div>
    </div>
  )
}

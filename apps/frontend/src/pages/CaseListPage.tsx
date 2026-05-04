import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Filter, Calendar, ChevronDown, ChevronUp, Plus, SortAsc, SortDesc, Upload, FolderOpen, X } from 'lucide-react'
import { useCases } from '../hooks/useCases'
import StatusBadge from '../components/common/StatusBadge'
import { ErrorState, EmptyState, TableSkeleton } from '../components/common/LoadingOverlay'
import PageHeader from '../components/common/PageHeader'
import type { CaseStatus, CaseType } from '../types/medical'

const STATUS_OPTIONS: { value: CaseStatus | ''; label: string }[] = [
  { value: '', label: 'All Statuses' },
  { value: 'pending_upload', label: 'Pending Upload' },
  { value: 'segmentation_in_progress', label: 'Segmenting' },
  { value: 'segmentation_review', label: 'Seg. Review' },
  { value: 'planning', label: 'Planning' },
  { value: 'review', label: 'In Review' },
  { value: 'approved', label: 'Approved' },
  { value: 'completed', label: 'Completed' },
]

const TYPE_OPTIONS: { value: CaseType | ''; label: string }[] = [
  { value: '', label: 'All Types' },
  { value: 'mandible_fracture', label: 'Mandible Fracture' },
  { value: 'midface_fracture', label: 'Midface Fracture' },
  { value: 'panfacial_fracture', label: 'Panfacial Fracture' },
  { value: 'orbital_fracture', label: 'Orbital Fracture' },
  { value: 'frontal_sinus_fracture', label: 'Frontal Sinus' },
  { value: 'orthognathic', label: 'Orthognathic' },
  { value: 'tumor_resection', label: 'Tumor Resection' },
  { value: 'reconstruction', label: 'Reconstruction' },
]

function fmtCaseType(type: string) {
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function truncateId(id: string) {
  if (!id) return '—'
  return id.length > 8 ? `${id.slice(0, 8)}...` : id
}

type SortKey = 'caseNumber' | 'status' | 'updatedAt'

export default function CaseListPage() {
  const navigate = useNavigate()

  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<CaseStatus | ''>('')
  const [typeFilter, setTypeFilter] = useState<CaseType | ''>('')
  const [sortKey, setSortKey] = useState<SortKey>('updatedAt')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [showFilters, setShowFilters] = useState(false)
  const [page, setPage] = useState(1)

  const hasActiveFilters = !!statusFilter || !!typeFilter || !!search

  const clearFilters = () => {
    setSearch('')
    setStatusFilter('')
    setTypeFilter('')
    setPage(1)
  }

  const { data, isLoading, error, refetch } = useCases({
    page,
    status: statusFilter ? [statusFilter] : undefined,
    type: typeFilter ? [typeFilter] : undefined,
    search: search || undefined,
    sortBy: sortKey === 'caseNumber' ? 'caseNumber' : 'updatedAt',
    sortOrder: sortDir,
  })

  const displayItems = [...(data?.items ?? [])]
    .filter((item) => {
      if (!search.trim()) return true
      const query = search.trim().toLowerCase()
      return [
        item.caseNumber,
        item.patientId,
        item.fractureClassification ?? '',
        item.caseType,
      ]
        .some((value) => value.toLowerCase().includes(query))
    })
    .sort((a, b) => {
      const direction = sortDir === 'asc' ? 1 : -1

      if (sortKey === 'caseNumber') {
        return a.caseNumber.localeCompare(b.caseNumber) * direction
      }

      if (sortKey === 'status') {
        return a.status.localeCompare(b.status) * direction
      }

      return (new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime()) * direction
    })

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (sortKey !== k) return <SortAsc size={12} className="text-slate-600" />
    return sortDir === 'asc' ? <SortAsc size={12} className="text-cyan-400" /> : <SortDesc size={12} className="text-cyan-400" />
  }

  return (
    <div className="p-6 space-y-4 animate-fade-in" data-testid="case-list-page">
      <PageHeader
        eyebrow="Case Management"
        title="Cases"
        description="Browse the active fracture-planning workload, filter by workflow state, and jump directly into segmentation, planning, or review."
        chips={[
          { label: `${data?.total ?? 0} total cases`, tone: 'neutral', icon: <FolderOpen size={12} /> },
          { label: `${hasActiveFilters ? 'Filters active' : 'All cases visible'}`, tone: hasActiveFilters ? 'info' : 'neutral', icon: <Search size={12} /> },
          { label: `Page ${page}`, tone: 'neutral' },
        ]}
        actions={
          <button
            onClick={() => navigate('/upload')}
            className="flex items-center gap-2 btn-primary"
            data-testid="new-case-btn"
          >
            <Plus size={15} />
            New Case
          </button>
        }
      />

      {data && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="hero-stat">
            <p className="hero-stat-label">Awaiting Segmentation</p>
            <p className="hero-stat-value">
              {displayItems.filter((item) => item.status === 'segmentation_in_progress' || item.status === 'segmentation_review').length}
            </p>
            <p className="mt-2 text-sm text-slate-500">Cases still moving through structure generation and QC.</p>
          </div>
          <div className="hero-stat">
            <p className="hero-stat-label">In Planning</p>
            <p className="hero-stat-value">
              {displayItems.filter((item) => item.status === 'planning').length}
            </p>
            <p className="mt-2 text-sm text-slate-500">Cases actively being reduced, reviewed, or manually adjusted.</p>
          </div>
          <div className="hero-stat">
            <p className="hero-stat-label">Ready For Review</p>
            <p className="hero-stat-value">
              {displayItems.filter((item) => item.status === 'review' || item.status === 'approved').length}
            </p>
            <p className="mt-2 text-sm text-slate-500">Plans that already have enough structure to support surgeon sign-off.</p>
          </div>
        </div>
      )}

      {/* Search + Filter bar */}
      <div className="surface-card p-3" data-testid="filter-bar">
        <div className="flex gap-2">
          {/* Search */}
          <div className="relative flex-1">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search case number, patient ID..."
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1) }}
              className="input-base pl-9"
              data-testid="search-input"
            />
          </div>

          {/* Status filter */}
          <div className="relative">
            <select
              value={statusFilter}
              onChange={e => { setStatusFilter(e.target.value as CaseStatus | ''); setPage(1) }}
              className="select-base w-44"
              data-testid="status-filter"
            >
              {STATUS_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          {/* Type filter */}
          <div className="relative">
            <select
              value={typeFilter}
              onChange={e => { setTypeFilter(e.target.value as CaseType | ''); setPage(1) }}
              className="select-base w-44"
              data-testid="type-filter"
            >
              {TYPE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          {/* Clear filters */}
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1.5 btn-ghost text-sm text-slate-400 hover:text-slate-200"
              data-testid="clear-filters-btn"
            >
              <X size={14} />
              Clear
            </button>
          )}

          {/* Advanced filters toggle */}
          <button
            onClick={() => setShowFilters(f => !f)}
            className={`flex items-center gap-1.5 btn-secondary ${showFilters ? 'border-cyan-800 text-cyan-400' : ''}`}
          >
            <Filter size={14} />
            Filters
            {showFilters ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {/* Advanced filters */}
        {showFilters && (
          <div className="mt-3 pt-3 border-t border-slate-700 grid grid-cols-3 gap-3 animate-slide-in-up" data-testid="advanced-filters">
            <div>
              <label className="label-sm block mb-1">Date From</label>
              <div className="relative">
                <Calendar size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input type="date" className="input-base pl-9 text-slate-400" data-testid="date-from" />
              </div>
            </div>
            <div>
              <label className="label-sm block mb-1">Date To</label>
              <div className="relative">
                <Calendar size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input type="date" className="input-base pl-9 text-slate-400" data-testid="date-to" />
              </div>
            </div>
            <div>
              <label className="label-sm block mb-1">Surgeon</label>
              <select className="select-base" data-testid="surgeon-filter">
                <option>All Surgeons</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="surface-card overflow-hidden" data-testid="cases-table">
        {error ? (
          <ErrorState description="Failed to load cases" onRetry={refetch} />
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table w-full">
              <thead>
                <tr>
                  <th>
                    <button onClick={() => handleSort('caseNumber')} className="flex items-center gap-1 hover:text-slate-300">
                      Case # <SortIcon k="caseNumber" />
                    </button>
                  </th>
                  <th>Patient ID</th>
                  <th>Type</th>
                  <th>
                    <button onClick={() => handleSort('status')} className="flex items-center gap-1 hover:text-slate-300">
                      Status <SortIcon k="status" />
                    </button>
                  </th>
                  <th>Surgeon</th>
                  <th>Seg. Status</th>
                  <th>
                    <button onClick={() => handleSort('updatedAt')} className="flex items-center gap-1 hover:text-slate-300">
                      Updated <SortIcon k="updatedAt" />
                    </button>
                  </th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  <TableSkeleton rows={8} cols={8} />
                ) : displayItems.length === 0 ? (
                  <tr>
                    <td colSpan={8}>
                      {hasActiveFilters ? (
                        <EmptyState
                          title="No cases match your filters"
                          description="Try adjusting your search or filter criteria."
                          icon={<Search size={32} />}
                          action={
                            <button onClick={clearFilters} className="btn-secondary" data-testid="empty-clear-filters-btn">
                              Clear Filters
                            </button>
                          }
                        />
                      ) : (
                        <EmptyState
                          title="No cases found"
                          description="Upload a DICOM study to get started."
                          icon={<FolderOpen size={32} />}
                          action={
                            <button onClick={() => navigate('/upload')} className="flex items-center gap-2 btn-primary" data-testid="empty-upload-btn">
                              <Upload size={15} /> Upload DICOM
                            </button>
                          }
                        />
                      )}
                    </td>
                  </tr>
                ) : displayItems.map(c => (
                  <tr
                    key={c.id}
                    onClick={() => navigate(`/cases/${c.id}`)}
                    className="cursor-pointer"
                    data-testid={`case-row-${c.id}`}
                  >
                    <td>
                      <span className="font-mono text-sm font-semibold text-slate-100">{c.caseNumber}</span>
                    </td>
                    <td>
                      <span className="font-mono text-xs text-slate-400">{truncateId(c.patientId)}</span>
                    </td>
                    <td><span className="text-xs text-slate-300">{fmtCaseType(c.caseType)}</span></td>
                    <td><StatusBadge status={c.status} size="sm" /></td>
                    <td><span className="text-xs text-slate-400">{c.surgeonId ? truncateId(c.surgeonId) : 'Unassigned'}</span></td>
                    <td>
                      <span className="text-xs text-slate-400 font-mono">
                        {c.latestSegmentationStatus ?? '—'}
                      </span>
                    </td>
                    <td><span className="text-xs font-mono text-slate-500">{fmtDate(c.updatedAt)}</span></td>
                    <td>
                      <button
                        onClick={(e) => { e.stopPropagation(); navigate(`/cases/${c.id}`) }}
                        className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
                        data-testid={`open-case-${c.id}`}
                      >
                        Open →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {data && data.total > data.pageSize && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-slate-700" data-testid="pagination">
            <span className="text-xs text-slate-500">
              Showing {(data.page - 1) * data.pageSize + 1}–{Math.min(data.page * data.pageSize, data.total)} of {data.total}
            </span>
            <div className="flex gap-1">
              <button
                className="btn-secondary text-xs px-3 py-1.5"
                disabled={data.page <= 1}
                onClick={() => setPage(p => Math.max(1, p - 1))}
                data-testid="pagination-prev"
              >
                Previous
              </button>
              <button
                className="btn-secondary text-xs px-3 py-1.5"
                disabled={!data.hasMore}
                onClick={() => setPage(p => p + 1)}
                data-testid="pagination-next"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

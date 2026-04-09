import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Scan, Search, ChevronLeft, ChevronRight } from 'lucide-react'
import { studiesApi } from '../lib/api'
import { PageLoading, ErrorState } from '../components/common/LoadingOverlay'
import type { Study } from '../types/medical'

function QualityBadge({ series }: { series: Study['series'] }) {
  const mainSeries = series[0]
  if (!mainSeries) return <span className="text-2xs text-slate-500">—</span>
  const thick = mainSeries.sliceThicknessMm
  const quality = thick <= 0.625 ? 'excellent' : thick <= 1.0 ? 'good' : thick <= 2.0 ? 'adequate' : 'poor'
  const config = {
    excellent: 'text-emerald-400 bg-emerald-950 border-emerald-800',
    good: 'text-cyan-400 bg-cyan-950 border-cyan-800',
    adequate: 'text-amber-400 bg-amber-950 border-amber-800',
    poor: 'text-red-400 bg-red-950 border-red-800',
  }[quality]
  return (
    <span className={`text-2xs font-semibold px-1.5 py-0.5 rounded border ${config}`}>
      {quality}
    </span>
  )
}

export default function StudiesPage() {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const pageSize = 20

  const { data, isLoading, error } = useQuery({
    queryKey: ['studies', page, pageSize],
    queryFn: () => studiesApi.list({ page, pageSize }),
    staleTime: 30_000,
  })

  if (isLoading) return <PageLoading label="Loading studies..." />
  if (error) return <ErrorState description="Failed to load studies" />

  const studies = data?.items ?? []
  const filtered = search
    ? studies.filter(s =>
        s.studyInstanceUid.toLowerCase().includes(search.toLowerCase()) ||
        s.studyDescription.toLowerCase().includes(search.toLowerCase())
      )
    : studies

  return (
    <div className="flex-1 flex flex-col p-6 animate-fade-in" data-testid="studies-page">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Scan size={20} className="text-cyan-400" />
          <h1 className="text-lg font-semibold text-slate-100">Imaging Studies</h1>
          <span className="text-sm text-slate-500 font-mono">{data?.total ?? 0} studies</span>
        </div>
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search studies..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input-base pl-8 w-64 text-sm"
            data-testid="studies-search"
          />
        </div>
      </div>

      <div className="flex-1 overflow-auto rounded-lg border border-slate-800">
        <table className="w-full text-sm" data-testid="studies-table">
          <thead className="bg-slate-800/50 sticky top-0">
            <tr>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Study UID</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Modality</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Date</th>
              <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Slices</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Patient</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Quality</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {filtered.map(study => {
              const mainSeries = study.series[0]
              const totalSlices = study.series.reduce((sum, s) => sum + s.sliceCount, 0)
              return (
                <tr key={study.id} className="hover:bg-slate-800/30 transition-colors" data-testid={`study-row-${study.id}`}>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-slate-300" title={study.studyInstanceUid}>
                      {study.studyInstanceUid.slice(0, 24)}...
                    </span>
                    <p className="text-2xs text-slate-500 mt-0.5">{study.studyDescription}</p>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs font-mono text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">
                      {mainSeries?.modality ?? '—'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                    {study.studyDate ? new Date(study.studyDate).toLocaleDateString() : '—'}
                  </td>
                  <td className="px-4 py-3 text-right text-xs font-mono text-slate-300">
                    {totalSlices}
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                    {study.patientId.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <QualityBadge series={study.series} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className="text-2xs font-semibold px-1.5 py-0.5 rounded border text-emerald-400 bg-emerald-950 border-emerald-800">
                      ingested
                    </span>
                  </td>
                </tr>
              )
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center text-sm text-slate-500">
                  No studies found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data && data.total > pageSize && (
        <div className="flex items-center justify-between mt-4">
          <span className="text-xs text-slate-500">
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, data.total)} of {data.total}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn-secondary p-1.5 disabled:opacity-30"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-xs text-slate-400 px-2 font-mono">Page {page}</span>
            <button
              onClick={() => setPage(p => p + 1)}
              disabled={!data.hasMore}
              className="btn-secondary p-1.5 disabled:opacity-30"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

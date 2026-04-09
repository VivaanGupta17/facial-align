import { Scan } from 'lucide-react'

export default function StudiesPage() {
  return (
    <div className="p-6 animate-fade-in" data-testid="studies-page">
      <div className="flex items-center gap-3 mb-6">
        <Scan size={20} className="text-cyan-400" />
        <h1 className="text-xl font-bold text-slate-100">Studies</h1>
      </div>
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-12 text-center">
        <Scan size={48} className="text-slate-600 mx-auto mb-4" />
        <h2 className="text-lg font-semibold text-slate-300 mb-2">DICOM Studies Browser</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto">
          Browse and manage uploaded DICOM studies, view series metadata, and link studies to surgical cases.
        </p>
        <p className="text-xs text-slate-600 mt-4">Coming soon</p>
      </div>
    </div>
  )
}

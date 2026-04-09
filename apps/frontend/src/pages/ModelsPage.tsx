import { BoxSelect } from 'lucide-react'

export default function ModelsPage() {
  return (
    <div className="p-6 animate-fade-in" data-testid="models-page">
      <div className="flex items-center gap-3 mb-6">
        <BoxSelect size={20} className="text-cyan-400" />
        <h1 className="text-xl font-bold text-slate-100">3D Models</h1>
      </div>
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-12 text-center">
        <BoxSelect size={48} className="text-slate-600 mx-auto mb-4" />
        <h2 className="text-lg font-semibold text-slate-300 mb-2">Segmented Model Gallery</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto">
          View and manage segmented 3D anatomical models. Preview meshes, compare versions, and export for surgical planning.
        </p>
        <p className="text-xs text-slate-600 mt-4">Coming soon</p>
      </div>
    </div>
  )
}

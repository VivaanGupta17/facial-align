import { useState } from 'react'
import {
  Download, FileBox, Loader2, CheckCircle, AlertTriangle,
  X, Package, Layers, Bone,
} from 'lucide-react'
import { exportApi, type ExportFileInfo, type ExportResponse } from '../../lib/api'

type ExportType = 'full_assembly' | 'individual_fragment' | 'corrected_mandible'
type StlFormat = 'binary' | 'ascii'

const EXPORT_TYPES: Array<{ id: ExportType; label: string; description: string; icon: React.ReactNode }> = [
  { id: 'full_assembly', label: 'Full Assembly', description: 'All fragments combined + individual files', icon: <Package size={16} /> },
  { id: 'individual_fragment', label: 'Individual Fragments', description: 'Separate STL per fragment', icon: <Layers size={16} /> },
  { id: 'corrected_mandible', label: 'Corrected Mandible', description: 'Mandible bone only', icon: <Bone size={16} /> },
]

interface ExportPanelProps {
  planId: string
  onClose: () => void
}

export default function ExportPanel({ planId, onClose }: ExportPanelProps) {
  const [exportType, setExportType] = useState<ExportType>('full_assembly')
  const [stlFormat, setStlFormat] = useState<StlFormat>('binary')
  const [isExporting, setIsExporting] = useState(false)
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null)

  const handleExport = async () => {
    setIsExporting(true)
    setError(null)
    setExportResult(null)
    try {
      const result = await exportApi.exportPlan(planId, exportType, stlFormat)
      setExportResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const handleDownload = async (file: ExportFileInfo) => {
    setDownloadingFile(file.filename)
    try {
      await exportApi.downloadStl(planId, file.filename)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
    } finally {
      setDownloadingFile(null)
    }
  }

  const handleDownloadAll = async () => {
    if (!exportResult) return
    for (const file of exportResult.files) {
      await handleDownload(file)
    }
  }

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg shadow-xl w-full max-w-lg" data-testid="export-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <FileBox size={18} className="text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-100">Export STL</h3>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-100 transition-colors"
          data-testid="export-panel-close"
        >
          <X size={16} />
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Export type selector */}
        <div>
          <label className="text-xs font-medium text-slate-400 mb-2 block">Export Type</label>
          <div className="space-y-1.5">
            {EXPORT_TYPES.map(t => (
              <button
                key={t.id}
                onClick={() => setExportType(t.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-md border text-left transition-colors ${
                  exportType === t.id
                    ? 'bg-cyan-950 border-cyan-800 text-cyan-400'
                    : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-slate-600'
                }`}
                data-testid={`export-type-${t.id}`}
              >
                {t.icon}
                <div>
                  <p className="text-xs font-medium">{t.label}</p>
                  <p className="text-2xs text-slate-500">{t.description}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Format toggle */}
        <div>
          <label className="text-xs font-medium text-slate-400 mb-2 block">STL Format</label>
          <div className="flex items-center bg-slate-900 rounded-md p-0.5 border border-slate-700">
            {(['binary', 'ascii'] as const).map(fmt => (
              <button
                key={fmt}
                onClick={() => setStlFormat(fmt)}
                className={`flex-1 px-3 py-1.5 rounded text-xs font-semibold transition-colors ${
                  stlFormat === fmt
                    ? 'bg-cyan-900 text-cyan-400'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                data-testid={`stl-format-${fmt}`}
              >
                {fmt === 'binary' ? 'Binary' : 'ASCII'}
              </button>
            ))}
          </div>
        </div>

        {/* Generate button */}
        <button
          onClick={handleExport}
          disabled={isExporting}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-md bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
          data-testid="generate-stl-btn"
        >
          {isExporting ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <FileBox size={16} />
              Generate STL
            </>
          )}
        </button>

        {/* Error display */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-950/50 border border-red-800 rounded-md text-xs text-red-300" data-testid="export-error">
            <AlertTriangle size={14} className="text-red-400 shrink-0" />
            {error}
          </div>
        )}

        {/* Results */}
        {exportResult && (
          <div className="space-y-3" data-testid="export-results">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                <CheckCircle size={14} />
                <span>Exported in {exportResult.totalExportTimeSeconds.toFixed(2)}s</span>
              </div>
              {exportResult.files.length > 1 && (
                <button
                  onClick={handleDownloadAll}
                  className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
                  data-testid="download-all-btn"
                >
                  <Download size={12} />
                  Download All
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {exportResult.files.map(file => (
                <div
                  key={file.filename}
                  className="flex items-center gap-3 p-2.5 bg-slate-900 border border-slate-700 rounded-md"
                  data-testid={`export-file-${file.filename}`}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-slate-200 truncate">{file.filename}</p>
                    <div className="flex items-center gap-3 mt-1 text-2xs text-slate-500">
                      <span>{file.vertexCount.toLocaleString()} verts</span>
                      <span>{file.faceCount.toLocaleString()} faces</span>
                      {file.volumeMm3 > 0 && (
                        <span>{file.volumeMm3.toFixed(1)} mm&sup3;</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      {file.isWatertight ? (
                        <span className="flex items-center gap-0.5 text-2xs text-emerald-400">
                          <CheckCircle size={10} /> Watertight
                        </span>
                      ) : (
                        <span className="flex items-center gap-0.5 text-2xs text-amber-400">
                          <AlertTriangle size={10} /> Not watertight
                        </span>
                      )}
                      {file.isPrintable ? (
                        <span className="flex items-center gap-0.5 text-2xs text-emerald-400">
                          <CheckCircle size={10} /> Printable
                        </span>
                      ) : (
                        <span className="flex items-center gap-0.5 text-2xs text-amber-400">
                          <AlertTriangle size={10} /> Not printable
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDownload(file)}
                    disabled={downloadingFile === file.filename}
                    className="shrink-0 flex items-center justify-center w-8 h-8 rounded-md bg-slate-800 border border-slate-600 text-slate-300 hover:text-cyan-400 hover:border-cyan-700 transition-colors disabled:opacity-50"
                    data-testid={`download-${file.filename}`}
                  >
                    {downloadingFile === file.filename ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Download size={14} />
                    )}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

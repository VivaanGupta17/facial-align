import {
  Box, Camera, ZoomIn, ZoomOut, Grid3X3, Download,
} from 'lucide-react'
import { useViewerStore } from '../../stores/viewerStore'
import type { ViewerState } from '../../types/medical'

interface ToolButtonProps {
  icon: React.ReactNode
  label: string
  active?: boolean
  onClick: () => void
  testId?: string
}

function ToolButton({ icon, label, active, onClick, testId }: ToolButtonProps) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={`flex items-center justify-center w-8 h-8 rounded-md transition-colors ${
        active
          ? 'bg-cyan-900 text-cyan-400 border border-cyan-700'
          : 'text-slate-400 hover:text-slate-100 hover:bg-slate-700'
      }`}
      data-testid={testId ?? `tool-${label.toLowerCase().replace(' ', '-')}`}
    >
      {icon}
    </button>
  )
}

const VIEW_MODES: Array<{ id: ViewerState['viewMode']; label: string }> = [
  { id: '3d', label: '3D' },
  { id: 'axial', label: 'Ax' },
  { id: 'coronal', label: 'Co' },
  { id: 'sagittal', label: 'Sa' },
]

interface ViewerToolbarProps {
  onScreenshot?: () => void
  onZoomFit?: () => void
  onZoomIn?: () => void
  onZoomOut?: () => void
  onExport?: () => void
}

export default function ViewerToolbar({
  onScreenshot,
  onZoomFit,
  onZoomIn,
  onZoomOut,
  onExport,
}: ViewerToolbarProps) {
  const { viewerState, setViewMode, toggleStructuresPanel, toggleGrid } = useViewerStore()
  const { viewMode, showStructuresPanel, showGrid } = viewerState

  return (
    <div
      className="flex flex-wrap items-center gap-2 border-b border-white/10 bg-[rgba(8,14,26,0.92)] px-3 py-2"
      data-testid="viewer-toolbar"
    >
      {/* View mode toggles */}
      <div className="flex items-center rounded-xl border border-white/10 bg-[rgba(15,23,42,0.76)] p-0.5" data-testid="view-mode-toggle">
        {VIEW_MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setViewMode(m.id)}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-colors ${
              viewMode === m.id
                ? 'bg-cyan-400 text-slate-950'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            data-testid={`view-mode-${m.id}`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="mx-1 h-5 w-px bg-white/10" />

      {/* View options */}
      <div className="flex items-center gap-0.5">
        <ToolButton
          icon={<Grid3X3 size={15} />}
          label="Toggle Grid"
          active={showGrid}
          onClick={toggleGrid}
          testId="tool-grid"
        />
        <ToolButton
          icon={<Box size={15} />}
          label="Structures Panel"
          active={showStructuresPanel}
          onClick={toggleStructuresPanel}
          testId="tool-structures-panel"
        />
      </div>

      <div className="mx-1 h-5 w-px bg-white/10" />

      {/* Actions */}
      <div className="flex items-center gap-0.5">
        <ToolButton icon={<ZoomOut size={15} />} label="Zoom Out" onClick={() => onZoomOut?.()} testId="tool-zoom-out" />
        <ToolButton icon={<ZoomIn size={15} />} label="Zoom In" onClick={() => onZoomIn?.()} testId="tool-zoom-in" />
        <ToolButton icon={<ZoomIn size={15} />} label="Zoom to Fit" onClick={() => onZoomFit?.()} testId="tool-zoom-fit" />
        <ToolButton icon={<Camera size={15} />} label="Screenshot" onClick={() => onScreenshot?.()} testId="tool-screenshot" />
        <ToolButton icon={<Download size={15} />} label="Export STL" onClick={() => onExport?.()} testId="tool-export-stl" />
      </div>

      <div className="ml-auto hidden items-center gap-2 rounded-full border border-white/10 bg-[rgba(15,23,42,0.72)] px-3 py-1 text-[11px] text-slate-500 lg:flex">
        <span>Rotate: drag</span>
        <span className="h-1 w-1 rounded-full bg-slate-700" />
        <span>Pan: right drag</span>
        <span className="h-1 w-1 rounded-full bg-slate-700" />
        <span>Zoom: scroll or +/-</span>
      </div>
    </div>
  )
}

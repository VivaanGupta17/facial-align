import {
  Box, Layers, Ruler, MessageSquare, Camera, ZoomIn, PanelRight,
  Grid3X3, Crosshair, RotateCcw,
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
}

export default function ViewerToolbar({ onScreenshot, onZoomFit }: ViewerToolbarProps) {
  const { viewerState, setViewMode, setActiveTool, toggleStructuresPanel, toggleGrid } = useViewerStore()
  const { viewMode, activeTool, showStructuresPanel, showGrid } = viewerState

  return (
    <div
      className="flex items-center gap-1 px-3 py-2 bg-slate-900 border-b border-slate-800"
      data-testid="viewer-toolbar"
    >
      {/* View mode toggles */}
      <div className="flex items-center bg-slate-800 rounded-md p-0.5 border border-slate-700" data-testid="view-mode-toggle">
        {VIEW_MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setViewMode(m.id)}
            className={`px-2.5 py-1 rounded text-xs font-semibold transition-colors ${
              viewMode === m.id
                ? 'bg-cyan-900 text-cyan-400'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            data-testid={`view-mode-${m.id}`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="w-px h-5 bg-slate-700 mx-1" />

      {/* Measurement tools */}
      <div className="flex items-center gap-0.5">
        <ToolButton
          icon={<Ruler size={15} />}
          label="Measure Distance"
          active={activeTool === 'measure_distance'}
          onClick={() => setActiveTool(activeTool === 'measure_distance' ? 'none' : 'measure_distance')}
          testId="tool-measure-distance"
        />
        <ToolButton
          icon={<Crosshair size={15} />}
          label="Measure Angle"
          active={activeTool === 'measure_angle'}
          onClick={() => setActiveTool(activeTool === 'measure_angle' ? 'none' : 'measure_angle')}
          testId="tool-measure-angle"
        />
        <ToolButton
          icon={<MessageSquare size={15} />}
          label="Annotate"
          active={activeTool === 'annotate'}
          onClick={() => setActiveTool(activeTool === 'annotate' ? 'none' : 'annotate')}
          testId="tool-annotate"
        />
      </div>

      <div className="w-px h-5 bg-slate-700 mx-1" />

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

      <div className="w-px h-5 bg-slate-700 mx-1" />

      {/* Actions */}
      <div className="flex items-center gap-0.5">
        <ToolButton icon={<ZoomIn size={15} />} label="Zoom to Fit" onClick={() => onZoomFit?.()} testId="tool-zoom-fit" />
        <ToolButton icon={<Camera size={15} />} label="Screenshot" onClick={() => onScreenshot?.()} testId="tool-screenshot" />
      </div>

      {/* Active tool indicator */}
      {activeTool !== 'none' && (
        <div className="ml-auto flex items-center gap-2 text-xs text-cyan-400 bg-cyan-950 border border-cyan-800 px-2 py-1 rounded">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          {activeTool.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} active
          <button onClick={() => setActiveTool('none')} className="text-slate-400 hover:text-slate-100 ml-1">×</button>
        </div>
      )}
    </div>
  )
}

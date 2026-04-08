/**
 * CrossSectionViewer — 2D CT cross-section viewer for Facial Align.
 *
 * Renders three orthogonal views (axial, coronal, sagittal) from a CT volume.
 * Since real volume data is streamed from the backend, this component:
 *  - Accepts an optional volumeUrl for a real NIfTI/raw volume
 *  - Renders a simulated gradient canvas when no real volume is loaded
 *  - Supports window/level controls (bone / soft tissue presets)
 *  - Overlays colour-coded segmentation masks per structure
 *  - Has slice navigation sliders
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, Layers, Sliders } from 'lucide-react'
import type { SegmentedStructure } from '../../types/medical'
import type { SlicePlane } from '../../lib/geometry'

// =============================================================================
// Window/Level presets for bone and soft tissue
// =============================================================================

interface WindowPreset {
  label: string
  center: number   // HU
  width: number    // HU
}

const WINDOW_PRESETS: WindowPreset[] = [
  { label: 'Bone',        center: 400,  width: 1500 },
  { label: 'Soft Tissue', center: 50,   width: 350  },
  { label: 'Brain',       center: 40,   width: 80   },
  { label: 'Lung',        center: -600, width: 1500 },
]

// =============================================================================
// Props
// =============================================================================

interface CrossSectionViewerProps {
  /** Structures to overlay (uses color + label for mask colouring) */
  structures?: SegmentedStructure[]
  /** Backend URL for real volume data (future) */
  volumeUrl?: string
  /** Total slice count per plane (approximation if real volume unavailable) */
  sliceCount?: number
  /** Initial active plane */
  defaultPlane?: SlicePlane
  className?: string
}

// =============================================================================
// Per-plane viewer state
// =============================================================================

interface PlaneState {
  sliceIndex: number
  totalSlices: number
}

const PLANE_LABELS: Record<SlicePlane, string> = {
  axial:    'Axial (Transverse)',
  coronal:  'Coronal (Frontal)',
  sagittal: 'Sagittal (Lateral)',
}

// =============================================================================
// Main component
// =============================================================================

export default function CrossSectionViewer({
  structures = [],
  sliceCount = 200,
  defaultPlane = 'axial',
  className = '',
}: CrossSectionViewerProps) {
  const [activePlane, setActivePlane] = useState<SlicePlane>(defaultPlane)
  const [planeStates, setPlaneStates] = useState<Record<SlicePlane, PlaneState>>({
    axial:    { sliceIndex: Math.floor(sliceCount / 2), totalSlices: sliceCount },
    coronal:  { sliceIndex: Math.floor(sliceCount / 2), totalSlices: sliceCount },
    sagittal: { sliceIndex: Math.floor(sliceCount / 2), totalSlices: sliceCount },
  })
  const [windowPresetIdx, setWindowPresetIdx] = useState(0)
  const [customCenter, setCustomCenter] = useState(WINDOW_PRESETS[0].center)
  const [customWidth, setCustomWidth]   = useState(WINDOW_PRESETS[0].width)
  const [showOverlay, setShowOverlay] = useState(true)
  const [showControls, setShowControls] = useState(false)

  const preset = WINDOW_PRESETS[windowPresetIdx]
  const wCenter = windowPresetIdx < WINDOW_PRESETS.length - 1 ? preset.center : customCenter
  const wWidth  = windowPresetIdx < WINDOW_PRESETS.length - 1 ? preset.width  : customWidth

  const currentState = planeStates[activePlane]

  const setSliceIndex = useCallback((plane: SlicePlane, idx: number) => {
    setPlaneStates(prev => ({
      ...prev,
      [plane]: { ...prev[plane], sliceIndex: Math.max(0, Math.min(prev[plane].totalSlices - 1, idx)) },
    }))
  }, [])

  const handlePresetChange = (idx: number) => {
    setWindowPresetIdx(idx)
    setCustomCenter(WINDOW_PRESETS[idx]?.center ?? customCenter)
    setCustomWidth(WINDOW_PRESETS[idx]?.width ?? customWidth)
  }

  return (
    <div className={`flex flex-col bg-slate-950 h-full ${className}`} data-testid="cross-section-viewer">
      {/* Plane tabs */}
      <div className="flex items-center gap-0.5 px-2 pt-2 pb-0 border-b border-slate-800">
        {(['axial', 'coronal', 'sagittal'] as SlicePlane[]).map(plane => (
          <button
            key={plane}
            onClick={() => setActivePlane(plane)}
            className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors capitalize ${
              activePlane === plane
                ? 'bg-slate-800 text-slate-100 border border-b-0 border-slate-700'
                : 'text-slate-500 hover:text-slate-300'
            }`}
            data-testid={`plane-tab-${plane}`}
          >
            {plane}
          </button>
        ))}

        {/* Right-side controls */}
        <div className="ml-auto flex items-center gap-1 pb-1">
          <button
            onClick={() => setShowOverlay(v => !v)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showOverlay ? 'text-cyan-400 bg-cyan-950' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle segmentation overlay"
            data-testid="toggle-overlay"
          >
            <Layers size={12} />
          </button>
          <button
            onClick={() => setShowControls(v => !v)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
              showControls ? 'text-amber-400 bg-amber-950' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="Window/level controls"
            data-testid="toggle-wl-controls"
          >
            <Sliders size={12} />
          </button>
        </div>
      </div>

      {/* Window/level controls panel */}
      {showControls && (
        <WindowLevelPanel
          presets={WINDOW_PRESETS}
          activePresetIdx={windowPresetIdx}
          center={customCenter}
          width={customWidth}
          onPresetChange={handlePresetChange}
          onCenterChange={setCustomCenter}
          onWidthChange={setCustomWidth}
        />
      )}

      {/* Viewer canvas area */}
      <div className="flex-1 relative min-h-0 flex flex-col">
        {/* Plane label */}
        <div className="absolute top-2 left-2 z-10 bg-slate-900/80 text-cyan-400 text-xs font-mono px-2 py-1 rounded border border-slate-700">
          {PLANE_LABELS[activePlane]}
        </div>

        {/* Slice position */}
        <div className="absolute top-2 right-2 z-10 bg-slate-900/80 text-slate-400 text-xs font-mono px-2 py-1 rounded border border-slate-700">
          {currentState.sliceIndex + 1} / {currentState.totalSlices}
        </div>

        {/* Canvas */}
        <div className="flex-1 flex items-center justify-center min-h-0">
          <SliceCanvas
            plane={activePlane}
            sliceIndex={currentState.sliceIndex}
            totalSlices={currentState.totalSlices}
            windowCenter={wCenter}
            windowWidth={wWidth}
            structures={structures}
            showOverlay={showOverlay}
          />
        </div>

        {/* Slice navigation arrows */}
        <button
          onClick={() => setSliceIndex(activePlane, currentState.sliceIndex - 1)}
          disabled={currentState.sliceIndex === 0}
          className="absolute left-2 top-1/2 -translate-y-1/2 z-10 w-7 h-7 bg-slate-800/80 hover:bg-slate-700 border border-slate-700 rounded flex items-center justify-center text-slate-400 disabled:opacity-30 transition-colors"
          data-testid="prev-slice"
        >
          <ChevronLeft size={14} />
        </button>
        <button
          onClick={() => setSliceIndex(activePlane, currentState.sliceIndex + 1)}
          disabled={currentState.sliceIndex >= currentState.totalSlices - 1}
          className="absolute right-2 top-1/2 -translate-y-1/2 z-10 w-7 h-7 bg-slate-800/80 hover:bg-slate-700 border border-slate-700 rounded flex items-center justify-center text-slate-400 disabled:opacity-30 transition-colors"
          data-testid="next-slice"
        >
          <ChevronRight size={14} />
        </button>
      </div>

      {/* Slice slider */}
      <div className="px-4 py-2 border-t border-slate-800 bg-slate-900 flex items-center gap-3">
        <span className="text-2xs text-slate-600 font-mono w-4">1</span>
        <input
          type="range"
          min={0}
          max={currentState.totalSlices - 1}
          value={currentState.sliceIndex}
          onChange={e => setSliceIndex(activePlane, parseInt(e.target.value, 10))}
          className="flex-1 h-1.5 appearance-none bg-slate-700 rounded-full accent-cyan-500"
          data-testid="slice-slider"
        />
        <span className="text-2xs text-slate-600 font-mono w-6 text-right">{currentState.totalSlices}</span>
      </div>

      {/* Crosshair position info */}
      <div className="px-4 py-1.5 border-t border-slate-800 bg-slate-900 flex items-center gap-4 text-2xs font-mono text-slate-600">
        <span>W: {wWidth}</span>
        <span>L: {wCenter}</span>
        <span className="ml-auto capitalize">{WINDOW_PRESETS[windowPresetIdx]?.label ?? 'Custom'}</span>
      </div>
    </div>
  )
}

// =============================================================================
// Slice canvas — renders a synthetic CT-like image
// =============================================================================

interface SliceCanvasProps {
  plane: SlicePlane
  sliceIndex: number
  totalSlices: number
  windowCenter: number
  windowWidth: number
  structures: SegmentedStructure[]
  showOverlay: boolean
}

function SliceCanvas({
  plane,
  sliceIndex,
  totalSlices,
  windowCenter,
  windowWidth,
  structures,
  showOverlay,
}: SliceCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const W = 400, H = 400

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Render a synthetic CT slice
    drawSyntheticSlice(ctx, W, H, plane, sliceIndex, totalSlices, windowCenter, windowWidth)

    if (showOverlay && structures.length > 0) {
      drawSegmentationOverlay(ctx, W, H, plane, sliceIndex, totalSlices, structures)
    }
  }, [plane, sliceIndex, totalSlices, windowCenter, windowWidth, structures, showOverlay])

  return (
    <canvas
      ref={canvasRef}
      width={W}
      height={H}
      className="max-w-full max-h-full object-contain"
      style={{ imageRendering: 'pixelated' }}
      data-testid="slice-canvas"
    />
  )
}

// =============================================================================
// Synthetic CT slice renderer (placeholder until real volume available)
// =============================================================================

function drawSyntheticSlice(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  plane: SlicePlane,
  sliceIndex: number,
  totalSlices: number,
  wCenter: number,
  wWidth: number,
) {
  const t = sliceIndex / Math.max(1, totalSlices - 1)
  const imageData = ctx.createImageData(W, H)
  const data = imageData.data

  // Window/level normalization: map HU [center-width/2, center+width/2] → [0,255]
  const huMin = wCenter - wWidth / 2
  const huMax = wCenter + wWidth / 2

  const normalize = (hu: number) => Math.max(0, Math.min(255, Math.round(((hu - huMin) / (huMax - huMin)) * 255)))

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const nx = px / W
      const ny = py / H
      const cx = nx - 0.5
      const cy = ny - 0.5
      const r = Math.sqrt(cx * cx + cy * cy)

      // Simulate skull cross-section profile
      let hu: number

      if (plane === 'axial') {
        // Head oval
        const inHead = (cx * cx) / (0.36) + (cy * cy) / (0.30) < 1
        const inSkull = (cx * cx) / (0.30) + (cy * cy) / (0.24) < 1
        const inBrain = (cx * cx) / (0.20) + (cy * cy) / (0.16) < 1
        const sliceT = Math.abs(t - 0.5) * 2 // 0=center, 1=edge

        if (!inHead) {
          hu = -1000 // air
        } else if (!inSkull) {
          hu = 700 + Math.sin(nx * 20 + ny * 15) * 50 // cortical bone
        } else if (!inBrain) {
          hu = 40 + Math.sin(nx * 30 + ny * 25 + sliceT * 5) * 20 // soft tissue
        } else {
          hu = 25 + Math.cos(nx * 50 + ny * 40) * 10 // brain
        }
      } else if (plane === 'coronal') {
        const inHead = (cx * cx) / (0.20) + (cy * cy) / (0.36) < 1
        const inSkull = (cx * cx) / (0.16) + (cy * cy) / (0.30) < 1
        const inSoft = (cx * cx) / (0.10) + (cy * cy) / (0.24) < 1

        if (!inHead) hu = -1000
        else if (!inSkull) hu = 700 + Math.sin(nx * 18 + ny * 12 + t * 3) * 60
        else if (!inSoft) hu = 45 + Math.sin(px * 0.1 + py * 0.08) * 15
        else hu = 30 + Math.cos(px * 0.2 + py * 0.15) * 8
      } else {
        // Sagittal
        const inHead = (cx * cx) / (0.28) + (cy * cy) / (0.36) < 1
        const inSkull = (cx * cx) / (0.22) + (cy * cy) / (0.30) < 1
        const inSoft = (cx * cx) / (0.14) + (cy * cy) / (0.24) < 1

        if (!inHead) hu = -1000
        else if (!inSkull) hu = 650 + Math.sin(nx * 22 + ny * 18 + t * 4) * 80
        else if (!inSoft) hu = 50 + Math.sin(px * 0.09 + py * 0.11) * 18
        else hu = 28 + Math.cos(px * 0.18 + py * 0.14) * 10
      }

      // Add subtle noise
      hu += (Math.random() - 0.5) * 8
      const v = normalize(hu)

      const idx = (py * W + px) * 4
      data[idx]     = v
      data[idx + 1] = v
      data[idx + 2] = v
      data[idx + 3] = 255
    }
  }

  ctx.putImageData(imageData, 0, 0)
}

// =============================================================================
// Segmentation overlay renderer
// =============================================================================

function drawSegmentationOverlay(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  plane: SlicePlane,
  sliceIndex: number,
  totalSlices: number,
  structures: SegmentedStructure[],
) {
  const t = sliceIndex / Math.max(1, totalSlices - 1)

  ctx.globalAlpha = 0.35
  ctx.globalCompositeOperation = 'source-over'

  for (const s of structures) {
    if (s.status === 'rejected') continue

    const color = hexToRgb(s.color)
    if (!color) continue

    // Compute a simple elliptical mask from the structure's bounding box
    const bb = s.boundingBox
    // Map world coords to canvas coords (rough normalised projection)
    const worldScale = 200 // mm half-width for the craniofacial region

    let centerX: number, centerY: number, rx: number, ry: number

    if (plane === 'axial') {
      centerX = ((bb.center.x + worldScale) / (2 * worldScale)) * W
      centerY = ((bb.center.z + worldScale) / (2 * worldScale)) * H
      rx = Math.abs(bb.max.x - bb.min.x) / (2 * worldScale) * W * 0.4
      ry = Math.abs(bb.max.z - bb.min.z) / (2 * worldScale) * H * 0.4
      // Only show near this slice (Y in world = depth)
      const structureSliceT = (bb.center.y + worldScale) / (2 * worldScale)
      if (Math.abs(t - structureSliceT) > 0.15) continue
    } else if (plane === 'coronal') {
      centerX = ((bb.center.x + worldScale) / (2 * worldScale)) * W
      centerY = ((bb.center.y + worldScale) / (2 * worldScale)) * H
      rx = Math.abs(bb.max.x - bb.min.x) / (2 * worldScale) * W * 0.4
      ry = Math.abs(bb.max.y - bb.min.y) / (2 * worldScale) * H * 0.4
      const structureSliceT = (bb.center.z + worldScale) / (2 * worldScale)
      if (Math.abs(t - structureSliceT) > 0.15) continue
    } else {
      centerX = ((bb.center.z + worldScale) / (2 * worldScale)) * W
      centerY = ((bb.center.y + worldScale) / (2 * worldScale)) * H
      rx = Math.abs(bb.max.z - bb.min.z) / (2 * worldScale) * W * 0.4
      ry = Math.abs(bb.max.y - bb.min.y) / (2 * worldScale) * H * 0.4
      const structureSliceT = (bb.center.x + worldScale) / (2 * worldScale)
      if (Math.abs(t - structureSliceT) > 0.15) continue
    }

    ctx.fillStyle = `rgba(${color.r},${color.g},${color.b},1)`
    ctx.beginPath()
    ctx.ellipse(centerX, centerY, Math.max(5, rx), Math.max(5, ry), 0, 0, Math.PI * 2)
    ctx.fill()

    // Structure label
    ctx.globalAlpha = 0.85
    ctx.fillStyle = `rgb(${color.r},${color.g},${color.b})`
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(s.label.slice(0, 10), centerX, centerY - (Math.max(5, ry)) - 3)
    ctx.globalAlpha = 0.35
  }

  ctx.globalAlpha = 1
  ctx.globalCompositeOperation = 'source-over'
}

// =============================================================================
// Window / Level control panel
// =============================================================================

interface WindowLevelPanelProps {
  presets: WindowPreset[]
  activePresetIdx: number
  center: number
  width: number
  onPresetChange: (idx: number) => void
  onCenterChange: (v: number) => void
  onWidthChange: (v: number) => void
}

function WindowLevelPanel({
  presets,
  activePresetIdx,
  center,
  width,
  onPresetChange,
  onCenterChange,
  onWidthChange,
}: WindowLevelPanelProps) {
  return (
    <div className="bg-slate-900 border-b border-slate-800 px-4 py-3 space-y-3" data-testid="wl-panel">
      {/* Preset buttons */}
      <div className="flex flex-wrap gap-1">
        {presets.map((p, i) => (
          <button
            key={p.label}
            onClick={() => onPresetChange(i)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              activePresetIdx === i
                ? 'bg-cyan-900 text-cyan-300 border border-cyan-700'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
            }`}
            data-testid={`wl-preset-${p.label.toLowerCase().replace(' ', '-')}`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Custom sliders */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <span className="text-2xs text-slate-500 w-20">Center (L)</span>
          <input
            type="range" min={-1000} max={1000} step={10}
            value={center}
            onChange={e => onCenterChange(parseInt(e.target.value, 10))}
            className="flex-1 h-1.5 accent-cyan-500"
            data-testid="wl-center"
          />
          <span className="text-2xs font-mono text-slate-400 w-10 text-right">{center}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-2xs text-slate-500 w-20">Width (W)</span>
          <input
            type="range" min={1} max={4000} step={10}
            value={width}
            onChange={e => onWidthChange(parseInt(e.target.value, 10))}
            className="flex-1 h-1.5 accent-amber-500"
            data-testid="wl-width"
          />
          <span className="text-2xs font-mono text-slate-400 w-10 text-right">{width}</span>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Utility
// =============================================================================

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) }
    : null
}

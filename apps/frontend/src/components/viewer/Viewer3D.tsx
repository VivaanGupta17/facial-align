import { useRef, Suspense, useCallback } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Grid, GizmoHelper, GizmoViewport, Environment, Stats } from '@react-three/drei'
import * as THREE from 'three'
import { useViewerStore, CAMERA_PRESETS } from '../../stores/viewerStore'
import { usePlanningStore } from '../../stores/planningStore'
import AnatomyMesh from './AnatomyMesh'
import ViewerToolbar from './ViewerToolbar'
import type { SegmentedStructure, FragmentTransform, StructureLabel } from '../../types/medical'

// ---------------------------
// Structures panel (sidebar)
// ---------------------------
function StructuresPanel({ structures }: { structures: SegmentedStructure[] }) {
  const { viewerState, setStructureVisible, setStructureOpacity, setStructureWireframe, setSelectedFragment } = useViewerStore()
  const { selectedFragmentId, selectFragment } = usePlanningStore()

  return (
    <div className="w-56 bg-slate-900 border-l border-slate-800 flex flex-col" data-testid="structures-panel">
      <div className="panel-header py-2.5">
        <span className="text-xs font-semibold text-slate-300">Structures</span>
        <span className="text-2xs font-mono text-slate-500">{structures.length}</span>
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {structures.map(s => {
          const vis = viewerState.structureVisibility[s.label]
          if (!vis) return null
          return (
            <div
              key={s.label}
              className={`px-3 py-2 border-b border-slate-800 last:border-b-0 cursor-pointer transition-colors ${
                selectedFragmentId === s.label ? 'bg-cyan-950/30 border-l-2 border-l-cyan-500' : 'hover:bg-slate-800/50'
              }`}
              onClick={() => { setSelectedFragment(s.label); selectFragment(s.label) }}
              data-testid={`structure-row-${s.label}`}
            >
              <div className="flex items-center gap-2 mb-1.5">
                {/* Visibility checkbox */}
                <input
                  type="checkbox"
                  checked={vis.visible}
                  onChange={e => { e.stopPropagation(); setStructureVisible(s.label, e.target.checked) }}
                  className="rounded border-slate-600 bg-slate-800"
                  data-testid={`vis-toggle-${s.label}`}
                />
                {/* Color swatch */}
                <span className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: s.color }} />
                {/* Name */}
                <span className={`text-xs flex-1 truncate ${selectedFragmentId === s.label ? 'text-cyan-300 font-semibold' : 'text-slate-300'}`} title={s.displayName}>{s.displayName}</span>
                {selectedFragmentId === s.label && (
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                )}
              </div>
              {/* Opacity slider */}
              <div className="flex items-center gap-2 pl-5">
                <span className="text-2xs text-slate-600">Opacity</span>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={vis.opacity}
                  onChange={e => setStructureOpacity(s.label, parseFloat(e.target.value))}
                  className="flex-1 h-0.5"
                  data-testid={`opacity-${s.label}`}
                />
                <span className="text-2xs font-mono text-slate-500 w-6">{Math.round(vis.opacity * 100)}%</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------
// Camera preset buttons
// ---------------------------
function CameraPresetButtons() {
  const { setCameraPreset, viewerState } = useViewerStore()

  return (
    <div className="absolute bottom-4 left-4 flex flex-col gap-1 z-10" data-testid="camera-presets">
      <p className="text-2xs text-slate-600 mb-0.5 font-mono">VIEW</p>
      {CAMERA_PRESETS.map(p => (
        <button
          key={p.name}
          onClick={() => setCameraPreset(p.name)}
          className={`px-2 py-1 rounded text-2xs font-mono font-semibold transition-colors ${
            viewerState.cameraPreset === p.name
              ? 'bg-cyan-900 text-cyan-300 border border-cyan-700'
              : 'bg-slate-800/80 text-slate-400 hover:text-slate-200 border border-slate-700'
          }`}
          data-testid={`preset-${p.name}`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}

// ---------------------------
// Measurement overlay
// ---------------------------
function MeasurementOverlay() {
  const { viewerState, removeMeasurement } = useViewerStore()
  if (!viewerState.measurements.length || !viewerState.showMeasurements) return null

  return (
    <div className="absolute top-2 right-2 z-10 space-y-1" data-testid="measurements-overlay">
      {viewerState.measurements.map(m => (
        <div
          key={m.id}
          className="flex items-center gap-2 bg-slate-900/90 border border-slate-700 rounded px-2 py-1 text-xs"
          data-testid={`measurement-${m.id}`}
        >
          <span className="font-mono text-cyan-400">{m.value.toFixed(2)} {m.unit}</span>
          <span className="text-slate-400">{m.label}</span>
          <button onClick={() => removeMeasurement(m.id)} className="text-slate-600 hover:text-red-400">×</button>
        </div>
      ))}
    </div>
  )
}

// ---------------------------
// 3D scene content
// ---------------------------
function SceneContent({
  structures,
  fragments: fragmentsProp,
}: {
  structures: SegmentedStructure[]
  fragments?: FragmentTransform[]
}) {
  const { viewerState, setSelectedFragment } = useViewerStore()
  const { selectFragment, currentPlan, selectedFragmentId: planSelectedId } = usePlanningStore()

  // Use prop fragments if provided, otherwise read reactively from planningStore
  const fragments = fragmentsProp ?? currentPlan?.fragmentTransforms

  const handleSelectFragment = useCallback((fragmentId: string) => {
    setSelectedFragment(fragmentId)
    selectFragment(fragmentId)
  }, [setSelectedFragment, selectFragment])

  // Map structure labels to shape variants
  const shapeMap: Partial<Record<StructureLabel, 'mandible' | 'maxilla' | 'zygoma' | 'orbit' | 'teeth' | 'bone' | 'fragment'>> = {
    mandible: 'mandible',
    maxilla: 'maxilla',
    zygoma_left: 'zygoma',
    zygoma_right: 'zygoma',
    orbit_left: 'orbit',
    orbit_right: 'orbit',
    teeth_upper: 'teeth',
    teeth_lower: 'teeth',
  }

  // Position each structure in a meaningful arrangement
  const positionMap: Partial<Record<StructureLabel, [number, number, number]>> = {
    mandible: [0, -3, 0],
    maxilla: [0, 0.5, 0],
    zygoma_left: [-4, 1.5, 0],
    zygoma_right: [4, 1.5, 0],
    orbit_left: [-3.5, 3.5, 0],
    orbit_right: [3.5, 3.5, 0],
    frontal_bone: [0, 6, 0],
    teeth_upper: [0, -0.5, 1],
    teeth_lower: [0, -2.5, 1],
    nasal_bones: [0, 2.5, 1.5],
  }

  return (
    <>
      {/* Lighting for bone visualization */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 20, 10]} intensity={0.8} castShadow />
      <directionalLight position={[-10, -5, -5]} intensity={0.3} color="#b0c4de" />
      <pointLight position={[0, 10, 20]} intensity={0.5} color="#cffafe" />

      {/* Grid */}
      {viewerState.showGrid && (
        <Grid
          args={[30, 30]}
          position={[0, -8, 0]}
          cellColor="#334155"
          sectionColor="#475569"
          cellSize={1}
          sectionSize={5}
          fadeDistance={30}
          infiniteGrid={false}
        />
      )}

      {/* Anatomical structures */}
      {structures.map(structure => {
        const vis = viewerState.structureVisibility[structure.label]
        if (!vis) return null
        return (
          <AnatomyMesh
            key={structure.label}
            label={structure.label}
            color={vis.color || structure.color}
            opacity={vis.opacity}
            wireframe={vis.wireframe}
            selected={vis.selected}
            visible={vis.visible}
            position={positionMap[structure.label] ?? [0, 0, 0]}
            shapeVariant={shapeMap[structure.label] ?? 'bone'}
            meshUri={structure.meshUri || undefined}
            onClick={() => handleSelectFragment(structure.label)}
          />
        )
      })}

      {/* Fragment transforms (from planning) */}
      {fragments?.map(frag => {
        const vis = viewerState.structureVisibility[frag.structureLabel]
        const t = frag.currentTransform
        const basePos = positionMap[frag.structureLabel] ?? [0, 0, 0]
        return (
          <AnatomyMesh
            key={frag.fragmentId}
            label={frag.structureLabel}
            color={vis?.color ?? '#ef4444'}
            opacity={vis?.opacity ?? 0.85}
            wireframe={vis?.wireframe ?? false}
            selected={viewerState.selectedFragmentId === frag.fragmentId}
            visible={vis?.visible ?? true}
            position={[
              basePos[0] + t.translation.x,
              basePos[1] + t.translation.y,
              basePos[2] + t.translation.z,
            ]}
            rotation={[t.rotation.x, t.rotation.y, t.rotation.z]}
            shapeVariant="fragment"
            onClick={() => handleSelectFragment(frag.fragmentId)}
          />
        )
      })}

      {/* Camera orbit controls */}
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        makeDefault
        minDistance={5}
        maxDistance={80}
        panSpeed={0.8}
        rotateSpeed={0.6}
        zoomSpeed={0.8}
        target={[0, 0, 0]}
      />

      {/* View helper */}
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
      </GizmoHelper>
    </>
  )
}

// ---------------------------
// Main Viewer3D component
// ---------------------------
interface Viewer3DProps {
  structures: SegmentedStructure[]
  fragments?: FragmentTransform[]
  height?: string
  showStructuresPanel?: boolean
  className?: string
}

export default function Viewer3D({
  structures,
  fragments,
  height = '100%',
  className = '',
}: Viewer3DProps) {
  const { viewerState } = useViewerStore()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleScreenshot = () => {
    const canvas = document.querySelector('canvas.three-canvas') as HTMLCanvasElement
    if (!canvas) return
    const link = document.createElement('a')
    link.download = `facial-align-${Date.now()}.png`
    link.href = canvas.toDataURL()
    link.click()
  }

  return (
    <div className={`flex flex-col bg-slate-900 ${className}`} style={{ height }} data-testid="viewer-3d">
      {/* Toolbar */}
      <ViewerToolbar onScreenshot={handleScreenshot} />

      {/* Main viewer area */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas */}
        <div className="relative flex-1 bg-slate-950">
          <Canvas
            className="three-canvas"
            camera={{ position: [0, 5, 25], fov: 45, near: 0.1, far: 1000 }}
            gl={{
              antialias: true,
              alpha: false,
              preserveDrawingBuffer: true, // needed for screenshots
            }}
            shadows
            dpr={[1, 2]}
            onCreated={({ gl }) => {
              gl.setClearColor(new THREE.Color('#020617'))
            }}
          >
            <Suspense fallback={null}>
              <SceneContent structures={structures} fragments={fragments} />
            </Suspense>
          </Canvas>

          {/* Overlays */}
          <CameraPresetButtons />
          <MeasurementOverlay />

          {/* View mode label */}
          {viewerState.viewMode !== '3d' && (
            <div className="absolute top-2 left-2 bg-slate-900/90 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-cyan-400">
              {viewerState.viewMode.toUpperCase()} VIEW
            </div>
          )}

          {/* Empty state */}
          {structures.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <div className="text-slate-600 mb-3">
                <svg viewBox="0 0 60 60" className="w-16 h-16 mx-auto" fill="none">
                  <circle cx="30" cy="30" r="28" stroke="#334155" strokeWidth="1.5" />
                  <path d="M20 25 Q30 18, 40 25" stroke="#475569" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                  <path d="M18 35 Q22 42, 30 44 Q38 42, 42 35" stroke="#475569" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                </svg>
              </div>
              <p className="text-sm text-slate-500 font-medium">No structures loaded</p>
              <p className="text-xs text-slate-600 mt-1">Run segmentation to populate the 3D viewer</p>
            </div>
          )}
        </div>

        {/* Structures panel */}
        {viewerState.showStructuresPanel && structures.length > 0 && (
          <StructuresPanel structures={structures} />
        )}
      </div>
    </div>
  )
}

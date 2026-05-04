import { useRef, Suspense, useCallback, useEffect, useMemo, useState, type RefObject } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Grid, GizmoHelper, GizmoViewport } from '@react-three/drei'
import * as THREE from 'three'
import { useViewerStore, CAMERA_PRESETS } from '../../stores/viewerStore'
import { usePlanningStore } from '../../stores/planningStore'
import AnatomyMesh from './AnatomyMesh'
import ViewerToolbar from './ViewerToolbar'
import type { SegmentedStructure, FragmentTransform, StructureLabel } from '../../types/medical'

const VIEW_MODE_PRESET_MAP = {
  axial: 'superior',
  coronal: 'anterior',
  sagittal: 'lateral_r',
} as const

// ---------------------------
// Structures panel (sidebar)
// ---------------------------
function StructuresPanel({ structures }: { structures: SegmentedStructure[] }) {
  const {
    viewerState,
    setStructureVisible,
    setStructureOpacity,
    setStructureWireframe,
    setSelectedFragment,
    isolateStructure,
    showAllStructures,
  } = useViewerStore()
  const { selectFragment } = usePlanningStore()
  const selectedStructureId = structures.some((structure) => structure.label === viewerState.selectedFragmentId)
    ? viewerState.selectedFragmentId
    : null

  return (
    <div className="flex w-64 flex-col border-l border-white/10 bg-[rgba(8,14,26,0.92)] backdrop-blur-xl" data-testid="structures-panel">
      <div className="panel-header py-2.5">
        <span className="text-xs font-semibold text-slate-300">Structures</span>
        <span className="text-2xs font-mono text-slate-500">{structures.length}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 border-b border-white/10 px-3 py-2">
        <button
          onClick={showAllStructures}
          className="rounded-xl border border-white/10 bg-[rgba(15,23,42,0.7)] px-2 py-1.5 text-xs text-slate-300 transition-colors hover:bg-white/5"
          data-testid="show-all-structures"
        >
          Show All
        </button>
        <button
          onClick={() => selectedStructureId && isolateStructure(selectedStructureId as StructureLabel)}
          disabled={!selectedStructureId}
          className="rounded-xl border border-white/10 bg-[rgba(15,23,42,0.7)] px-2 py-1.5 text-xs text-slate-300 transition-colors hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
          data-testid="isolate-selected-structure"
        >
          Isolate
        </button>
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {structures.map(s => {
          const vis = viewerState.structureVisibility[s.label]
          if (!vis) return null
          return (
            <div
              key={s.label}
              className={`border-b border-white/5 px-3 py-2 last:border-b-0 cursor-pointer transition-colors ${
                selectedStructureId === s.label ? 'bg-cyan-950/30 border-l-2 border-l-cyan-500' : 'hover:bg-slate-800/50'
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
                <span className={`text-xs flex-1 truncate ${selectedStructureId === s.label ? 'text-cyan-300 font-semibold' : 'text-slate-300'}`} title={s.displayName}>{s.displayName}</span>
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    setStructureWireframe(s.label, !vis.wireframe)
                  }}
                  className={`rounded px-1 py-0.5 text-[10px] font-semibold transition-colors ${
                    vis.wireframe ? 'bg-cyan-900 text-cyan-300' : 'bg-slate-800 text-slate-500 hover:text-slate-300'
                  }`}
                  data-testid={`wireframe-toggle-${s.label}`}
                >
                  WF
                </button>
                {selectedStructureId === s.label && (
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

function computeFitDistance(box: THREE.Box3, camera: THREE.PerspectiveCamera) {
  const size = box.getSize(new THREE.Vector3())
  const maxDim = Math.max(size.x, size.y, size.z, 1)
  const fov = THREE.MathUtils.degToRad(camera.fov)
  return Math.max((maxDim / (2 * Math.tan(fov / 2))) * 1.4, 16)
}

function SceneCameraController({
  contentRef,
  controlsRef,
  viewMode,
  cameraPreset,
  zoomFitNonce,
  zoomInNonce,
  zoomOutNonce,
}: {
  contentRef: RefObject<THREE.Group>
  controlsRef: RefObject<any>
  viewMode: '3d' | 'axial' | 'coronal' | 'sagittal'
  cameraPreset: string
  zoomFitNonce: number
  zoomInNonce: number
  zoomOutNonce: number
}) {
  const { camera } = useThree()
  const perspectiveCamera = camera as THREE.PerspectiveCamera

  const getBounds = useCallback(() => {
    const group = contentRef.current
    if (!group) return null
    const box = new THREE.Box3().setFromObject(group)
    return box.isEmpty() ? null : box
  }, [contentRef])

  const applyPreset = useCallback((presetName: string) => {
    const preset = CAMERA_PRESETS.find((item) => item.name === presetName) ?? CAMERA_PRESETS[0]
    const bounds = getBounds()
    const center = bounds?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3(0, 0, 0)
    const distance = bounds ? computeFitDistance(bounds, perspectiveCamera) : 26
    const direction = new THREE.Vector3(preset.position.x, preset.position.y, preset.position.z).normalize()

    perspectiveCamera.position.copy(center.clone().add(direction.multiplyScalar(distance)))
    perspectiveCamera.up.set(preset.up.x, preset.up.y, preset.up.z)
    perspectiveCamera.lookAt(center)
    controlsRef.current?.target.copy(center)
    controlsRef.current?.update?.()
  }, [controlsRef, getBounds, perspectiveCamera])

  const zoomByFactor = useCallback((factor: number) => {
    const controls = controlsRef.current
    if (!controls) return
    const target = controls.target.clone()
    const direction = perspectiveCamera.position.clone().sub(target)
    const nextDistance = THREE.MathUtils.clamp(direction.length() * factor, 8, 160)
    direction.setLength(nextDistance)
    perspectiveCamera.position.copy(target.clone().add(direction))
    controls.update?.()
  }, [controlsRef, perspectiveCamera])

  useEffect(() => {
    const resolvedPreset = viewMode === '3d'
      ? cameraPreset
      : VIEW_MODE_PRESET_MAP[viewMode]
    applyPreset(resolvedPreset)
  }, [applyPreset, cameraPreset, viewMode])

  useEffect(() => {
    if (zoomFitNonce > 0) {
      const resolvedPreset = viewMode === '3d'
        ? cameraPreset
        : VIEW_MODE_PRESET_MAP[viewMode]
      applyPreset(resolvedPreset)
    }
  }, [applyPreset, cameraPreset, viewMode, zoomFitNonce])

  useEffect(() => {
    if (zoomInNonce > 0) {
      zoomByFactor(0.82)
    }
  }, [zoomByFactor, zoomInNonce])

  useEffect(() => {
    if (zoomOutNonce > 0) {
      zoomByFactor(1.22)
    }
  }, [zoomByFactor, zoomOutNonce])

  return null
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
  controlsRef,
  zoomFitNonce,
  zoomInNonce,
  zoomOutNonce,
}: {
  structures: SegmentedStructure[]
  fragments?: FragmentTransform[]
  controlsRef: RefObject<any>
  zoomFitNonce: number
  zoomInNonce: number
  zoomOutNonce: number
}) {
  const { viewerState, setSelectedFragment } = useViewerStore()
  const { selectFragment, currentPlan } = usePlanningStore()
  const contentRef = useRef<THREE.Group>(null)

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
      <SceneCameraController
        contentRef={contentRef}
        controlsRef={controlsRef}
        viewMode={viewerState.viewMode}
        cameraPreset={viewerState.cameraPreset}
        zoomFitNonce={zoomFitNonce}
        zoomInNonce={zoomInNonce}
        zoomOutNonce={zoomOutNonce}
      />

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

      <group ref={contentRef}>
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
              selected={viewerState.selectedFragmentId === structure.label}
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
      </group>

      {/* Camera orbit controls */}
      <OrbitControls
        ref={controlsRef}
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
  const { viewerState, setSelectedFragment } = useViewerStore()
  const { selectFragment } = usePlanningStore()
  const controlsRef = useRef<any>(null)
  const [zoomFitNonce, setZoomFitNonce] = useState(0)
  const [zoomInNonce, setZoomInNonce] = useState(0)
  const [zoomOutNonce, setZoomOutNonce] = useState(0)

  const handleScreenshot = () => {
    const canvas = document.querySelector('canvas.three-canvas') as HTMLCanvasElement
    if (!canvas) return
    const link = document.createElement('a')
    link.download = `facial-align-${Date.now()}.png`
    link.href = canvas.toDataURL()
    link.click()
  }

  const selectedItemLabel = useMemo(() => {
    if (!viewerState.selectedFragmentId) return null
    const structure = structures.find((item) => item.label === viewerState.selectedFragmentId)
    if (structure) return structure.displayName
    const fragment = fragments?.find((item) => item.fragmentId === viewerState.selectedFragmentId)
    return fragment?.displayName ?? viewerState.selectedFragmentId
  }, [fragments, structures, viewerState.selectedFragmentId])

  return (
    <div className={`flex flex-col bg-[rgba(4,9,17,0.94)] ${className}`} style={{ height }} data-testid="viewer-3d">
      {/* Toolbar */}
      <ViewerToolbar
        onScreenshot={handleScreenshot}
        onZoomFit={() => setZoomFitNonce((count) => count + 1)}
        onZoomIn={() => setZoomInNonce((count) => count + 1)}
        onZoomOut={() => setZoomOutNonce((count) => count + 1)}
      />

      {/* Main viewer area */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas */}
        <div className="relative flex-1 bg-[radial-gradient(circle_at_top,rgba(8,145,178,0.08),transparent_26%),linear-gradient(180deg,#020617_0%,#000814_100%)]">
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
            onPointerMissed={() => {
              setSelectedFragment(null)
              selectFragment(null)
            }}
          >
            <Suspense fallback={null}>
              <SceneContent
                structures={structures}
                fragments={fragments}
                controlsRef={controlsRef}
                zoomFitNonce={zoomFitNonce}
                zoomInNonce={zoomInNonce}
                zoomOutNonce={zoomOutNonce}
              />
            </Suspense>
          </Canvas>

          {/* Overlays */}
          <CameraPresetButtons />
          <MeasurementOverlay />

          <div className="absolute bottom-4 right-4 z-10 w-64 rounded-2xl border border-white/10 bg-[rgba(8,14,26,0.78)] px-4 py-3 backdrop-blur-xl">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">Clinician View</p>
            <p className="mt-1 text-sm font-medium text-slate-100">
              {selectedItemLabel ?? 'No structure selected'}
            </p>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              Use the structure panel to isolate anatomy, adjust opacity, or switch to wireframe while reviewing fracture alignment.
            </p>
          </div>

          {/* View mode label */}
          {viewerState.viewMode !== '3d' && (
            <div className="absolute top-2 left-2 rounded-xl border border-cyan-400/20 bg-[rgba(8,14,26,0.88)] px-3 py-1.5 text-xs font-mono text-cyan-300">
              {viewerState.viewMode.toUpperCase()} ORIENTATION
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

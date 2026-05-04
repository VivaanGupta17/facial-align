/**
 * 3D Measurement Tools for the Facial Align surgical viewer.
 *
 * Components:
 *  - DistanceMeasurement  — click two points, renders a line + label in mm
 *  - AngleMeasurement     — click three points, renders two lines + angle in degrees
 *  - MeasurementList      — sidebar list of all active measurements
 *  - MeasurementController — orchestrates active tool state and point picking
 *
 * Uses React Three Fiber (drei Html + Line).
 */

import { useState, useCallback, useRef, type ReactNode } from 'react'
import { Html, Line } from '@react-three/drei'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Trash2, Ruler, Triangle, Eye, EyeOff, Plus } from 'lucide-react'
import { useViewerStore } from '../../stores/viewerStore'
import { angleBetweenPoints, pointToPointDistance } from '../../lib/geometry'
import type { MeasurementAnnotation, Vector3 } from '../../types/medical'

// =============================================================================
// Types
// =============================================================================

export type MeasurementTool = 'distance' | 'angle' | 'none'

// =============================================================================
// Single distance measurement in 3D scene
// =============================================================================

interface DistanceMeasurementProps {
  id: string
  pointA: Vector3
  pointB: Vector3
  color?: string
  label?: string
  visible: boolean
  selected?: boolean
  onRemove?: (id: string) => void
}

export function DistanceMeasurement({
  id,
  pointA,
  pointB,
  color = '#22d3ee',
  label,
  visible,
  selected = false,
  onRemove,
}: DistanceMeasurementProps) {
  if (!visible) return null

  const a = new THREE.Vector3(pointA.x, pointA.y, pointA.z)
  const b = new THREE.Vector3(pointB.x, pointB.y, pointB.z)
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5)
  const distMm = pointToPointDistance(pointA, pointB)

  const lineColor = selected ? '#ffffff' : color
  const points: [number, number, number][] = [
    [a.x, a.y, a.z],
    [b.x, b.y, b.z],
  ]

  return (
    <group>
      {/* Line */}
      <Line
        points={points}
        color={lineColor}
        lineWidth={selected ? 2.5 : 1.5}
        dashed={false}
      />

      {/* Endpoint markers */}
      <mesh position={a}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial color={lineColor} />
      </mesh>
      <mesh position={b}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial color={lineColor} />
      </mesh>

      {/* Label */}
      <Html position={[mid.x, mid.y, mid.z]} center distanceFactor={12}>
        <div
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono font-semibold pointer-events-auto select-none
            ${selected ? 'bg-white text-slate-900' : 'bg-slate-900/90 border border-slate-700 text-cyan-300'}
            whitespace-nowrap shadow-lg`}
          data-testid={`measurement-label-${id}`}
        >
          <span>{distMm.toFixed(2)} mm</span>
          {label && <span className="text-slate-500 font-normal">{label}</span>}
          {onRemove && (
            <button
              onClick={(e) => { e.stopPropagation(); onRemove(id) }}
              className="ml-1 text-slate-500 hover:text-red-400 transition-colors"
              title="Remove measurement"
            >
              ×
            </button>
          )}
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// Angle measurement in 3D scene
// =============================================================================

interface AngleMeasurementProps {
  id: string
  pointA: Vector3  // first arm end
  pointB: Vector3  // vertex
  pointC: Vector3  // second arm end
  color?: string
  label?: string
  visible: boolean
  selected?: boolean
  onRemove?: (id: string) => void
}

export function AngleMeasurement({
  id,
  pointA,
  pointB,
  pointC,
  color = '#a78bfa',
  label,
  visible,
  selected = false,
  onRemove,
}: AngleMeasurementProps) {
  if (!visible) return null

  const a = new THREE.Vector3(pointA.x, pointA.y, pointA.z)
  const b = new THREE.Vector3(pointB.x, pointB.y, pointB.z)
  const c = new THREE.Vector3(pointC.x, pointC.y, pointC.z)
  const angleDeg = angleBetweenPoints(pointA, pointB, pointC)

  const lineColor = selected ? '#ffffff' : color
  const arm1: [number, number, number][] = [[b.x, b.y, b.z], [a.x, a.y, a.z]]
  const arm2: [number, number, number][] = [[b.x, b.y, b.z], [c.x, c.y, c.z]]

  // Arc approximation: sample points along the angle arc
  const arcPoints = buildArcPoints(a, b, c, 0.8, 16)

  // Label position: slightly above the vertex
  const labelPos = new THREE.Vector3(b.x, b.y + 0.5, b.z)

  return (
    <group>
      {/* Arms */}
      <Line points={arm1} color={lineColor} lineWidth={selected ? 2.5 : 1.5} />
      <Line points={arm2} color={lineColor} lineWidth={selected ? 2.5 : 1.5} />

      {/* Arc */}
      {arcPoints.length >= 2 && (
        <Line points={arcPoints} color={lineColor} lineWidth={1} dashed />
      )}

      {/* Vertex marker */}
      <mesh position={b}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color={lineColor} />
      </mesh>

      {/* Endpoint markers */}
      <mesh position={a}>
        <sphereGeometry args={[0.07, 8, 8]} />
        <meshBasicMaterial color={lineColor} />
      </mesh>
      <mesh position={c}>
        <sphereGeometry args={[0.07, 8, 8]} />
        <meshBasicMaterial color={lineColor} />
      </mesh>

      {/* Label */}
      <Html position={[labelPos.x, labelPos.y, labelPos.z]} center distanceFactor={12}>
        <div
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono font-semibold pointer-events-auto select-none
            ${selected ? 'bg-white text-slate-900' : 'bg-slate-900/90 border border-slate-700 text-violet-300'}
            whitespace-nowrap shadow-lg`}
          data-testid={`angle-label-${id}`}
        >
          <span>{angleDeg.toFixed(1)}°</span>
          {label && <span className="text-slate-500 font-normal">{label}</span>}
          {onRemove && (
            <button
              onClick={(e) => { e.stopPropagation(); onRemove(id) }}
              className="ml-1 text-slate-500 hover:text-red-400 transition-colors"
            >×</button>
          )}
        </div>
      </Html>
    </group>
  )
}

// =============================================================================
// Scene measurement renderer — renders all stored measurements
// =============================================================================

export function SceneMeasurements() {
  const { viewerState, removeMeasurement } = useViewerStore()
  const { measurements, showMeasurements } = viewerState

  if (!showMeasurements || measurements.length === 0) return null

  return (
    <group>
      {measurements.map((m) => {
        if (m.type === 'distance' && m.points.length >= 2) {
          return (
            <DistanceMeasurement
              key={m.id}
              id={m.id}
              pointA={m.points[0]}
              pointB={m.points[1]}
              color={m.color}
              label={m.label}
              visible={m.visible}
              onRemove={removeMeasurement}
            />
          )
        }
        if (m.type === 'angle' && m.points.length >= 3) {
          return (
            <AngleMeasurement
              key={m.id}
              id={m.id}
              pointA={m.points[0]}
              pointB={m.points[1]}
              pointC={m.points[2]}
              color={m.color}
              label={m.label}
              visible={m.visible}
              onRemove={removeMeasurement}
            />
          )
        }
        return null
      })}
    </group>
  )
}

// =============================================================================
// Point picker for 3D measurements
// =============================================================================

interface MeasurementControllerProps {
  children?: ReactNode
}

/**
 * Overlay component that intercepts clicks in the 3D scene to pick measurement points.
 * When the active tool is 'measure_distance' or 'measure_angle', it accumulates
 * clicked points and creates a measurement when enough points are collected.
 */
export function MeasurementController({ children }: MeasurementControllerProps) {
  const { viewerState, setActiveTool, addMeasurement } = useViewerStore()
  const { activeTool } = viewerState
  const { camera, gl } = useThree()
  const pendingPoints = useRef<Vector3[]>([])

  const requiredPoints = activeTool === 'measure_distance' ? 2 : activeTool === 'measure_angle' ? 3 : 0

  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (requiredPoints === 0) return

    // Raycast from click position
    const rect = gl.domElement.getBoundingClientRect()
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1

    const raycaster = new THREE.Raycaster()
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera)

    // Use a virtual plane at z=0 for point picking (simplified)
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)
    const target = new THREE.Vector3()
    raycaster.ray.intersectPlane(plane, target)

    const pt: Vector3 = { x: target.x, y: target.y, z: target.z }
    pendingPoints.current = [...pendingPoints.current, pt]

    if (pendingPoints.current.length >= requiredPoints) {
      const points = pendingPoints.current.slice(0, requiredPoints)
      pendingPoints.current = []

      const id = `meas-${Date.now()}`
      const isDistance = requiredPoints === 2
      const value = isDistance
        ? pointToPointDistance(points[0], points[1])
        : angleBetweenPoints(points[0], points[1], points[2])

      const measurement: MeasurementAnnotation = {
        id,
        type: isDistance ? 'distance' : 'angle',
        points,
        value,
        unit: isDistance ? 'mm' : 'deg',
        label: isDistance ? 'Distance' : 'Angle',
        color: isDistance ? '#22d3ee' : '#a78bfa',
        visible: true,
      }

      addMeasurement(measurement)
      setActiveTool('none')
    }
  }, [requiredPoints, camera, gl, addMeasurement, setActiveTool])

  if (requiredPoints === 0) return <>{children}</>

  return (
    <div
      style={{ position: 'absolute', inset: 0, cursor: 'crosshair', zIndex: 5 }}
      onClick={handleClick}
    >
      {/* Pending point count indicator */}
      <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-slate-900/90 border border-slate-700 rounded px-3 py-1 text-xs font-mono text-cyan-400 pointer-events-none">
        {requiredPoints - pendingPoints.current.length} point{requiredPoints - pendingPoints.current.length !== 1 ? 's' : ''} remaining — click in the viewer
      </div>
      {children}
    </div>
  )
}

// =============================================================================
// MeasurementList — sidebar UI for managing measurements
// =============================================================================

interface MeasurementListProps {
  className?: string
}

export function MeasurementList({ className = '' }: MeasurementListProps) {
  const { viewerState, removeMeasurement, clearMeasurements, setActiveTool, addMeasurement } = useViewerStore()
  const { measurements, activeTool } = viewerState
  const [hiddenIds, setHiddenIds] = useState<Set<string>>(new Set())

  const toggleVisibility = useCallback((id: string) => {
    setHiddenIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
    // Update measurement visibility in store
    const m = measurements.find(x => x.id === id)
    if (m) {
      addMeasurement({ ...m, visible: !hiddenIds.has(id) })
      removeMeasurement(id)
    }
  }, [measurements, hiddenIds, addMeasurement, removeMeasurement])

  return (
    <div className={`flex flex-col bg-slate-900 ${className}`} data-testid="measurement-list">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800">
        <span className="text-xs font-semibold text-slate-300">Measurements</span>
        <div className="flex items-center gap-1">
          {measurements.length > 0 && (
            <button
              onClick={clearMeasurements}
              title="Clear all measurements"
              className="text-slate-500 hover:text-red-400 transition-colors"
              data-testid="clear-all-measurements"
            >
              <Trash2 size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Tool picker */}
      <div className="flex gap-1 p-2 border-b border-slate-800">
        <button
          onClick={() => setActiveTool(activeTool === 'measure_distance' ? 'none' : 'measure_distance')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors ${
            activeTool === 'measure_distance'
              ? 'bg-cyan-900 text-cyan-300 border border-cyan-700'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
          }`}
          data-testid="tool-distance"
        >
          <Ruler size={12} />
          Distance
        </button>
        <button
          onClick={() => setActiveTool(activeTool === 'measure_angle' ? 'none' : 'measure_angle')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors ${
            activeTool === 'measure_angle'
              ? 'bg-violet-900 text-violet-300 border border-violet-700'
              : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
          }`}
          data-testid="tool-angle"
        >
          <Triangle size={12} />
          Angle
        </button>
      </div>

      {/* Measurements list */}
      <div className="flex-1 overflow-y-auto">
        {measurements.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 text-center text-slate-600">
            <Plus size={20} className="mb-2" />
            <p className="text-xs">No measurements yet</p>
            <p className="text-2xs mt-0.5">Select a tool above</p>
          </div>
        ) : (
          measurements.map((m) => (
            <MeasurementRow
              key={m.id}
              measurement={m}
              hidden={hiddenIds.has(m.id)}
              onToggleVisibility={() => toggleVisibility(m.id)}
              onRemove={() => removeMeasurement(m.id)}
            />
          ))
        )}
      </div>
    </div>
  )
}

// =============================================================================
// Individual measurement row
// =============================================================================

interface MeasurementRowProps {
  measurement: MeasurementAnnotation
  hidden: boolean
  onToggleVisibility: () => void
  onRemove: () => void
}

function MeasurementRow({ measurement: m, hidden, onToggleVisibility, onRemove }: MeasurementRowProps) {
  const TypeIcon = m.type === 'distance' ? Ruler : Triangle

  return (
    <div
      className="flex items-center gap-2 px-3 py-2 border-b border-slate-800 last:border-b-0 hover:bg-slate-800/50 transition-colors"
      data-testid={`measurement-row-${m.id}`}
    >
      {/* Color swatch */}
      <span
        className="w-2 h-2 rounded-full shrink-0"
        style={{ backgroundColor: m.color }}
      />

      {/* Type icon */}
      <TypeIcon size={11} className={m.type === 'distance' ? 'text-cyan-500' : 'text-violet-500'} />

      {/* Value */}
      <div className="flex-1 min-w-0">
        <span className={`text-xs font-mono font-semibold ${hidden ? 'text-slate-600' : 'text-slate-200'}`}>
          {m.value.toFixed(m.type === 'distance' ? 2 : 1)} {m.unit}
        </span>
        <span className="text-2xs text-slate-500 ml-1">{m.label}</span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 shrink-0">
        <button
          onClick={onToggleVisibility}
          className="text-slate-600 hover:text-slate-300 transition-colors"
          title={hidden ? 'Show' : 'Hide'}
          data-testid={`toggle-vis-${m.id}`}
        >
          {hidden ? <EyeOff size={12} /> : <Eye size={12} />}
        </button>
        <button
          onClick={onRemove}
          className="text-slate-600 hover:text-red-400 transition-colors"
          title="Remove"
          data-testid={`remove-meas-${m.id}`}
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// Utility: build arc sample points between two arm directions at a vertex
// =============================================================================

function buildArcPoints(
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
  radius: number,
  segments: number
): [number, number, number][] {
  const ba = a.clone().sub(b).normalize()
  const bc = c.clone().sub(b).normalize()

  // Check for degenerate case
  if (ba.length() < 0.001 || bc.length() < 0.001) return []
  const dot = Math.max(-1, Math.min(1, ba.dot(bc)))
  const angle = Math.acos(dot)
  if (angle < 0.01) return []

  const axis = new THREE.Vector3().crossVectors(ba, bc)
  if (axis.length() < 0.001) return []
  axis.normalize()

  const points: [number, number, number][] = []
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * angle
    const pt = ba.clone()
      .applyAxisAngle(axis, t)
      .multiplyScalar(radius)
      .add(b)
    points.push([pt.x, pt.y, pt.z])
  }
  return points
}

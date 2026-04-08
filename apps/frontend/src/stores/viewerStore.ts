import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type {
  ViewerState,
  StructureLabel,
  StructureVisibility,
  MeasurementAnnotation,
  CameraPreset,
} from '../types/medical'

// Default colors per anatomical structure
const DEFAULT_STRUCTURE_COLORS: Partial<Record<StructureLabel, string>> = {
  mandible: '#eab308',
  maxilla: '#3b82f6',
  zygoma_left: '#8b5cf6',
  zygoma_right: '#8b5cf6',
  orbit_left: '#f97316',
  orbit_right: '#f97316',
  frontal_bone: '#6366f1',
  nasal_bones: '#ec4899',
  teeth_upper: '#f8fafc',
  teeth_lower: '#e2e8f0',
  sphenoid: '#14b8a6',
  temporal_left: '#84cc16',
  temporal_right: '#84cc16',
  skull_base: '#64748b',
  fragment_1: '#ef4444',
  fragment_2: '#f97316',
  fragment_3: '#eab308',
  fragment_4: '#22c55e',
}

export const CAMERA_PRESETS: CameraPreset[] = [
  { name: 'anterior', label: 'Anterior', position: { x: 0, y: 0, z: 200 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
  { name: 'posterior', label: 'Posterior', position: { x: 0, y: 0, z: -200 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
  { name: 'lateral_l', label: 'Lateral L', position: { x: -200, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
  { name: 'lateral_r', label: 'Lateral R', position: { x: 200, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } },
  { name: 'superior', label: 'Superior', position: { x: 0, y: 200, z: 0 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: -1 } },
  { name: 'inferior', label: 'Inferior', position: { x: 0, y: -200, z: 0 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } },
]

const makeDefaultVisibility = (label: StructureLabel): StructureVisibility => ({
  label,
  visible: true,
  opacity: 1.0,
  color: DEFAULT_STRUCTURE_COLORS[label] ?? '#64748b',
  wireframe: false,
  selected: false,
})

const ALL_LABELS: StructureLabel[] = [
  'mandible', 'maxilla', 'zygoma_left', 'zygoma_right',
  'orbit_left', 'orbit_right', 'frontal_bone', 'nasal_bones',
  'teeth_upper', 'teeth_lower', 'sphenoid', 'temporal_left',
  'temporal_right', 'skull_base', 'fragment_1', 'fragment_2', 'fragment_3', 'fragment_4',
]

const defaultVisibility = Object.fromEntries(
  ALL_LABELS.map(l => [l, makeDefaultVisibility(l)])
) as Record<StructureLabel, StructureVisibility>

interface ViewerStoreState {
  viewerState: ViewerState
  // Actions
  setViewMode: (mode: ViewerState['viewMode']) => void
  setStructureVisible: (label: StructureLabel, visible: boolean) => void
  setStructureOpacity: (label: StructureLabel, opacity: number) => void
  setStructureColor: (label: StructureLabel, color: string) => void
  setStructureWireframe: (label: StructureLabel, wireframe: boolean) => void
  setSelectedFragment: (id: string | null) => void
  setCameraPreset: (preset: string) => void
  setActiveTool: (tool: ViewerState['activeTool']) => void
  toggleStructuresPanel: () => void
  toggleGrid: () => void
  addMeasurement: (m: MeasurementAnnotation) => void
  removeMeasurement: (id: string) => void
  clearMeasurements: () => void
  resetVisibility: () => void
}

export const useViewerStore = create<ViewerStoreState>()(
  devtools(
    (set) => ({
      viewerState: {
        viewMode: '3d',
        structureVisibility: defaultVisibility,
        selectedFragmentId: null,
        cameraPreset: 'anterior',
        showGrid: true,
        showAxes: false,
        showMeasurements: true,
        activeTool: 'none',
        measurements: [],
        showStructuresPanel: true,
      },

      setViewMode: (viewMode) =>
        set(s => ({ viewerState: { ...s.viewerState, viewMode } })),

      setStructureVisible: (label, visible) =>
        set(s => ({
          viewerState: {
            ...s.viewerState,
            structureVisibility: {
              ...s.viewerState.structureVisibility,
              [label]: { ...s.viewerState.structureVisibility[label], visible },
            },
          },
        })),

      setStructureOpacity: (label, opacity) =>
        set(s => ({
          viewerState: {
            ...s.viewerState,
            structureVisibility: {
              ...s.viewerState.structureVisibility,
              [label]: { ...s.viewerState.structureVisibility[label], opacity },
            },
          },
        })),

      setStructureColor: (label, color) =>
        set(s => ({
          viewerState: {
            ...s.viewerState,
            structureVisibility: {
              ...s.viewerState.structureVisibility,
              [label]: { ...s.viewerState.structureVisibility[label], color },
            },
          },
        })),

      setStructureWireframe: (label, wireframe) =>
        set(s => ({
          viewerState: {
            ...s.viewerState,
            structureVisibility: {
              ...s.viewerState.structureVisibility,
              [label]: { ...s.viewerState.structureVisibility[label], wireframe },
            },
          },
        })),

      setSelectedFragment: (id) =>
        set(s => ({ viewerState: { ...s.viewerState, selectedFragmentId: id } })),

      setCameraPreset: (cameraPreset) =>
        set(s => ({ viewerState: { ...s.viewerState, cameraPreset } })),

      setActiveTool: (activeTool) =>
        set(s => ({ viewerState: { ...s.viewerState, activeTool } })),

      toggleStructuresPanel: () =>
        set(s => ({ viewerState: { ...s.viewerState, showStructuresPanel: !s.viewerState.showStructuresPanel } })),

      toggleGrid: () =>
        set(s => ({ viewerState: { ...s.viewerState, showGrid: !s.viewerState.showGrid } })),

      addMeasurement: (m) =>
        set(s => ({ viewerState: { ...s.viewerState, measurements: [...s.viewerState.measurements, m] } })),

      removeMeasurement: (id) =>
        set(s => ({ viewerState: { ...s.viewerState, measurements: s.viewerState.measurements.filter(m => m.id !== id) } })),

      clearMeasurements: () =>
        set(s => ({ viewerState: { ...s.viewerState, measurements: [] } })),

      resetVisibility: () =>
        set(s => ({ viewerState: { ...s.viewerState, structureVisibility: defaultVisibility } })),
    }),
    { name: 'ViewerStore' }
  )
)

import { useViewerStore } from './viewerStore'

describe('viewerStore', () => {
  beforeEach(() => {
    useViewerStore.setState({
      viewerState: {
        viewMode: '3d',
        structureVisibility: useViewerStore.getInitialState().viewerState.structureVisibility,
        selectedFragmentId: null,
        cameraPreset: 'anterior',
        showGrid: true,
        showAxes: false,
        showMeasurements: true,
        activeTool: 'none',
        measurements: [],
        showStructuresPanel: true,
      },
    })
  })

  it('isolates a structure and can restore all structures', () => {
    const { isolateStructure, showAllStructures } = useViewerStore.getState()

    isolateStructure('mandible')
    const isolated = useViewerStore.getState().viewerState.structureVisibility
    expect(isolated.mandible.visible).toBe(true)
    expect(isolated.maxilla.visible).toBe(false)

    showAllStructures()
    const restored = useViewerStore.getState().viewerState.structureVisibility
    expect(restored.mandible.visible).toBe(true)
    expect(restored.maxilla.visible).toBe(true)
  })

  it('updates structure opacity and toggles grid visibility', () => {
    const { setStructureOpacity, toggleGrid } = useViewerStore.getState()

    setStructureOpacity('mandible', 0.45)
    toggleGrid()

    const state = useViewerStore.getState().viewerState
    expect(state.structureVisibility.mandible.opacity).toBe(0.45)
    expect(state.showGrid).toBe(false)
  })
})

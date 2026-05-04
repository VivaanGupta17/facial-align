import { makePlan, makeTransform } from '../test/factories'
import { usePlanningStore } from './planningStore'

describe('planningStore', () => {
  beforeEach(() => {
    usePlanningStore.setState(usePlanningStore.getInitialState())
    usePlanningStore.getState().setPlan(makePlan())
  })

  it('tracks transform history and supports undo/redo', () => {
    const store = usePlanningStore.getState()
    const nextTransform = makeTransform({
      translation: { x: 5, y: 0, z: 0 },
    })

    store.updateFragmentTransform('fragment-1', nextTransform, 'manual')
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x).toBe(5)
    expect(usePlanningStore.getState().transformHistory).toHaveLength(1)

    store.undo()
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x).toBe(0)

    store.redo()
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x).toBe(5)
  })

  it('accepts AI suggestions and toggles fragment locks', () => {
    const store = usePlanningStore.getState()

    store.acceptAiSuggestion('fragment-1')
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x).toBe(2)

    store.toggleFragmentLock('fragment-1')
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].isLocked).toBe(true)
  })
})

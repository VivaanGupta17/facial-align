import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import { planningApi } from '../../lib/api'
import { makePlan } from '../../test/factories'
import { usePlanningStore } from '../../stores/planningStore'
import FragmentControls from './FragmentControls'

describe('FragmentControls', () => {
  beforeEach(() => {
    usePlanningStore.setState(usePlanningStore.getInitialState())
    usePlanningStore.getState().setPlan(makePlan())
    usePlanningStore.getState().selectFragment('fragment-1')
  })

  it('updates a transform value, saves it, and supports undo', async () => {
    vi.spyOn(planningApi, 'updateFragmentTransform').mockResolvedValue(makePlan())

    render(<FragmentControls />)

    fireEvent.click(screen.getByTestId('transform-value-Translation-X'))
    const input = screen.getByTestId('transform-input-Translation-X')
    fireEvent.change(input, { target: { value: '5.5' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => {
      expect(
        usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x
      ).toBe(5.5)
    })

    fireEvent.click(screen.getByTestId('save-changes'))
    await waitFor(() => {
      expect(planningApi.updateFragmentTransform).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByTestId('undo-transform'))
    expect(
      usePlanningStore.getState().currentPlan?.fragmentTransforms[0].currentTransform.translation.x
    ).toBe(0)
  })

  it('locks and unlocks the selected fragment', () => {
    render(<FragmentControls />)

    fireEvent.click(screen.getByTestId('lock-fragment'))
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].isLocked).toBe(true)

    fireEvent.click(screen.getByTestId('unlock-fragment'))
    expect(usePlanningStore.getState().currentPlan?.fragmentTransforms[0].isLocked).toBe(false)
  })
})

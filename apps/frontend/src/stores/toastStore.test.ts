import { vi } from 'vitest'

import { useToastStore } from './toastStore'

describe('toastStore', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    useToastStore.setState({ toasts: [] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('adds a toast and automatically removes it after its duration', () => {
    useToastStore.getState().addToast({
      type: 'success',
      message: 'Saved',
      duration: 1000,
    })

    expect(useToastStore.getState().toasts).toHaveLength(1)
    vi.advanceTimersByTime(1000)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })
})

import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import { useViewerStore } from '../../stores/viewerStore'
import ViewerToolbar from './ViewerToolbar'

describe('ViewerToolbar', () => {
  beforeEach(() => {
    useViewerStore.setState(useViewerStore.getInitialState())
  })

  it('toggles view mode, grid, and callback actions', () => {
    const onZoomIn = vi.fn()
    const onZoomOut = vi.fn()
    const onZoomFit = vi.fn()

    render(
      <ViewerToolbar onZoomIn={onZoomIn} onZoomOut={onZoomOut} onZoomFit={onZoomFit} />
    )

    fireEvent.click(screen.getByTestId('view-mode-axial'))
    expect(useViewerStore.getState().viewerState.viewMode).toBe('axial')

    fireEvent.click(screen.getByTestId('tool-grid'))
    expect(useViewerStore.getState().viewerState.showGrid).toBe(false)

    fireEvent.click(screen.getByTestId('tool-zoom-in'))
    fireEvent.click(screen.getByTestId('tool-zoom-out'))
    fireEvent.click(screen.getByTestId('tool-zoom-fit'))

    expect(onZoomIn).toHaveBeenCalledOnce()
    expect(onZoomOut).toHaveBeenCalledOnce()
    expect(onZoomFit).toHaveBeenCalledOnce()
  })
})

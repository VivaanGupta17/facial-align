import { fireEvent, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import { casesApi } from '../lib/api'
import { makeCaseListItem } from '../test/factories'
import { renderWithProviders } from '../test/render'
import CaseListPage from './CaseListPage'

describe('CaseListPage', () => {
  it('passes the selected status filter into the cases query', async () => {
    const listSpy = vi.spyOn(casesApi, 'list').mockImplementation(async (params = {}) => {
      const status = params.status?.[0]
      const item =
        status === 'approved'
          ? makeCaseListItem({
              id: 'case-approved',
              caseNumber: 'FA-2026-APPROVED',
              status: 'approved',
            })
          : makeCaseListItem()

      return {
        items: [item],
        total: 1,
        page: 1,
        pageSize: 20,
        pages: 1,
        hasMore: false,
      }
    })

    renderWithProviders(<CaseListPage />)

    expect(await screen.findByText('FA-2026-CASE01')).toBeInTheDocument()

    fireEvent.change(screen.getByTestId('status-filter'), { target: { value: 'approved' } })

    await waitFor(() => {
      expect(screen.getByText('FA-2026-APPROVED')).toBeInTheDocument()
    })
    expect(listSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: ['approved'],
      })
    )
  })

  it('filters the rendered rows when using the search box', async () => {
    vi.spyOn(casesApi, 'list').mockResolvedValue({
      items: [
        makeCaseListItem({ id: 'case-a', caseNumber: 'FA-REL-1001', patientId: 'patient-a' }),
        makeCaseListItem({ id: 'case-b', caseNumber: 'FA-REL-1002', patientId: 'patient-b' }),
      ],
      total: 2,
      page: 1,
      pageSize: 20,
      pages: 1,
      hasMore: false,
    })

    renderWithProviders(<CaseListPage />)

    expect(await screen.findByText('FA-REL-1001')).toBeInTheDocument()
    expect(screen.getByText('FA-REL-1002')).toBeInTheDocument()

    fireEvent.change(screen.getByTestId('search-input'), { target: { value: '1002' } })

    await waitFor(() => {
      expect(screen.queryByText('FA-REL-1001')).not.toBeInTheDocument()
      expect(screen.getByText('FA-REL-1002')).toBeInTheDocument()
    })
  })
})

import { screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import { casesApi, dashboardApi } from '../lib/api'
import { makeCaseListItem, makeDashboardStats, makeSystemHealth } from '../test/factories'
import { renderWithProviders } from '../test/render'
import DashboardPage from './DashboardPage'

describe('DashboardPage', () => {
  it('renders stats, health posture, and recent cases', async () => {
    vi.spyOn(dashboardApi, 'getStats').mockResolvedValue(makeDashboardStats())
    vi.spyOn(dashboardApi, 'getSystemHealth').mockResolvedValue(makeSystemHealth())
    vi.spyOn(casesApi, 'getRecent').mockResolvedValue([
      makeCaseListItem(),
      makeCaseListItem({
        id: 'case-002',
        caseNumber: 'FA-2026-CASE02',
        status: 'review',
      }),
    ])

    renderWithProviders(<DashboardPage />)

    expect(screen.getByText('Surgical Command Center')).toBeInTheDocument()
    expect(await screen.findByText('Active Cases')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByTestId('recent-cases-table')).toBeInTheDocument()
    })
    expect(screen.getByText('FA-2026-CASE02')).toBeInTheDocument()
  })
})

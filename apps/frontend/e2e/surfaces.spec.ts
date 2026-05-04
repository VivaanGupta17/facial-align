import { expect, test } from '@playwright/test'

import { attachScreenshot, freezeClock, loginAsDemoSurgeon } from './support'

test.describe('Supporting Clinician Surfaces', () => {
  test.beforeEach(async ({ page }) => {
    await freezeClock(page)
    await loginAsDemoSurgeon(page)
  })

  test('renders models and compute posture honestly', async ({ page }, testInfo) => {
    await page.goto('/models')
    await expect(page.getByTestId('models-page')).toBeVisible()
    await expect(page.getByTestId('models-table')).toBeVisible()
    await expect(
      page.getByTestId('models-table').locator('tr', { hasText: 'Dental Segmentation' })
    ).toBeVisible()

    await page.goto('/compute')
    await expect(page.getByTestId('compute-page')).toBeVisible()
    await expect(page.getByTestId('queue-panel')).toBeVisible()
    await expect(page.getByTestId('latency-panel')).toBeVisible()
    await expect(page.getByTestId('storage-panel')).toBeVisible()

    await attachScreenshot(page, testInfo, 'compute-runtime-status')
  })

  test('renders studies, settings, and logout-ready operator surfaces', async ({ page }, testInfo) => {
    await page.goto('/studies')
    await expect(page.getByTestId('studies-page')).toBeVisible()
    await expect(page.getByTestId('studies-table')).toBeVisible()
    await page.getByTestId('studies-search').fill('1.2.826')

    await page.goto('/settings')
    await expect(page.getByTestId('settings-page')).toBeVisible()
    await page.getByTestId('settings-nav-notifications').click()
    await expect(page.getByTestId('settings-notifications')).toBeVisible()
    await page.getByTestId('settings-nav-shortcuts').click()
    await expect(page.getByTestId('settings-shortcuts')).toBeVisible()

    await attachScreenshot(page, testInfo, 'settings-shortcuts')
  })
})

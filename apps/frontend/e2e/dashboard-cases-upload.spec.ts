import { expect, test } from '@playwright/test'

import { attachScreenshot, freezeClock, loginAsDemoSurgeon } from './support'

const SAMPLE_UPLOAD = new URL('./fixtures/sample-upload.dcm', import.meta.url).pathname

test.describe('Dashboard, Cases, And Upload', () => {
  test.beforeEach(async ({ page }) => {
    await freezeClock(page)
    await loginAsDemoSurgeon(page)
  })

  test('loads dashboard and navigates through primary actions', async ({ page }, testInfo) => {
    await expect(page.getByTestId('dashboard-page')).toBeVisible()
    await expect(page.getByTestId('recent-cases-panel')).toBeVisible()
    await expect(page.getByTestId('system-health-panel')).toBeVisible()

    await page.getByTestId('quick-action-upload-dicom').click()
    await expect(page.getByTestId('upload-page')).toBeVisible()

    await page.getByTestId('nav-dashboard').click()
    await expect(page.getByTestId('dashboard-page')).toBeVisible()

    await page.getByTestId('quick-action-view-3d-models').click()
    await expect(page.getByTestId('models-page')).toBeVisible()

    await attachScreenshot(page, testInfo, 'dashboard-models-route')
  })

  test('filters case list and opens a seeded case detail view', async ({ page }, testInfo) => {
    await page.goto('/cases')
    await expect(page.getByTestId('case-list-page')).toBeVisible()

    await page.getByTestId('search-input').fill('FA-REL-1002')
    await expect(page.locator('tr:has-text("FA-REL-1002")')).toHaveCount(1)

    await page.getByTestId('status-filter').selectOption('planning')
    await expect(page.locator('tr:has-text("FA-REL-1002")')).toHaveCount(1)

    await page.locator('tr:has-text("FA-REL-1002")').first().click()
    await expect(page.getByTestId('case-detail-page')).toBeVisible()
    await expect(page.getByTestId('overview-tab')).toBeVisible()

    await attachScreenshot(page, testInfo, 'cases-case-detail-overview')
  })

  test('walks the real upload flow into case creation', async ({ page }, testInfo) => {
    await page.goto('/upload')
    await expect(page.getByTestId('upload-page')).toBeVisible()
    await expect(page.getByTestId('upload-btn')).toBeDisabled()

    await page.getByTestId('file-input').setInputFiles(SAMPLE_UPLOAD)
    await page.getByTestId('patient-mrn-input').fill('MRN-PLAYWRIGHT-001')
    await expect(page.getByTestId('upload-btn')).toBeEnabled()

    await page.getByTestId('upload-btn').click()
    await expect(page.getByTestId('step-2')).toBeVisible()

    await page.getByTestId('step2-next').click()
    await expect(page.getByTestId('step-3')).toBeVisible()

    await page.getByTestId('study-label-input').fill('Playwright Upload Study')
    await page.getByTestId('clinical-notes').fill('Automated release-test upload flow.')
    await page.getByTestId('create-case-btn').click()

    await expect(page.getByTestId('step-4')).toBeVisible()
    await expect(
      page.getByTestId('view-case-btn').or(page.getByTestId('view-cases-btn'))
    ).toBeVisible()

    await attachScreenshot(page, testInfo, 'upload-case-created')
  })
})

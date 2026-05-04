import { expect, test } from '@playwright/test'

import { attachScreenshot, freezeClock, loginAsDemoSurgeon, openSeededCase } from './support'

test.describe('Clinician Workspaces', () => {
  test.beforeEach(async ({ page }) => {
    await freezeClock(page)
    await loginAsDemoSurgeon(page)
  })

  test('supports segmentation review interactions', async ({ page }, testInfo) => {
    await openSeededCase(page, 'FA-REL-1003')
    await page.getByTestId('tab-segmentation').click()

    await expect(page.getByTestId('segmentation-review')).toBeVisible()
    await expect(page.getByTestId('segmentation-panel')).toBeVisible()
    await expect(page.getByTestId('structures-list')).toBeVisible()

    const enabledAcceptButton = page.locator('button[data-testid^="accept-"]:not([disabled])').first()
    if (await enabledAcceptButton.count()) {
      await enabledAcceptButton.click()
    } else {
      await expect(page.locator('button[data-testid^="accept-"][disabled]').first()).toBeVisible()
    }
    await attachScreenshot(page, testInfo, 'segmentation-review-workspace')
  })

  test('supports planning viewer and fragment controls', async ({ page }, testInfo) => {
    await openSeededCase(page, 'FA-REL-1002')
    await page.getByTestId('tab-planning').click()

    await expect(page.getByTestId('reduction-workspace')).toBeVisible()
    await expect(page.getByTestId('viewer-3d')).toBeVisible()

    await page.getByTestId('tool-zoom-in').click()
    await page.getByTestId('tool-zoom-out').click()
    await page.getByTestId('tool-zoom-fit').click()
    await page.getByTestId('preset-anterior').click()
    await page.getByTestId('preset-lateral_l').click()
    await page.getByTestId('preset-superior').click()

    const structuresPanel = page.getByTestId('structures-panel')
    if (!(await structuresPanel.isVisible().catch(() => false))) {
      await page.getByTestId('tool-structures-panel').click()
    }
    await expect(structuresPanel).toBeVisible()

    const firstStructure = structuresPanel.locator('[data-testid^="structure-row-"]').first()
    await firstStructure.click()
    const firstOpacity = structuresPanel.locator('input[type="range"]').first()
    await firstOpacity.evaluate((element) => {
      const input = element as HTMLInputElement
      input.value = '0.55'
      input.dispatchEvent(new Event('input', { bubbles: true }))
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })

    await page.getByTestId('fragment-list').locator('[data-testid^="fragment-item-"]').first().click()
    await expect(page.getByTestId('fragment-controls')).toBeVisible()

    await page.getByTestId('transform-value-Translation-X').click()
    await page.getByTestId('transform-input-Translation-X').fill('1.25')
    await page.getByTestId('transform-input-Translation-X').press('Enter')

    await page.getByTestId('lock-fragment').click()
    await page.getByTestId('unlock-fragment').click()
    await page.getByTestId('undo-transform').click()
    await page.getByTestId('redo-transform').click()
    await page.getByTestId('save-changes').click()

    await attachScreenshot(page, testInfo, 'planning-workspace-fragment-selected')
  })

  test('supports surgeon review in pending, revision, and approved states', async ({ page }, testInfo) => {
    await openSeededCase(page, 'FA-REL-1007')
    await page.getByTestId('tab-review').click()

    await expect(page.getByTestId('surgeon-review')).toBeVisible()
    await page.getByTestId('checklist-item-seg-accuracy').click()
    await page.getByTestId('surgeon-notes').fill('Pending review checked in release-test browser suite.')
    await attachScreenshot(page, testInfo, 'review-pending')

    await openSeededCase(page, 'FA-REL-1004')
    await page.getByTestId('tab-review').click()
    await expect(page.getByTestId('decision-status')).toContainText(/Revision Requested/i)

    await openSeededCase(page, 'FA-REL-1001')
    await page.getByTestId('tab-review').click()
    await expect(page.getByTestId('decision-status')).toContainText(/Plan Approved/i)
    await attachScreenshot(page, testInfo, 'review-approved')
  })
})

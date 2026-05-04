import { expect, type Page, type TestInfo } from '@playwright/test'

export const DEMO_SURGEON = {
  email: 'surgeon@facialign.local',
  password: 'surgeon',
}

const FIXED_TIME = new Date('2026-05-03T14:30:00Z').getTime()

export async function freezeClock(page: Page) {
  await page.addInitScript(({ fixedTime }) => {
    const NativeDate = Date

    class MockDate extends NativeDate {
      constructor(...args: ConstructorParameters<typeof Date>) {
        if (args.length === 0) {
          super(fixedTime)
          return
        }
        super(...args)
      }

      static now() {
        return fixedTime
      }
    }

    Object.setPrototypeOf(MockDate, NativeDate)
    // @ts-expect-error browser-side override for deterministic screenshots/tests
    window.Date = MockDate
  }, { fixedTime: FIXED_TIME })
}

export async function loginAsDemoSurgeon(page: Page) {
  await page.goto('/login')
  await expect(page.getByTestId('login-page')).toBeVisible()
  await page.getByTestId('login-email').fill(DEMO_SURGEON.email)
  await page.getByTestId('login-password').fill(DEMO_SURGEON.password)
  await page.getByTestId('login-submit').click()
  await page.waitForURL('**/dashboard', { timeout: 15_000 })
  await expect(page.getByTestId('dashboard-page')).toBeVisible({ timeout: 15_000 })
}

export async function registerFreshUser(page: Page) {
  const uniqueEmail = `playwright+${Date.now()}@facialign.local`

  await page.goto('/register')
  await expect(page.getByTestId('register-page')).toBeVisible()
  await page.getByTestId('register-name').fill('Playwright Surgeon')
  await page.getByTestId('register-email').fill(uniqueEmail)
  await page.getByTestId('register-password').fill('playwright-secret')
  await page.getByTestId('register-institution').fill('Integration Hospital')
  await page.getByTestId('register-specialty').fill('CMF Surgery')
  await page.getByTestId('register-submit').click()
  await expect(page.getByTestId('dashboard-page')).toBeVisible()
}

export async function openSeededCase(page: Page, caseNumber: string) {
  await page.goto('/cases')
  await expect(page.getByTestId('case-list-page')).toBeVisible({ timeout: 15_000 })
  await page.getByTestId('search-input').fill(caseNumber)
  const row = page.locator(`tr:has-text("${caseNumber}")`).first()
  await expect(row).toBeVisible({ timeout: 15_000 })
  await row.click()
  await expect(page.getByTestId('case-detail-page')).toBeVisible({ timeout: 15_000 })
}

export async function attachScreenshot(
  page: Page,
  testInfo: TestInfo,
  name: string,
  fullPage = true,
) {
  if (process.env.PLAYWRIGHT_VISUAL_ASSERTS === '1') {
    await expect(page).toHaveScreenshot(`${name}.png`, {
      fullPage,
      animations: 'disabled',
      caret: 'hide',
    })
  }

  const path = testInfo.outputPath(`${name}.png`)
  await page.screenshot({
    path,
    fullPage,
    animations: 'disabled',
    caret: 'hide',
  })
  await testInfo.attach(name, {
    path,
    contentType: 'image/png',
  })
}

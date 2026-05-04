import { expect, test } from '@playwright/test'

import { attachScreenshot, freezeClock, loginAsDemoSurgeon, registerFreshUser } from './support'

test.describe('Auth Flow', () => {
  test.beforeEach(async ({ page }) => {
    await freezeClock(page)
  })

  test('redirects protected routes to login', async ({ page }, testInfo) => {
    await page.goto('/dashboard')
    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByTestId('login-page')).toBeVisible()
    await attachScreenshot(page, testInfo, 'auth-redirect-login')
  })

  test('shows login failure state for invalid credentials', async ({ page }, testInfo) => {
    await page.goto('/login')
    await page.getByTestId('login-email').fill('wrong@facialign.local')
    await page.getByTestId('login-password').fill('incorrect-password')
    await page.getByTestId('login-submit').click()

    await expect(page.getByTestId('login-error')).toBeVisible()
    await attachScreenshot(page, testInfo, 'auth-login-error')
  })

  test('supports demo login and logout', async ({ page }, testInfo) => {
    await loginAsDemoSurgeon(page)
    await expect(page.getByTestId('topbar')).toBeVisible()

    await page.goto('/settings')
    await expect(page.getByTestId('settings-page')).toBeVisible()
    await page.getByTestId('logout-btn').click()

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByTestId('login-page')).toBeVisible()
    await attachScreenshot(page, testInfo, 'auth-logout')
  })

  test('supports registration into the protected shell', async ({ page }, testInfo) => {
    await registerFreshUser(page)
    await expect(page.getByTestId('app-shell')).toBeVisible()
    await attachScreenshot(page, testInfo, 'auth-register-dashboard')
  })
})

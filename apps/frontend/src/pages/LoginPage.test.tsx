import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'

import { authApi } from '../lib/api'
import LoginPage from './LoginPage'

describe('LoginPage', () => {
  it('logs a user in and stores tokens', async () => {
    vi.spyOn(authApi, 'login').mockResolvedValue({
      access_token: 'access-token',
      refresh_token: 'refresh-token',
      token_type: 'bearer',
      expires_in: 3600,
    })

    render(
      <MemoryRouter initialEntries={['/login']}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/dashboard" element={<div>Dashboard Home</div>} />
        </Routes>
      </MemoryRouter>
    )

    fireEvent.change(screen.getByTestId('login-email'), { target: { value: 'surgeon@example.com' } })
    fireEvent.change(screen.getByTestId('login-password'), { target: { value: 'secure-password' } })
    fireEvent.click(screen.getByTestId('login-submit'))

    await waitFor(() => {
      expect(screen.getByText('Dashboard Home')).toBeInTheDocument()
    })
    expect(localStorage.getItem('auth_token')).toBe('access-token')
    expect(localStorage.getItem('refresh_token')).toBe('refresh-token')
  })

  it('shows the API error when login fails', async () => {
    vi.spyOn(authApi, 'login').mockRejectedValue(new Error('Invalid email or password'))

    render(
      <MemoryRouter>
        <LoginPage />
      </MemoryRouter>
    )

    fireEvent.change(screen.getByTestId('login-email'), { target: { value: 'surgeon@example.com' } })
    fireEvent.change(screen.getByTestId('login-password'), { target: { value: 'wrong-password' } })
    fireEvent.click(screen.getByTestId('login-submit'))

    expect(await screen.findByTestId('login-error')).toHaveTextContent('Invalid email or password')
  })
})

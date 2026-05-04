import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'

import { authApi } from '../lib/api'
import RegisterPage from './RegisterPage'

describe('RegisterPage', () => {
  it('registers a user and sends them to the dashboard', async () => {
    vi.spyOn(authApi, 'register').mockResolvedValue({
      access_token: 'access-token',
      refresh_token: 'refresh-token',
      token_type: 'bearer',
      expires_in: 3600,
    })

    render(
      <MemoryRouter initialEntries={['/register']}>
        <Routes>
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/dashboard" element={<div>Dashboard Home</div>} />
        </Routes>
      </MemoryRouter>
    )

    fireEvent.change(screen.getByTestId('register-name'), { target: { value: 'Dr. Jane Smith' } })
    fireEvent.change(screen.getByTestId('register-email'), { target: { value: 'surgeon@example.com' } })
    fireEvent.change(screen.getByTestId('register-password'), { target: { value: 'secure-password' } })
    fireEvent.change(screen.getByTestId('register-institution'), { target: { value: 'Test Hospital' } })
    fireEvent.change(screen.getByTestId('register-specialty'), { target: { value: 'OMFS' } })
    fireEvent.click(screen.getByTestId('register-submit'))

    await waitFor(() => {
      expect(screen.getByText('Dashboard Home')).toBeInTheDocument()
    })
  })

  it('shows registration failures inline', async () => {
    vi.spyOn(authApi, 'register').mockRejectedValue(new Error('Email already registered'))

    render(
      <MemoryRouter>
        <RegisterPage />
      </MemoryRouter>
    )

    fireEvent.change(screen.getByTestId('register-name'), { target: { value: 'Dr. Jane Smith' } })
    fireEvent.change(screen.getByTestId('register-email'), { target: { value: 'surgeon@example.com' } })
    fireEvent.change(screen.getByTestId('register-password'), { target: { value: 'secure-password' } })
    fireEvent.click(screen.getByTestId('register-submit'))

    expect(await screen.findByTestId('register-error')).toHaveTextContent('Email already registered')
  })
})

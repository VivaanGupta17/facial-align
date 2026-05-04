import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import AuthGuard from './components/auth/AuthGuard'
import { PageLoading } from './components/common/LoadingOverlay'

const LoginPage = lazy(() => import('./pages/LoginPage'))
const RegisterPage = lazy(() => import('./pages/RegisterPage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const CaseListPage = lazy(() => import('./pages/CaseListPage'))
const CaseDetailPage = lazy(() => import('./pages/CaseDetailPage'))
const UploadPage = lazy(() => import('./pages/UploadPage'))
const StudiesPage = lazy(() => import('./pages/StudiesPage'))
const ModelsPage = lazy(() => import('./pages/ModelsPage'))
const ComputePage = lazy(() => import('./pages/ComputePage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

function FullScreenRouteFallback({ label }: { label: string }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-6">
      <PageLoading label={label} />
    </div>
  )
}

function AppRouteFallback({ label }: { label: string }) {
  return (
    <div className="flex min-h-[50vh] items-center justify-center px-6 py-12">
      <PageLoading label={label} />
    </div>
  )
}

function renderPublicRoute(element: ReactNode, label: string) {
  return <Suspense fallback={<FullScreenRouteFallback label={label} />}>{element}</Suspense>
}

function renderAppRoute(element: ReactNode, label: string) {
  return <Suspense fallback={<AppRouteFallback label={label} />}>{element}</Suspense>
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={renderPublicRoute(<LoginPage />, 'Preparing sign in...')} />
        <Route path="/register" element={renderPublicRoute(<RegisterPage />, 'Preparing registration...')} />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            <AuthGuard>
              <AppShell />
            </AuthGuard>
          }
        >
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={renderAppRoute(<DashboardPage />, 'Loading dashboard...')} />
          <Route path="cases" element={renderAppRoute(<CaseListPage />, 'Loading case registry...')} />
          <Route path="cases/:caseId" element={renderAppRoute(<CaseDetailPage />, 'Opening case workspace...')} />
          <Route path="upload" element={renderAppRoute(<UploadPage />, 'Preparing upload workflow...')} />
          <Route path="studies" element={renderAppRoute(<StudiesPage />, 'Loading studies...')} />
          <Route path="models" element={renderAppRoute(<ModelsPage />, 'Loading model registry...')} />
          <Route path="compute" element={renderAppRoute(<ComputePage />, 'Loading compute status...')} />
          <Route path="settings" element={renderAppRoute(<SettingsPage />, 'Loading settings...')} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import ToastContainer from '../common/Toast'

export default function AppShell() {
  return (
    <div className="app-shell-background flex h-screen overflow-hidden" data-testid="app-shell">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content area */}
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-grid opacity-[0.025]" />
        <TopBar />
        <main
          className="relative z-10 flex-1 overflow-auto"
          data-testid="main-content"
        >
          <Outlet />
        </main>
      </div>

      {/* Toast notifications */}
      <ToastContainer />
    </div>
  )
}

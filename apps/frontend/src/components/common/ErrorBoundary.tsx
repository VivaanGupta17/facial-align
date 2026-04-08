/**
 * React Error Boundary for Facial Align.
 *
 * Catches render-time and lifecycle errors in child trees.
 * Provides fallback UI with severity levels, context logging, and retry.
 */

import React, { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw, Bug, Wifi, ShieldAlert } from 'lucide-react'
import { formatError, logError, type FormattedError, type ErrorContext } from '../../lib/errors'

// =============================================================================
// Severity levels
// =============================================================================

export type ErrorSeverity = 'fatal' | 'error' | 'warning'

// =============================================================================
// Props & State
// =============================================================================

interface ErrorBoundaryProps {
  children: ReactNode
  /** Context passed to the error logger (component name, current case, etc.) */
  context?: ErrorContext
  /** Severity level — controls how the fallback is displayed */
  severity?: ErrorSeverity
  /** Render a custom fallback. Receives formatted error + reset callback. */
  fallback?: (error: FormattedError, reset: () => void) => ReactNode
  /** Called after the boundary catches an error */
  onError?: (error: Error, info: ErrorInfo) => void
  /** If true, render a compact inline error rather than a full-page card */
  inline?: boolean
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
  formatted: FormattedError | null
  errorInfo: ErrorInfo | null
}

// =============================================================================
// Error Boundary Class Component
// =============================================================================

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null,
    formatted: null,
    errorInfo: null,
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
      formatted: formatError(error),
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    const ctx: ErrorContext = {
      ...this.props.context,
      extra: {
        ...this.props.context?.extra,
        componentStack: info.componentStack,
      },
    }
    logError(error, ctx)
    this.setState({ errorInfo: info })
    this.props.onError?.(error, info)
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      formatted: null,
      errorInfo: null,
    })
  }

  render(): ReactNode {
    const { hasError, formatted } = this.state
    const { children, fallback, inline, severity = 'error' } = this.props

    if (!hasError || !formatted) return children

    if (fallback) return fallback(formatted, this.handleReset)

    if (inline) {
      return (
        <InlineErrorFallback
          formatted={formatted}
          onReset={this.handleReset}
          severity={severity}
        />
      )
    }

    return (
      <FullErrorFallback
        formatted={formatted}
        onReset={this.handleReset}
        severity={severity}
        errorInfo={this.state.errorInfo}
      />
    )
  }
}

// =============================================================================
// Full-page fallback
// =============================================================================

const SEVERITY_STYLES: Record<ErrorSeverity, {
  border: string
  iconBg: string
  iconColor: string
  badge: string
  icon: ReactNode
}> = {
  fatal: {
    border: 'border-red-800',
    iconBg: 'bg-red-950',
    iconColor: 'text-red-400',
    badge: 'bg-red-950 text-red-400 border-red-800',
    icon: <ShieldAlert size={28} />,
  },
  error: {
    border: 'border-slate-700',
    iconBg: 'bg-slate-800',
    iconColor: 'text-red-400',
    badge: 'bg-red-950/60 text-red-400 border-red-900',
    icon: <AlertTriangle size={28} />,
  },
  warning: {
    border: 'border-amber-900',
    iconBg: 'bg-amber-950',
    iconColor: 'text-amber-400',
    badge: 'bg-amber-950/60 text-amber-400 border-amber-900',
    icon: <AlertTriangle size={28} />,
  },
}

interface FallbackProps {
  formatted: FormattedError
  onReset: () => void
  severity: ErrorSeverity
  errorInfo?: ErrorInfo | null
}

function FullErrorFallback({ formatted, onReset, severity, errorInfo }: FallbackProps) {
  const styles = SEVERITY_STYLES[severity]

  return (
    <div
      className="min-h-[300px] flex items-center justify-center p-8 bg-slate-900"
      data-testid="error-boundary-fallback"
    >
      <div className={`max-w-md w-full bg-slate-900 border ${styles.border} rounded-xl p-6 space-y-4`}>
        {/* Icon + title */}
        <div className="flex items-start gap-4">
          <div className={`w-12 h-12 rounded-lg ${styles.iconBg} flex items-center justify-center shrink-0 ${styles.iconColor}`}>
            {styles.icon}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-base font-semibold text-slate-100">{formatted.title}</h3>
              <span className={`text-2xs font-mono px-1.5 py-0.5 rounded border ${styles.badge}`}>
                {formatted.code}
              </span>
            </div>
            <p className="text-sm text-slate-400">{formatted.description}</p>
          </div>
        </div>

        {/* Component stack (dev only) */}
        {process.env.NODE_ENV === 'development' && errorInfo?.componentStack && (
          <details className="group">
            <summary className="flex items-center gap-1 text-xs text-slate-500 cursor-pointer hover:text-slate-300 select-none">
              <Bug size={11} />
              Component trace
            </summary>
            <pre className="mt-2 text-2xs font-mono text-slate-600 bg-slate-950 rounded p-3 overflow-auto max-h-32 whitespace-pre-wrap break-all">
              {errorInfo.componentStack}
            </pre>
          </details>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-1">
          {formatted.recoverable && (
            <button
              onClick={onReset}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium transition-colors"
              data-testid="error-boundary-retry"
            >
              <RefreshCw size={14} />
              {formatted.actionLabel ?? 'Retry'}
            </button>
          )}
          {!formatted.recoverable && (
            <p className="text-xs text-slate-500">Contact your system administrator if this issue persists.</p>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Inline compact fallback
// =============================================================================

function InlineErrorFallback({ formatted, onReset, severity }: Omit<FallbackProps, 'errorInfo'>) {
  const styles = SEVERITY_STYLES[severity]

  return (
    <div
      className={`flex items-center gap-3 px-3 py-2 rounded-lg border ${styles.border} bg-slate-900 text-sm`}
      data-testid="error-boundary-inline"
    >
      <span className={styles.iconColor}>
        <AlertTriangle size={16} />
      </span>
      <div className="flex-1 min-w-0">
        <span className="font-medium text-slate-200">{formatted.title}:</span>{' '}
        <span className="text-slate-400 truncate">{formatted.description}</span>
      </div>
      {formatted.recoverable && (
        <button
          onClick={onReset}
          className="shrink-0 flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          data-testid="error-boundary-retry-inline"
        >
          <RefreshCw size={12} />
          Retry
        </button>
      )}
    </div>
  )
}

// =============================================================================
// Network error fallback — special case for connectivity issues
// =============================================================================

interface NetworkFallbackProps {
  onRetry?: () => void
}

export function NetworkErrorFallback({ onRetry }: NetworkFallbackProps) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-4 py-12 text-center"
      data-testid="network-error-fallback"
    >
      <div className="w-14 h-14 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
        <Wifi size={24} className="text-slate-500" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-200">Connection Error</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-xs">
          Unable to reach the Facial Align server. Check your network connection.
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-medium transition-colors"
        >
          <RefreshCw size={13} />
          Retry
        </button>
      )}
    </div>
  )
}

// =============================================================================
// Convenience wrappers
// =============================================================================

/** Error boundary wrapping the surgical viewer */
export function ViewerErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      severity="error"
      context={{ component: 'Viewer3D' }}
      inline={false}
    >
      {children}
    </ErrorBoundary>
  )
}

/** Error boundary wrapping a planning panel */
export function PlanningErrorBoundary({
  children,
  caseId,
  planId,
}: {
  children: ReactNode
  caseId?: string
  planId?: string
}) {
  return (
    <ErrorBoundary
      severity="error"
      context={{ component: 'PlanningWorkspace', caseId, planId }}
    >
      {children}
    </ErrorBoundary>
  )
}

/** Lightweight inline boundary for individual UI sections */
export function SectionErrorBoundary({
  children,
  label,
}: {
  children: ReactNode
  label?: string
}) {
  return (
    <ErrorBoundary
      severity="warning"
      context={{ component: label }}
      inline
    >
      {children}
    </ErrorBoundary>
  )
}

export default ErrorBoundary

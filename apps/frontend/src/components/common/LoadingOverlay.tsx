interface SkeletonProps {
  className?: string
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      className={`bg-slate-700/60 rounded animate-pulse ${className}`}
      data-testid="skeleton"
    />
  )
}

interface LoadingOverlayProps {
  label?: string
  progress?: number
  className?: string
}

export function LoadingOverlay({ label, progress, className = '' }: LoadingOverlayProps) {
  return (
    <div className={`absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm z-50 ${className}`} data-testid="loading-overlay">
      <div className="flex flex-col items-center gap-4">
        {/* Spinner */}
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 rounded-full border-2 border-slate-700" />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyan-500 animate-spin" />
        </div>
        {label && <p className="text-sm text-slate-300 font-medium">{label}</p>}
        {progress != null && (
          <div className="w-48">
            <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-center text-xs text-slate-500 mt-1 font-mono">{Math.round(progress)}%</p>
          </div>
        )}
      </div>
    </div>
  )
}

/** Inline spinner */
export function Spinner({ size = 16, className = '' }: { size?: number; className?: string }) {
  return (
    <div
      className={`rounded-full border-2 border-transparent border-t-cyan-500 animate-spin ${className}`}
      style={{ width: size, height: size }}
      data-testid="spinner"
    />
  )
}

/** Full page loading state */
export function PageLoading({ label = 'Loading...' }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 gap-4" data-testid="page-loading">
      <div className="relative w-10 h-10">
        <div className="absolute inset-0 rounded-full border-2 border-slate-700" />
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyan-500 animate-spin" />
      </div>
      <p className="text-sm text-slate-400">{label}</p>
    </div>
  )
}

/** Empty state */
export function EmptyState({
  title,
  description,
  action,
  icon,
}: {
  title: string
  description?: string
  action?: React.ReactNode
  icon?: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 py-12 px-6 text-center" data-testid="empty-state">
      {icon && <div className="mb-4 text-slate-600">{icon}</div>}
      <h3 className="text-base font-semibold text-slate-300 mb-1">{title}</h3>
      {description && <p className="text-sm text-slate-500 max-w-sm">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

/** Error state */
export function ErrorState({
  title = 'Something went wrong',
  description,
  onRetry,
}: {
  title?: string
  description?: string
  onRetry?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 py-12 px-6 text-center" data-testid="error-state">
      <div className="w-10 h-10 rounded-full bg-red-950 border border-red-800 flex items-center justify-center mb-4">
        <span className="text-red-400 text-lg">!</span>
      </div>
      <h3 className="text-base font-semibold text-red-400 mb-1">{title}</h3>
      {description && <p className="text-sm text-slate-500 max-w-sm">{description}</p>}
      {onRetry && (
        <button onClick={onRetry} className="mt-4 btn-secondary">
          Try again
        </button>
      )}
    </div>
  )
}

/** Table skeleton rows */
export function TableSkeleton({ rows = 5, cols = 5 }: { rows?: number; cols?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i}>
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j} className="px-4 py-3">
              <Skeleton className={`h-4 ${j === 0 ? 'w-24' : j === cols - 1 ? 'w-16' : 'w-full max-w-32'}`} />
            </td>
          ))}
        </tr>
      ))}
    </>
  )
}

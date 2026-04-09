interface JobProgressBarProps {
  stage: string
  progress: number
  className?: string
}

export default function JobProgressBar({ stage, progress, className = '' }: JobProgressBarProps) {
  const clampedProgress = Math.min(100, Math.max(0, progress))

  return (
    <div className={`w-full ${className}`} data-testid="job-progress-bar">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-cyan-400 capitalize">
          {stage.replace(/_/g, ' ')}
        </span>
        <span className="text-xs font-mono text-slate-400">{Math.round(clampedProgress)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all duration-500 ease-out relative"
          style={{ width: `${clampedProgress}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
        </div>
      </div>
    </div>
  )
}

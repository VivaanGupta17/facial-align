interface ConfidenceBarProps {
  value: number // 0–1
  threshold?: number // 0–1
  showLabel?: boolean
  showValue?: boolean
  height?: 'sm' | 'md' | 'lg'
  className?: string
}

function getConfidenceColor(value: number, threshold: number): { bar: string; text: string } {
  if (value >= 0.9) return { bar: 'bg-emerald-500', text: 'text-emerald-400' }
  if (value >= threshold) return { bar: 'bg-cyan-500', text: 'text-cyan-400' }
  if (value >= threshold - 0.1) return { bar: 'bg-amber-500', text: 'text-amber-400' }
  return { bar: 'bg-red-500', text: 'text-red-400' }
}

export default function ConfidenceBar({
  value,
  threshold = 0.85,
  showLabel = false,
  showValue = true,
  height = 'sm',
  className = '',
}: ConfidenceBarProps) {
  const pct = Math.round(value * 100)
  const { bar, text } = getConfidenceColor(value, threshold)
  const thresholdPct = Math.round(threshold * 100)

  const heightClass = { sm: 'h-1.5', md: 'h-2', lg: 'h-3' }[height]

  return (
    <div className={`${className}`} data-testid="confidence-bar">
      {(showLabel || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {showLabel && <span className="text-xs text-slate-400">Confidence</span>}
          {showValue && (
            <span className={`text-xs font-mono font-semibold ml-auto ${text}`}>
              {pct}%
            </span>
          )}
        </div>
      )}
      <div className={`relative ${heightClass} rounded-full bg-slate-700 overflow-hidden`}>
        {/* Fill */}
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ${bar}`}
          style={{ width: `${pct}%` }}
        />
        {/* Threshold marker */}
        {threshold > 0 && (
          <div
            className="absolute inset-y-0 w-px bg-slate-400/50"
            style={{ left: `${thresholdPct}%` }}
            title={`Threshold: ${thresholdPct}%`}
          />
        )}
      </div>
    </div>
  )
}

/** Inline confidence indicator (small badge-style) */
export function ConfidenceBadge({ value, threshold = 0.85 }: { value: number; threshold?: number }) {
  const pct = Math.round(value * 100)
  const { text } = getConfidenceColor(value, threshold)
  const passes = value >= threshold
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs font-mono font-semibold ${text}`}
      data-testid="confidence-badge"
    >
      <span className={`w-1.5 h-1.5 rounded-full ${passes ? 'bg-emerald-500' : 'bg-red-500'}`} />
      {pct}%
    </span>
  )
}

/** Large confidence ring display */
export function ConfidenceRing({ value, size = 80 }: { value: number; size?: number }) {
  const pct = Math.round(value * 100)
  const radius = (size - 8) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDash = (value * circumference).toFixed(2)
  const { text } = getConfidenceColor(value, 0.85)

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }} data-testid="confidence-ring">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="#334155" strokeWidth={6} fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          stroke="currentColor" strokeWidth={6} fill="none"
          strokeDasharray={`${strokeDash} ${circumference}`}
          strokeLinecap="round"
          className={value >= 0.9 ? 'text-emerald-500' : value >= 0.85 ? 'text-cyan-500' : value >= 0.75 ? 'text-amber-500' : 'text-red-500'}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-base font-mono font-bold ${text}`}>{pct}</span>
        <span className="text-2xs text-slate-500">%</span>
      </div>
    </div>
  )
}

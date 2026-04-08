import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  trend?: number // positive = up, negative = down
  trendLabel?: string
  icon?: React.ReactNode
  color?: 'default' | 'cyan' | 'green' | 'red' | 'amber'
  size?: 'sm' | 'md' | 'lg'
  className?: string
  onClick?: () => void
  subtitle?: string
}

const colorMap = {
  default: { value: 'text-slate-100', label: 'text-slate-400', icon: 'text-slate-400', bg: '' },
  cyan: { value: 'text-cyan-300', label: 'text-cyan-500', icon: 'text-cyan-400', bg: 'bg-cyan-950/40 border-cyan-900/50' },
  green: { value: 'text-emerald-300', label: 'text-emerald-500', icon: 'text-emerald-400', bg: 'bg-emerald-950/40 border-emerald-900/50' },
  red: { value: 'text-red-300', label: 'text-red-500', icon: 'text-red-400', bg: 'bg-red-950/40 border-red-900/50' },
  amber: { value: 'text-amber-300', label: 'text-amber-500', icon: 'text-amber-400', bg: 'bg-amber-950/40 border-amber-900/50' },
}

export default function MetricCard({
  label,
  value,
  unit,
  trend,
  trendLabel,
  icon,
  color = 'default',
  size = 'md',
  className = '',
  onClick,
  subtitle,
}: MetricCardProps) {
  const c = colorMap[color]
  const sizeClass = size === 'sm' ? 'p-3' : size === 'lg' ? 'p-5' : 'p-4'
  const valueSize = size === 'sm' ? 'text-xl' : size === 'lg' ? 'text-4xl' : 'text-2xl'

  const TrendIcon = trend == null ? null : trend > 0 ? TrendingUp : trend < 0 ? TrendingDown : Minus
  const trendColor = trend == null ? '' : trend > 0 ? 'text-emerald-400' : trend < 0 ? 'text-red-400' : 'text-slate-400'

  return (
    <div
      className={`bg-slate-800 border border-slate-700 rounded-lg ${sizeClass} ${c.bg ? `${c.bg} border` : ''} ${onClick ? 'cursor-pointer hover:bg-slate-750 transition-colors' : ''} ${className}`}
      onClick={onClick}
      data-testid="metric-card"
    >
      {/* Header: icon + label */}
      <div className="flex items-start justify-between mb-2">
        <span className={`text-xs font-semibold uppercase tracking-wider ${c.label}`}>{label}</span>
        {icon && <span className={c.icon}>{icon}</span>}
      </div>

      {/* Value */}
      <div className="flex items-end gap-1.5">
        <span className={`font-mono font-bold ${valueSize} ${c.value} leading-none`}>{value}</span>
        {unit && <span className="text-slate-500 text-xs font-mono pb-0.5">{unit}</span>}
      </div>

      {/* Subtitle */}
      {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}

      {/* Trend */}
      {(trend != null || trendLabel) && (
        <div className={`flex items-center gap-1 mt-2 text-xs ${trendColor}`}>
          {TrendIcon && <TrendIcon size={12} />}
          {trendLabel && <span>{trendLabel}</span>}
        </div>
      )}
    </div>
  )
}

/** Inline metric display (no card) */
export function MetricInline({
  label,
  value,
  unit,
  ideal,
  withinRange,
}: {
  label: string
  value: number | string
  unit?: string
  ideal?: string
  withinRange?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-b-0" data-testid="metric-inline">
      <span className="text-xs text-slate-400">{label}</span>
      <div className="flex items-center gap-2">
        {ideal && <span className="text-2xs text-slate-600 font-mono">(ideal: {ideal})</span>}
        <span className={`font-mono text-sm font-semibold ${withinRange === true ? 'text-emerald-400' : withinRange === false ? 'text-red-400' : 'text-slate-100'}`}>
          {value}{unit && <span className="text-slate-500 text-xs"> {unit}</span>}
        </span>
      </div>
    </div>
  )
}

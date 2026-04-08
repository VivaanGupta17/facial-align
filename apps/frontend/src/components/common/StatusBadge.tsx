import type { CaseStatus } from '../../types/medical'

interface StatusConfig {
  label: string
  className: string
  dot: string
}

const STATUS_CONFIG: Record<CaseStatus, StatusConfig> = {
  pending_upload: { label: 'Pending Upload', className: 'bg-slate-800 text-slate-400 border-slate-700', dot: 'bg-slate-400' },
  uploading: { label: 'Uploading', className: 'bg-blue-950 text-blue-400 border-blue-800', dot: 'bg-blue-400' },
  processing: { label: 'Processing', className: 'bg-blue-950 text-blue-400 border-blue-800', dot: 'bg-blue-400 animate-pulse' },
  segmentation_in_progress: { label: 'Segmenting', className: 'bg-indigo-950 text-indigo-400 border-indigo-800', dot: 'bg-indigo-400 animate-pulse' },
  segmentation_review: { label: 'Seg. Review', className: 'bg-purple-950 text-purple-400 border-purple-800', dot: 'bg-purple-400' },
  planning: { label: 'Planning', className: 'bg-cyan-950 text-cyan-400 border-cyan-800', dot: 'bg-cyan-400' },
  review: { label: 'In Review', className: 'bg-amber-950 text-amber-400 border-amber-800', dot: 'bg-amber-400' },
  approved: { label: 'Approved', className: 'bg-emerald-950 text-emerald-400 border-emerald-800', dot: 'bg-emerald-400' },
  rejected: { label: 'Rejected', className: 'bg-red-950 text-red-400 border-red-800', dot: 'bg-red-400' },
  completed: { label: 'Completed', className: 'bg-emerald-950 text-emerald-300 border-emerald-800', dot: 'bg-emerald-300' },
  archived: { label: 'Archived', className: 'bg-slate-900 text-slate-500 border-slate-700', dot: 'bg-slate-500' },
}

interface StatusBadgeProps {
  status: CaseStatus
  size?: 'sm' | 'md'
  showDot?: boolean
  className?: string
}

export default function StatusBadge({ status, size = 'md', showDot = true, className = '' }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status]
  const sizeClass = size === 'sm' ? 'text-2xs px-1.5 py-0.5' : 'text-xs px-2 py-1'

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-semibold rounded border ${sizeClass} ${config.className} ${className}`}
      data-testid={`status-badge-${status}`}
    >
      {showDot && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${config.dot}`} />}
      {config.label}
    </span>
  )
}

/** Priority badge */
export function PriorityBadge({ priority }: { priority: 'routine' | 'urgent' | 'stat' }) {
  const config = {
    routine: 'text-slate-400 bg-slate-800 border-slate-700',
    urgent: 'text-amber-400 bg-amber-950 border-amber-800',
    stat: 'text-red-400 bg-red-950 border-red-800',
  }[priority]

  return (
    <span className={`inline-flex items-center text-2xs font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border ${config}`} data-testid={`priority-badge-${priority}`}>
      {priority}
    </span>
  )
}

/**
 * Reusable EmptyState component for Facial Align.
 *
 * Used when lists, views, or panels have no data to display.
 * Supports icon, title, description, and an optional action button.
 */

import { type ReactNode } from 'react'
import {
  FolderOpen,
  ScanLine,
  Cpu,
  FileX,
  ClipboardList,
  Layers,
} from 'lucide-react'

// =============================================================================
// Preset icons by context
// =============================================================================

export const EMPTY_STATE_ICONS = {
  cases: FolderOpen,
  segmentation: ScanLine,
  planning: Cpu,
  noFile: FileX,
  review: ClipboardList,
  structures: Layers,
} as const

export type EmptyStatePreset = keyof typeof EMPTY_STATE_ICONS

// =============================================================================
// Size variants
// =============================================================================

const SIZE_CONFIG = {
  sm: {
    wrapper: 'py-6',
    iconContainer: 'w-10 h-10',
    iconSize: 18,
    title: 'text-sm font-semibold',
    desc: 'text-xs',
    btn: 'px-3 py-1.5 text-xs',
  },
  md: {
    wrapper: 'py-10',
    iconContainer: 'w-14 h-14',
    iconSize: 24,
    title: 'text-base font-semibold',
    desc: 'text-sm',
    btn: 'px-4 py-2 text-sm',
  },
  lg: {
    wrapper: 'py-16',
    iconContainer: 'w-20 h-20',
    iconSize: 32,
    title: 'text-lg font-semibold',
    desc: 'text-sm',
    btn: 'px-5 py-2.5 text-sm',
  },
} as const

// =============================================================================
// Props
// =============================================================================

interface EmptyStateProps {
  /** Predefined icon key, custom Lucide icon component, or an SVG ReactNode */
  icon?: EmptyStatePreset | ReactNode
  /** Bold heading */
  title: string
  /** Supporting text */
  description?: string
  /** Primary action button label */
  actionLabel?: string
  /** Called when the action button is clicked */
  onAction?: () => void
  /** Whether the action button is in a loading state */
  actionLoading?: boolean
  /** Secondary/muted action link label */
  secondaryLabel?: string
  /** Called when the secondary link is clicked */
  onSecondary?: () => void
  /** Size variant */
  size?: keyof typeof SIZE_CONFIG
  /** Additional Tailwind classes on the container */
  className?: string
  /** Test ID for automated tests */
  testId?: string
}

// =============================================================================
// Component
// =============================================================================

export default function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  actionLoading = false,
  secondaryLabel,
  onSecondary,
  size = 'md',
  className = '',
  testId,
}: EmptyStateProps) {
  const cfg = SIZE_CONFIG[size]
  const IconNode = resolveIcon(icon, cfg.iconSize)

  return (
    <div
      className={`flex flex-col items-center justify-center text-center ${cfg.wrapper} px-6 ${className}`}
      data-testid={testId ?? 'empty-state'}
    >
      {/* Icon container */}
      {IconNode && (
        <div
          className={`${cfg.iconContainer} rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-4 text-slate-500`}
        >
          {IconNode}
        </div>
      )}

      {/* Text */}
      <h3 className={`${cfg.title} text-slate-200 mb-1`}>{title}</h3>
      {description && (
        <p className={`${cfg.desc} text-slate-500 max-w-xs leading-relaxed`}>{description}</p>
      )}

      {/* Actions */}
      {(actionLabel || secondaryLabel) && (
        <div className="flex flex-col items-center gap-2 mt-5">
          {actionLabel && onAction && (
            <button
              onClick={onAction}
              disabled={actionLoading}
              className={`inline-flex items-center justify-center gap-2 ${cfg.btn} rounded-lg bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium transition-colors`}
              data-testid="empty-state-action"
            >
              {actionLoading ? (
                <span className="inline-block w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              ) : null}
              {actionLabel}
            </button>
          )}
          {secondaryLabel && onSecondary && (
            <button
              onClick={onSecondary}
              className={`${cfg.btn} text-slate-400 hover:text-slate-200 transition-colors`}
              data-testid="empty-state-secondary"
            >
              {secondaryLabel}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Preset instances — convenience components for common use cases
// =============================================================================

/** No cases have been created yet */
export function NoCasesEmpty({
  onCreateCase,
}: {
  onCreateCase?: () => void
}) {
  return (
    <EmptyState
      icon="cases"
      title="No Cases Yet"
      description="Create a new surgical case by uploading a DICOM CT series."
      actionLabel="Upload DICOM"
      onAction={onCreateCase}
      testId="empty-cases"
    />
  )
}

/** No segmentation results available */
export function NoSegmentationEmpty({
  onRunSegmentation,
  isRunning,
}: {
  onRunSegmentation?: () => void
  isRunning?: boolean
}) {
  return (
    <EmptyState
      icon="segmentation"
      title="No Segmentation Results"
      description="Run AI segmentation to automatically identify craniofacial structures from the CT volume."
      actionLabel="Run Segmentation"
      onAction={onRunSegmentation}
      actionLoading={isRunning}
      testId="empty-segmentation"
    />
  )
}

/** No reduction plan generated */
export function NoPlanEmpty({
  onGeneratePlan,
  isGenerating,
}: {
  onGeneratePlan?: () => void
  isGenerating?: boolean
}) {
  return (
    <EmptyState
      icon="planning"
      title="No Reduction Plan"
      description="Generate an AI-powered reduction plan to begin surgical planning."
      actionLabel="Generate Plan"
      onAction={onGeneratePlan}
      actionLoading={isGenerating}
      testId="empty-plan"
    />
  )
}

/** Measurement list is empty */
export function NoMeasurementsEmpty() {
  return (
    <EmptyState
      icon="structures"
      size="sm"
      title="No Measurements"
      description="Use the measurement tools to add distance or angle annotations."
      testId="empty-measurements"
    />
  )
}

/** Review checklist has no items */
export function NoReviewEmpty() {
  return (
    <EmptyState
      icon="review"
      title="No Review Items"
      description="The review checklist for this case is empty."
      testId="empty-review"
    />
  )
}

// =============================================================================
// Utility — resolve icon prop to a ReactNode
// =============================================================================

function resolveIcon(
  icon: EmptyStateProps['icon'],
  size: number
): ReactNode | null {
  if (!icon) return null

  if (typeof icon === 'string') {
    const IconComp = EMPTY_STATE_ICONS[icon as EmptyStatePreset]
    if (!IconComp) return null
    return <IconComp size={size} />
  }

  // ReactNode or React element passed directly
  return icon
}

import type { ReactNode } from 'react'

interface HeaderChip {
  label: string
  tone?: 'neutral' | 'info' | 'success' | 'warning' | 'danger'
  icon?: ReactNode
}

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  chips?: HeaderChip[]
  actions?: ReactNode
}

const toneClasses: Record<NonNullable<HeaderChip['tone']>, string> = {
  neutral: 'chip chip-neutral',
  info: 'chip chip-info',
  success: 'chip chip-success',
  warning: 'chip chip-warning',
  danger: 'chip chip-danger',
}

export default function PageHeader({
  eyebrow,
  title,
  description,
  chips = [],
  actions,
}: PageHeaderProps) {
  return (
    <div className="page-header-shell">
      <div className="page-header-copy">
        {eyebrow && <p className="page-header-eyebrow">{eyebrow}</p>}
        <h1 className="page-header-title">{title}</h1>
        {description && <p className="page-header-description">{description}</p>}
        {chips.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {chips.map((chip) => (
              <span
                key={chip.label}
                className={toneClasses[chip.tone ?? 'neutral']}
              >
                {chip.icon}
                {chip.label}
              </span>
            ))}
          </div>
        )}
      </div>

      {actions && (
        <div className="page-header-actions">
          {actions}
        </div>
      )}
    </div>
  )
}

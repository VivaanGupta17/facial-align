import { CheckCircle, XCircle, Cpu, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import { ConfidenceRing } from './ConfidenceBar'

interface RecommendationCardProps {
  title?: string
  recommendation: string
  confidence: number // 0–1
  modelName?: string
  modelVersion?: string
  onAccept?: () => void
  onReject?: () => void
  acceptLabel?: string
  rejectLabel?: string
  isLoading?: boolean
  className?: string
  expanded?: boolean
  metadata?: Array<{ label: string; value: string | number }>
}

export default function RecommendationCard({
  title = 'AI Recommendation',
  recommendation,
  confidence,
  modelName,
  modelVersion,
  onAccept,
  onReject,
  acceptLabel = 'Accept',
  rejectLabel = 'Reject',
  isLoading = false,
  className = '',
  expanded: defaultExpanded = true,
  metadata,
}: RecommendationCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)

  const confidenceLabel = confidence >= 0.9 ? 'Very High' : confidence >= 0.8 ? 'High' : confidence >= 0.7 ? 'Moderate' : confidence >= 0.6 ? 'Low' : 'Very Low'

  return (
    <div
      className={`border border-cyan-900/60 rounded-lg bg-gradient-to-b from-cyan-950/50 to-slate-800/80 ${className}`}
      style={{ boxShadow: '0 0 20px rgba(6, 182, 212, 0.08)' }}
      data-testid="recommendation-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-cyan-900/40">
        <div className="w-6 h-6 rounded flex items-center justify-center bg-cyan-900/60">
          <Cpu size={13} className="text-cyan-400" />
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-xs font-semibold text-cyan-400">{title}</span>
          {modelName && (
            <span className="ml-2 text-2xs font-mono text-slate-500">{modelName}{modelVersion ? ` v${modelVersion}` : ''}</span>
          )}
        </div>
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-slate-500 hover:text-slate-300"
          data-testid="rec-card-toggle"
        >
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {expanded && (
        <div className="p-4 space-y-3 animate-fade-in">
          {/* Confidence + text */}
          <div className="flex gap-4 items-start">
            <ConfidenceRing value={confidence} size={60} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-semibold text-slate-300">{confidenceLabel} Confidence</span>
                {confidence < 0.75 && (
                  <span className="flex items-center gap-1 text-2xs text-amber-400">
                    <Info size={10} /> Manual review recommended
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">{recommendation}</p>
            </div>
          </div>

          {/* Metadata */}
          {metadata && metadata.length > 0 && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-cyan-900/30 pt-3">
              {metadata.map(m => (
                <div key={m.label} className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">{m.label}</span>
                  <span className="font-mono text-slate-300">{m.value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Actions */}
          {(onAccept || onReject) && (
            <div className="flex gap-2 pt-1 border-t border-cyan-900/30">
              {onAccept && (
                <button
                  onClick={onAccept}
                  disabled={isLoading}
                  className="flex items-center gap-1.5 btn-success flex-1 justify-center text-xs"
                  data-testid="rec-accept-btn"
                >
                  <CheckCircle size={13} />
                  {acceptLabel}
                </button>
              )}
              {onReject && (
                <button
                  onClick={onReject}
                  disabled={isLoading}
                  className="flex items-center gap-1.5 btn-secondary flex-1 justify-center text-xs"
                  data-testid="rec-reject-btn"
                >
                  <XCircle size={13} />
                  {rejectLabel}
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

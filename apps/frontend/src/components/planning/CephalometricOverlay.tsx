/**
 * CephalometricOverlay — SVG overlay showing cephalometric landmarks
 * and standard measurements (SNA, SNB, ANB, etc.) on a lateral view.
 *
 * Features:
 *  - Landmark points rendered on a lateral-view canvas
 *  - Connecting lines for standard cephalometric analyses
 *  - Measurement panel with colour-coded normal/borderline/abnormal
 *  - Measurement group toggles
 */

import { useState, useCallback } from 'react'
import { Eye, EyeOff, Info } from 'lucide-react'

// =============================================================================
// Types
// =============================================================================

interface CephPoint {
  id: string
  label: string
  abbr: string      // Standard cephalometric abbreviation
  x: number         // SVG coordinate (0–400)
  y: number
  description: string
}

interface CephMeasurement {
  id: string
  label: string
  value: number
  unit: 'deg' | 'mm'
  normalMin: number
  normalMax: number
  borderlineRange: number   // ± this from normal bounds = borderline
  group: MeasurementGroup
  description: string
  /** IDs of the two landmark points to draw the lines through */
  linePoints?: [string, string][]
}

type MeasurementGroup = 'skeletal' | 'dental' | 'soft_tissue' | 'vertical'

type ClinicalStatus = 'normal' | 'borderline' | 'abnormal'

// =============================================================================
// Standard cephalometric landmark positions (lateral skull silhouette)
// Coordinates are within a 400×500 SVG viewBox representing a lateral skull.
// =============================================================================

const LANDMARKS: CephPoint[] = [
  { id: 'S',  abbr: 'S',  label: 'Sella',           x: 198, y: 82,  description: 'Center of sella turcica' },
  { id: 'N',  abbr: 'N',  label: 'Nasion',          x: 162, y: 110, description: 'Fronto-nasal suture' },
  { id: 'A',  abbr: 'A',  label: 'A-Point',         x: 148, y: 196, description: 'Deepest point on premaxilla' },
  { id: 'B',  abbr: 'B',  label: 'B-Point',         x: 152, y: 258, description: 'Deepest point on symphysis' },
  { id: 'Pg', abbr: 'Pg', label: 'Pogonion',        x: 155, y: 278, description: 'Most anterior point of chin' },
  { id: 'Me', abbr: 'Me', label: 'Menton',          x: 155, y: 296, description: 'Most inferior point of mandible' },
  { id: 'Go', abbr: 'Go', label: 'Gonion',          x: 240, y: 284, description: 'Most posterior-inferior point of mandible' },
  { id: 'Ar', abbr: 'Ar', label: 'Articulare',      x: 242, y: 108, description: 'Intersection of posterior cranial base and condyle' },
  { id: 'Or', abbr: 'Or', label: 'Orbitale',        x: 126, y: 128, description: 'Most inferior point of orbit' },
  { id: 'Po', abbr: 'Po', label: 'Porion',          x: 234, y: 118, description: 'Most superior point of external auditory meatus' },
  { id: 'ANS',abbr: 'ANS',label: 'ANS',             x: 140, y: 188, description: 'Anterior nasal spine tip' },
  { id: 'PNS',abbr: 'PNS',label: 'PNS',             x: 208, y: 184, description: 'Posterior nasal spine tip' },
  { id: 'U1', abbr: 'U1', label: 'Upper Incisor',   x: 156, y: 216, description: 'Tip of most prominent upper incisor' },
  { id: 'L1', abbr: 'L1', label: 'Lower Incisor',   x: 160, y: 226, description: 'Tip of most prominent lower incisor' },
  { id: 'Ls', abbr: 'Ls', label: 'Upper Lip',       x: 138, y: 218, description: 'Most anterior point of upper lip' },
  { id: 'Li', abbr: 'Li', label: 'Lower Lip',       x: 140, y: 232, description: 'Most anterior point of lower lip' },
  { id: 'Pog', abbr: 'Pog\'', label: 'Soft Tissue Pogonion', x: 136, y: 272, description: 'Most anterior point of soft tissue chin' },
]

const LANDMARK_MAP = Object.fromEntries(LANDMARKS.map(l => [l.id, l]))

// =============================================================================
// Standard measurements with normative data
// =============================================================================

const MEASUREMENTS: CephMeasurement[] = [
  // Skeletal
  {
    id: 'sna', label: 'SNA', value: 82.4, unit: 'deg',
    normalMin: 80, normalMax: 84, borderlineRange: 2, group: 'skeletal',
    description: 'Maxillary sagittal position relative to cranial base',
    linePoints: [['S', 'N'], ['N', 'A']],
  },
  {
    id: 'snb', label: 'SNB', value: 78.8, unit: 'deg',
    normalMin: 78, normalMax: 82, borderlineRange: 2, group: 'skeletal',
    description: 'Mandibular sagittal position relative to cranial base',
    linePoints: [['S', 'N'], ['N', 'B']],
  },
  {
    id: 'anb', label: 'ANB', value: 3.6, unit: 'deg',
    normalMin: 1, normalMax: 4, borderlineRange: 1, group: 'skeletal',
    description: 'Maxillo-mandibular discrepancy (2–4° = Class I)',
    linePoints: [['A', 'N'], ['N', 'B']],
  },
  {
    id: 'sn_mp', label: 'SN-MP', value: 34.2, unit: 'deg',
    normalMin: 28, normalMax: 38, borderlineRange: 3, group: 'skeletal',
    description: 'Mandibular plane angle relative to sella-nasion',
    linePoints: [['S', 'N'], ['Go', 'Me']],
  },
  // Dental
  {
    id: 'u1_sn', label: 'U1-SN', value: 108.1, unit: 'deg',
    normalMin: 100, normalMax: 114, borderlineRange: 4, group: 'dental',
    description: 'Upper incisor inclination to sella-nasion plane',
    linePoints: [['S', 'N'], ['U1', 'A']],
  },
  {
    id: 'impa', label: 'IMPA', value: 94.5, unit: 'deg',
    normalMin: 87, normalMax: 99, borderlineRange: 4, group: 'dental',
    description: 'Lower incisor inclination to mandibular plane (Tweed)',
    linePoints: [['Go', 'Me'], ['L1', 'B']],
  },
  {
    id: 'interincisal', label: 'Interincisal', value: 122.3, unit: 'deg',
    normalMin: 120, normalMax: 135, borderlineRange: 5, group: 'dental',
    description: 'Angle between long axes of upper and lower incisors',
    linePoints: [['U1', 'A'], ['L1', 'B']],
  },
  // Vertical
  {
    id: 'ans_me', label: 'ANS-Me', value: 66.2, unit: 'mm',
    normalMin: 60, normalMax: 72, borderlineRange: 4, group: 'vertical',
    description: 'Lower facial height (ANS to Menton)',
    linePoints: [['ANS', 'Me']],
  },
  {
    id: 'n_ans', label: 'N-ANS', value: 54.8, unit: 'mm',
    normalMin: 50, normalMax: 58, borderlineRange: 3, group: 'vertical',
    description: 'Upper facial height (Nasion to ANS)',
    linePoints: [['N', 'ANS']],
  },
  // Soft tissue
  {
    id: 'e_line_ul', label: 'UL-E line', value: -1.2, unit: 'mm',
    normalMin: -4, normalMax: 0, borderlineRange: 1, group: 'soft_tissue',
    description: 'Upper lip position relative to Ricketts E-plane',
  },
  {
    id: 'e_line_ll', label: 'LL-E line', value: 0.4, unit: 'mm',
    normalMin: -2, normalMax: 2, borderlineRange: 1, group: 'soft_tissue',
    description: 'Lower lip position relative to Ricketts E-plane',
  },
]

// =============================================================================
// Clinical status helper
// =============================================================================

function getClinicalStatus(m: CephMeasurement): ClinicalStatus {
  const { value, normalMin, normalMax, borderlineRange } = m
  if (value >= normalMin && value <= normalMax) return 'normal'
  if (
    value >= normalMin - borderlineRange &&
    value <= normalMax + borderlineRange
  ) return 'borderline'
  return 'abnormal'
}

const STATUS_COLORS: Record<ClinicalStatus, { text: string; bg: string; border: string; dot: string }> = {
  normal:     { text: 'text-emerald-400', bg: 'bg-emerald-950/60', border: 'border-emerald-800', dot: 'bg-emerald-400' },
  borderline: { text: 'text-amber-400',   bg: 'bg-amber-950/60',   border: 'border-amber-800',   dot: 'bg-amber-400'   },
  abnormal:   { text: 'text-red-400',     bg: 'bg-red-950/60',     border: 'border-red-800',     dot: 'bg-red-400'     },
}

const GROUP_LABELS: Record<MeasurementGroup, string> = {
  skeletal:    'Skeletal',
  dental:      'Dental',
  soft_tissue: 'Soft Tissue',
  vertical:    'Vertical',
}

const GROUP_COLORS: Record<MeasurementGroup, string> = {
  skeletal:    '#22d3ee',   // cyan
  dental:      '#a78bfa',   // violet
  soft_tissue: '#fb923c',   // orange
  vertical:    '#4ade80',   // green
}

// =============================================================================
// SVG line styles per measurement group
// =============================================================================

function getLandmarkStroke(group: MeasurementGroup): string {
  return GROUP_COLORS[group] + 'aa'
}

// =============================================================================
// Props
// =============================================================================

interface CephalometricOverlayProps {
  /** Optional overrides for specific measurement values (e.g. from computed plan) */
  measurementOverrides?: Partial<Record<string, number>>
  className?: string
}

// =============================================================================
// Main component
// =============================================================================

export default function CephalometricOverlay({
  measurementOverrides = {},
  className = '',
}: CephalometricOverlayProps) {
  const [visibleGroups, setVisibleGroups] = useState<Set<MeasurementGroup>>(
    new Set(['skeletal', 'dental', 'vertical', 'soft_tissue'])
  )
  const [selectedMeasurementId, setSelectedMeasurementId] = useState<string | null>(null)
  const [hoveredLandmark, setHoveredLandmark] = useState<string | null>(null)

  // Merge overrides into measurements
  const measurements: CephMeasurement[] = MEASUREMENTS.map(m => ({
    ...m,
    value: measurementOverrides[m.id] ?? m.value,
  }))

  const toggleGroup = useCallback((group: MeasurementGroup) => {
    setVisibleGroups(prev => {
      const next = new Set(prev)
      if (next.has(group)) next.delete(group)
      else next.add(group)
      return next
    })
  }, [])

  const visibleMeasurements = measurements.filter(m => visibleGroups.has(m.group))
  const selectedMeasurement = selectedMeasurementId
    ? measurements.find(m => m.id === selectedMeasurementId)
    : null

  return (
    <div className={`flex h-full min-h-0 bg-slate-900 ${className}`} data-testid="cephalometric-overlay">
      {/* SVG lateral view */}
      <div className="flex-1 relative flex items-center justify-center bg-slate-950 min-w-0">
        <CephalometricSVG
          landmarks={LANDMARKS}
          measurements={visibleMeasurements}
          selectedMeasurementId={selectedMeasurementId}
          hoveredLandmark={hoveredLandmark}
          onSelectMeasurement={setSelectedMeasurementId}
          onHoverLandmark={setHoveredLandmark}
        />

        {/* Hovered landmark tooltip */}
        {hoveredLandmark && LANDMARK_MAP[hoveredLandmark] && (
          <div className="absolute bottom-4 left-4 bg-slate-900/95 border border-slate-700 rounded px-3 py-2 pointer-events-none">
            <p className="text-xs font-semibold text-slate-100">
              {LANDMARK_MAP[hoveredLandmark].abbr} — {LANDMARK_MAP[hoveredLandmark].label}
            </p>
            <p className="text-2xs text-slate-400 mt-0.5">{LANDMARK_MAP[hoveredLandmark].description}</p>
          </div>
        )}
      </div>

      {/* Measurement panel */}
      <div className="w-64 shrink-0 flex flex-col border-l border-slate-800 bg-slate-900 overflow-hidden">
        {/* Group toggles */}
        <div className="p-3 border-b border-slate-800">
          <p className="text-xs font-semibold text-slate-300 mb-2">Measurement Groups</p>
          <div className="flex flex-wrap gap-1">
            {(Object.keys(GROUP_LABELS) as MeasurementGroup[]).map(group => (
              <button
                key={group}
                onClick={() => toggleGroup(group)}
                className={`flex items-center gap-1 px-2 py-1 rounded text-2xs font-medium transition-colors border ${
                  visibleGroups.has(group)
                    ? 'text-slate-200 border-slate-600 bg-slate-700'
                    : 'text-slate-600 border-slate-800 hover:border-slate-700'
                }`}
                data-testid={`group-toggle-${group}`}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: visibleGroups.has(group) ? GROUP_COLORS[group] : '#475569' }}
                />
                {GROUP_LABELS[group]}
                {visibleGroups.has(group) ? <Eye size={10} /> : <EyeOff size={10} />}
              </button>
            ))}
          </div>
        </div>

        {/* Measurements list */}
        <div className="flex-1 overflow-y-auto">
          {(Object.keys(GROUP_LABELS) as MeasurementGroup[])
            .filter(g => visibleGroups.has(g))
            .map(group => {
              const groupMeasurements = visibleMeasurements.filter(m => m.group === group)
              if (groupMeasurements.length === 0) return null

              return (
                <div key={group} className="border-b border-slate-800 last:border-b-0">
                  <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: GROUP_COLORS[group] }} />
                    <span className="text-2xs font-semibold text-slate-400 uppercase tracking-wide">
                      {GROUP_LABELS[group]}
                    </span>
                  </div>
                  {groupMeasurements.map(m => (
                    <MeasurementRow
                      key={m.id}
                      measurement={m}
                      selected={selectedMeasurementId === m.id}
                      onClick={() => setSelectedMeasurementId(prev => prev === m.id ? null : m.id)}
                    />
                  ))}
                </div>
              )
            })}
        </div>

        {/* Selected measurement detail */}
        {selectedMeasurement && (
          <div className="p-3 border-t border-slate-700 bg-slate-800/50">
            <div className="flex items-start gap-2">
              <Info size={13} className="text-slate-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-slate-200">{selectedMeasurement.label}</p>
                <p className="text-2xs text-slate-400 mt-0.5">{selectedMeasurement.description}</p>
                <p className="text-2xs text-slate-500 mt-1">
                  Normal: {selectedMeasurement.normalMin}–{selectedMeasurement.normalMax} {selectedMeasurement.unit}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// =============================================================================
// SVG renderer
// =============================================================================

interface CephalometricSVGProps {
  landmarks: CephPoint[]
  measurements: CephMeasurement[]
  selectedMeasurementId: string | null
  hoveredLandmark: string | null
  onSelectMeasurement: (id: string) => void
  onHoverLandmark: (id: string | null) => void
}

function CephalometricSVG({
  landmarks,
  measurements,
  selectedMeasurementId,
  hoveredLandmark,
  onSelectMeasurement,
  onHoverLandmark,
}: CephalometricSVGProps) {
  const viewBox = '80 60 220 280'

  return (
    <svg
      viewBox={viewBox}
      className="w-full h-full max-w-md"
      style={{ maxHeight: '520px' }}
      data-testid="cephalometric-svg"
    >
      {/* Background skull silhouette */}
      <SkullSilhouette />

      {/* Measurement lines */}
      {measurements.map(m => {
        if (!m.linePoints) return null
        const selected = selectedMeasurementId === m.id
        const status = getClinicalStatus(m)
        const stroke = selected
          ? STATUS_COLORS[status].dot.replace('bg-', '#').replace('-400', '') + 'ff'
          : GROUP_COLORS[m.group] + (selected ? 'ff' : '88')

        return m.linePoints.map((pair, i) => {
          const ptA = LANDMARK_MAP[pair[0]]
          const ptB = LANDMARK_MAP[pair[1]]
          if (!ptA || !ptB) return null
          return (
            <line
              key={`${m.id}-line-${i}`}
              x1={ptA.x} y1={ptA.y}
              x2={ptB.x} y2={ptB.y}
              stroke={GROUP_COLORS[m.group]}
              strokeWidth={selected ? 1.5 : 0.8}
              strokeDasharray={selected ? 'none' : '3 2'}
              strokeOpacity={selected ? 1 : 0.6}
              style={{ cursor: 'pointer' }}
              onClick={() => onSelectMeasurement(m.id)}
            />
          )
        })
      })}

      {/* Landmark points */}
      {landmarks.map(lm => {
        const isHovered = hoveredLandmark === lm.id
        return (
          <g
            key={lm.id}
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => onHoverLandmark(lm.id)}
            onMouseLeave={() => onHoverLandmark(null)}
          >
            <circle
              cx={lm.x} cy={lm.y}
              r={isHovered ? 4.5 : 3}
              fill={isHovered ? '#22d3ee' : '#64748b'}
              stroke={isHovered ? '#0e7490' : '#334155'}
              strokeWidth={1}
              data-testid={`landmark-${lm.id}`}
            />
            {isHovered && (
              <text
                x={lm.x + 5} y={lm.y - 3}
                fontSize="7"
                fill="#22d3ee"
                fontFamily="monospace"
                fontWeight="bold"
              >
                {lm.abbr}
              </text>
            )}
          </g>
        )
      })}

      {/* Measurement value labels */}
      {measurements.map(m => {
        if (!m.linePoints?.[0]) return null
        const ptA = LANDMARK_MAP[m.linePoints[0][0]]
        const ptB = LANDMARK_MAP[m.linePoints[0][1]]
        if (!ptA || !ptB) return null

        const mx = (ptA.x + ptB.x) / 2 + 3
        const my = (ptA.y + ptB.y) / 2 - 2
        const status = getClinicalStatus(m)
        const labelColor = status === 'normal' ? '#4ade80' : status === 'borderline' ? '#fbbf24' : '#f87171'
        const selected = selectedMeasurementId === m.id

        return (
          <text
            key={`${m.id}-label`}
            x={mx} y={my}
            fontSize={selected ? '7.5' : '6.5'}
            fill={labelColor}
            fontFamily="monospace"
            fontWeight={selected ? 'bold' : 'normal'}
            style={{ cursor: 'pointer', pointerEvents: 'none' }}
          >
            {m.value.toFixed(1)}{m.unit}
          </text>
        )
      })}
    </svg>
  )
}

// =============================================================================
// Measurement list row
// =============================================================================

interface MeasurementRowProps {
  measurement: CephMeasurement
  selected: boolean
  onClick: () => void
}

function MeasurementRow({ measurement: m, selected, onClick }: MeasurementRowProps) {
  const status = getClinicalStatus(m)
  const styles = STATUS_COLORS[status]

  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-all border-b border-slate-800/50 last:border-b-0 ${
        selected ? 'bg-slate-700/50' : 'hover:bg-slate-800/40'
      }`}
      data-testid={`ceph-measurement-${m.id}`}
    >
      {/* Status dot */}
      <span className={`w-2 h-2 rounded-full shrink-0 ${styles.dot}`} />

      {/* Name */}
      <span className="text-xs font-semibold text-slate-200 w-24 shrink-0">{m.label}</span>

      {/* Value */}
      <span className={`text-xs font-mono font-bold flex-1 text-right ${styles.text}`}>
        {m.value.toFixed(1)}{m.unit}
      </span>

      {/* Normal range */}
      <span className="text-2xs text-slate-600 font-mono whitespace-nowrap">
        {m.normalMin}–{m.normalMax}
      </span>
    </div>
  )
}

// =============================================================================
// Skull silhouette (simplified SVG path for a lateral skull outline)
// =============================================================================

function SkullSilhouette() {
  return (
    <g opacity={0.18}>
      {/* Cranium */}
      <path
        d="M162 110 Q148 90 155 75 Q172 58 198 62 Q228 58 242 78 Q252 92 248 110 Q244 130 240 140 Q238 158 242 172 Q246 188 242 210 Q238 230 228 252 Q220 270 216 280 Q210 290 200 296 Q186 300 172 292 Q158 282 155 270 Q148 256 148 238 Q148 222 150 208 Q150 192 145 182 Q138 168 136 152 Q134 138 140 126 Z"
        fill="none"
        stroke="#64748b"
        strokeWidth="1.5"
      />
      {/* Mandible */}
      <path
        d="M148 196 Q140 200 136 210 Q130 224 132 240 Q134 258 140 272 Q148 284 160 292 Q170 298 184 296 Q196 296 208 288 Q222 278 232 266 Q242 252 244 238 Q246 224 242 212"
        fill="none"
        stroke="#64748b"
        strokeWidth="1"
        strokeDasharray="4 3"
      />
      {/* Nasal bones */}
      <path
        d="M162 110 Q152 118 144 128 Q138 138 136 150"
        fill="none"
        stroke="#64748b"
        strokeWidth="1"
      />
      {/* Orbital rim */}
      <path
        d="M126 118 Q124 126 126 134 Q130 140 138 142 Q148 142 154 136 Q158 130 156 122 Q152 114 144 112 Q136 112 126 118 Z"
        fill="none"
        stroke="#64748b"
        strokeWidth="0.8"
      />
    </g>
  )
}

import { useState, useRef, useEffect, useCallback } from 'react'
import { RotateCcw, Cpu, Check, X, Undo2, Redo2, Lock, Unlock, Move3d, Save, XCircle } from 'lucide-react'
import { usePlanningStore } from '../../stores/planningStore'
import { planningApi } from '../../lib/api'
import { ConfidenceBadge } from '../common/ConfidenceBar'
import type { FragmentTransform } from '../../types/medical'

interface TransformSliderProps {
  label: string
  axis: 'X' | 'Y' | 'Z'
  value: number
  min: number
  max: number
  step: number
  unit: string
  onChange: (v: number) => void
  disabled?: boolean
}

function TransformSlider({ label, axis, value, min, max, step, unit, onChange, disabled }: TransformSliderProps) {
  const axisColor = { X: 'text-red-400', Y: 'text-emerald-400', Z: 'text-blue-400' }[axis]
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const isOutOfRange = (v: number) => v < min || v > max

  const handleValueClick = () => {
    if (disabled) return
    setEditValue(value.toFixed(2))
    setIsEditing(true)
  }

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.select()
    }
  }, [isEditing])

  const commitEdit = () => {
    const parsed = parseFloat(editValue)
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, parsed))
      onChange(clamped)
    }
    setIsEditing(false)
  }

  const cancelEdit = () => {
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      commitEdit()
    } else if (e.key === 'Escape') {
      cancelEdit()
    }
  }

  return (
    <div className="flex items-center gap-2" data-testid={`transform-slider-${label}-${axis}`}>
      <span className={`text-xs font-mono font-bold w-3 ${axisColor}`}>{axis}</span>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="flex-1"
      />
      {isEditing ? (
        <input
          ref={inputRef}
          type="number"
          value={editValue}
          onChange={e => setEditValue(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={handleKeyDown}
          step={step}
          className={`w-20 text-xs font-mono text-right bg-slate-800 border rounded px-1 py-0.5 outline-none ${
            isOutOfRange(parseFloat(editValue) || 0)
              ? 'border-red-500 text-red-400'
              : 'border-cyan-500 text-slate-100'
          }`}
          data-testid={`transform-input-${label}-${axis}`}
        />
      ) : (
        <span
          className="text-xs font-mono text-slate-300 w-20 text-right cursor-pointer hover:text-cyan-400 transition-colors"
          onClick={handleValueClick}
          title="Click to edit value"
          data-testid={`transform-value-${label}-${axis}`}
        >
          {value.toFixed(2)}<span className="text-slate-500"> {unit}</span>
        </span>
      )}
    </div>
  )
}

export default function FragmentControls() {
  const {
    currentPlan,
    selectedFragmentId,
    transformHistory,
    historyIndex,
    isDirty,
    updateFragmentTransform,
    acceptAiSuggestion,
    resetFragment,
    undo,
    redo,
    markSaved,
  } = usePlanningStore()

  const [activeSection, setActiveSection] = useState<'translate' | 'rotate'>('translate')
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [lastSavedTransforms, setLastSavedTransforms] = useState<Record<string, FragmentTransform> | null>(null)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  if (!currentPlan || !selectedFragmentId) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6 text-center" data-testid="fragment-controls-empty">
        <Move3d size={32} className="text-slate-600 mb-3" />
        <p className="text-sm font-medium text-slate-400">No Fragment Selected</p>
        <p className="text-xs text-slate-600 mt-1">Click a fragment in the 3D viewer to select it</p>
      </div>
    )
  }

  const fragment = currentPlan.fragmentTransforms.find((f: FragmentTransform) => f.fragmentId === selectedFragmentId)
  if (!fragment) return null

  // Store last saved state on first render
  if (!lastSavedTransforms) {
    const saved: Record<string, FragmentTransform> = {}
    currentPlan.fragmentTransforms.forEach((f: FragmentTransform) => {
      saved[f.fragmentId] = { ...f }
    })
    setLastSavedTransforms(saved)
  }

  const t = fragment.currentTransform
  const hasSuggestion = !!fragment.suggestedTransform

  // Debounced auto-save to backend
  const debouncedSave = useCallback((fragmentId: string, transform: unknown) => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }
    debounceTimerRef.current = setTimeout(async () => {
      try {
        await planningApi.updateFragmentTransform(currentPlan.id, fragmentId, transform)
      } catch {
        // Auto-save failures are silent — user can still explicitly save
      }
    }, 500)
  }, [currentPlan.id])

  const updateTranslation = (axis: 'x' | 'y' | 'z', val: number) => {
    if (!fragment || fragment.isLocked) return
    const newTransform = {
      ...fragment.currentTransform,
      translation: {
        ...fragment.currentTransform.translation,
        [axis]: val,
      },
    }
    updateFragmentTransform(selectedFragmentId, newTransform, 'manual')
    debouncedSave(selectedFragmentId, newTransform)
  }

  const updateRotation = (axis: 'x' | 'y' | 'z', val: number) => {
    if (!fragment || fragment.isLocked) return
    const newTransform = {
      ...fragment.currentTransform,
      rotation: {
        ...fragment.currentTransform.rotation,
        [axis]: val,
      },
    }
    updateFragmentTransform(selectedFragmentId, newTransform, 'manual')
    debouncedSave(selectedFragmentId, newTransform)
  }

  const handleSave = async () => {
    setIsSaving(true)
    setSaveStatus('idle')
    try {
      await planningApi.updateFragmentTransform(
        currentPlan.id,
        selectedFragmentId,
        fragment.currentTransform
      )
      markSaved()
      // Update last saved state
      const saved: Record<string, FragmentTransform> = {}
      currentPlan.fragmentTransforms.forEach((f: FragmentTransform) => {
        saved[f.fragmentId] = { ...f }
      })
      setLastSavedTransforms(saved)
      setSaveStatus('success')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus('idle'), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const handleDiscard = () => {
    if (!lastSavedTransforms || !lastSavedTransforms[selectedFragmentId]) return
    const savedFragment = lastSavedTransforms[selectedFragmentId]
    updateFragmentTransform(selectedFragmentId, savedFragment.currentTransform, 'reset')
  }

  return (
    <div className="flex flex-col h-full" data-testid="fragment-controls">
      {/* Fragment info header */}
      <div className="p-3 border-b border-slate-700 bg-slate-900">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-100">{fragment.displayName}</span>
            {fragment.isLocked && <Lock size={12} className="text-amber-400" />}
            {fragment.isAligned && (
              <span className="text-2xs bg-emerald-950 border border-emerald-800 text-emerald-400 px-1.5 py-0.5 rounded">Aligned</span>
            )}
          </div>
          {fragment.isLocked ? (
            <button className="btn-icon" title="Unlock fragment" data-testid="unlock-fragment">
              <Lock size={13} />
            </button>
          ) : (
            <button className="btn-icon" title="Lock fragment" data-testid="lock-fragment">
              <Unlock size={13} />
            </button>
          )}
        </div>
        <div className="grid grid-cols-3 gap-2 mt-2 text-center">
          {[
            { label: 'Volume', value: `${fragment.volumeCm3.toFixed(1)} cm³` },
            { label: 'X center', value: `${fragment.centroid.x.toFixed(1)} mm` },
            { label: 'Y center', value: `${fragment.centroid.y.toFixed(1)} mm` },
          ].map(m => (
            <div key={m.label} className="bg-slate-800 rounded p-1.5">
              <p className="text-2xs text-slate-500">{m.label}</p>
              <p className="text-xs font-mono text-slate-200">{m.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Transform section toggle */}
      <div className="flex border-b border-slate-700">
        {(['translate', 'rotate'] as const).map(s => (
          <button
            key={s}
            onClick={() => setActiveSection(s)}
            className={`flex-1 py-2 text-xs font-semibold capitalize transition-colors ${
              activeSection === s ? 'text-cyan-400 border-b-2 border-cyan-500' : 'text-slate-400 hover:text-slate-200'
            }`}
            data-testid={`transform-section-${s}`}
          >
            {s === 'translate' ? 'Translation (mm)' : 'Rotation (°)'}
          </button>
        ))}
      </div>

      {/* Transform sliders */}
      <div className="p-3 space-y-3 border-b border-slate-700">
        {activeSection === 'translate' ? (
          <>
            <TransformSlider label="Translation" axis="X" value={t.translation.x} min={-20} max={20} step={0.1} unit="mm"
              onChange={v => updateTranslation('x', v)} disabled={fragment.isLocked} />
            <TransformSlider label="Translation" axis="Y" value={t.translation.y} min={-20} max={20} step={0.1} unit="mm"
              onChange={v => updateTranslation('y', v)} disabled={fragment.isLocked} />
            <TransformSlider label="Translation" axis="Z" value={t.translation.z} min={-20} max={20} step={0.1} unit="mm"
              onChange={v => updateTranslation('z', v)} disabled={fragment.isLocked} />
          </>
        ) : (
          <>
            <TransformSlider label="Rotation" axis="X" value={t.rotation.x} min={-180} max={180} step={0.5} unit="°"
              onChange={v => updateRotation('x', v)} disabled={fragment.isLocked} />
            <TransformSlider label="Rotation" axis="Y" value={t.rotation.y} min={-180} max={180} step={0.5} unit="°"
              onChange={v => updateRotation('y', v)} disabled={fragment.isLocked} />
            <TransformSlider label="Rotation" axis="Z" value={t.rotation.z} min={-180} max={180} step={0.5} unit="°"
              onChange={v => updateRotation('z', v)} disabled={fragment.isLocked} />
          </>
        )}
      </div>

      {/* Save / Discard buttons */}
      <div className="p-3 border-b border-slate-700">
        <div className="flex gap-2">
          <button
            onClick={handleSave}
            disabled={!isDirty || isSaving}
            className={`flex items-center gap-1.5 flex-1 justify-center text-xs font-semibold py-1.5 rounded transition-colors ${
              saveStatus === 'success'
                ? 'bg-emerald-900 text-emerald-300 border border-emerald-700'
                : saveStatus === 'error'
                ? 'bg-red-900 text-red-300 border border-red-700'
                : isDirty
                ? 'bg-cyan-900 text-cyan-300 border border-cyan-700 hover:bg-cyan-800'
                : 'bg-slate-800 text-slate-500 border border-slate-700'
            } disabled:opacity-50`}
            data-testid="save-changes"
          >
            <Save size={13} />
            {isSaving ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : saveStatus === 'error' ? 'Save Failed' : 'Save Changes'}
          </button>
          <button
            onClick={handleDiscard}
            disabled={!isDirty}
            className="flex items-center gap-1.5 flex-1 justify-center btn-ghost text-xs disabled:opacity-40"
            data-testid="discard-changes"
          >
            <XCircle size={13} /> Discard
          </button>
        </div>
      </div>

      {/* AI Suggestion */}
      {hasSuggestion && (
        <div className="p-3 border-b border-slate-700 bg-cyan-950/20" data-testid="ai-suggestion">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Cpu size={13} className="text-cyan-400" />
              <span className="text-xs font-semibold text-cyan-400">AI Suggested Position</span>
            </div>
            {fragment.suggestionConfidence != null && (
              <ConfidenceBadge value={fragment.suggestionConfidence} threshold={0.8} />
            )}
          </div>

          {fragment.suggestionConfidence != null && fragment.suggestionConfidence < 0.75 && (
            <p className="text-2xs text-amber-400 mb-2">⚠ Low confidence — manual verification recommended</p>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => acceptAiSuggestion(selectedFragmentId)}
              disabled={fragment.isLocked}
              className="flex items-center gap-1.5 flex-1 justify-center btn-success text-xs"
              data-testid="accept-ai-suggestion"
            >
              <Check size={13} /> Accept
            </button>
            <button
              onClick={() => resetFragment(selectedFragmentId)}
              className="flex items-center gap-1.5 flex-1 justify-center btn-secondary text-xs"
              data-testid="reset-fragment"
            >
              <RotateCcw size={13} /> Reset
            </button>
          </div>
        </div>
      )}

      {/* Undo/Redo */}
      <div className="p-3 border-b border-slate-700">
        <div className="flex gap-2">
          <button
            onClick={undo}
            disabled={historyIndex < 0}
            className="flex items-center gap-1.5 flex-1 justify-center btn-ghost text-xs disabled:opacity-40"
            data-testid="undo-transform"
          >
            <Undo2 size={13} /> Undo
          </button>
          <button
            onClick={redo}
            disabled={historyIndex >= transformHistory.length - 1}
            className="flex items-center gap-1.5 flex-1 justify-center btn-ghost text-xs disabled:opacity-40"
            data-testid="redo-transform"
          >
            Redo <Redo2 size={13} />
          </button>
        </div>
      </div>

      {/* Transform history */}
      <div className="flex-1 overflow-y-auto p-3" data-testid="transform-history">
        <p className="label-xs mb-2">Transform History ({transformHistory.length})</p>
        {transformHistory.length === 0 ? (
          <p className="text-xs text-slate-600 italic">No transforms applied</p>
        ) : (
          <div className="space-y-1.5">
            {[...transformHistory].reverse().map((entry, i) => (
              <div
                key={entry.id}
                className={`flex items-center gap-2 p-2 rounded text-xs ${i === 0 ? 'bg-slate-700 border border-slate-600' : 'bg-slate-800'}`}
                data-testid={`history-entry-${entry.id}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  entry.source === 'ai_suggestion' ? 'bg-cyan-400' : entry.source === 'reset' ? 'bg-amber-400' : 'bg-slate-400'
                }`} />
                <span className="flex-1 text-slate-300 truncate">{entry.description}</span>
                <span className="font-mono text-slate-600 text-2xs shrink-0">
                  {new Date(entry.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

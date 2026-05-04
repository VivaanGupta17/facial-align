/**
 * Keyboard shortcut management for the Facial Align planning interface.
 *
 * Features:
 *  - Declarative shortcut registry with conflict detection
 *  - useEffect-based event listener management
 *  - Shortcut context (only active in planning, global, viewer, etc.)
 *  - Built-in shortcuts: Ctrl+Z, Ctrl+Shift+Z, Space, Escape, R, F, 1/2/3
 */

import { useEffect, useCallback, useRef, type RefObject } from 'react'
import { useViewerStore } from '../stores/viewerStore'
import { usePlanningStore } from '../stores/planningStore'

// =============================================================================
// Types
// =============================================================================

export type ShortcutContext = 'global' | 'planning' | 'viewer' | 'review'

export interface KeyboardShortcut {
  /** Key identifier (e.g. 'z', 'escape', 'space', '1') */
  key: string
  /** Label for display in help UI */
  label: string
  /** Description shown in shortcut cheatsheet */
  description: string
  /** Require Ctrl/Cmd */
  ctrl?: boolean
  /** Require Shift */
  shift?: boolean
  /** Require Alt */
  alt?: boolean
  /** Contexts where this shortcut is active (default: global) */
  contexts?: ShortcutContext[]
  /** Callback to invoke */
  handler: (event: KeyboardEvent) => void
  /** Prevent default browser behavior */
  preventDefault?: boolean
}

// Normalise key string for comparison
function normaliseKey(key: string): string {
  return key.toLowerCase().trim()
}

/** Build a canonical shortcut ID for conflict detection */
function shortcutId(s: Pick<KeyboardShortcut, 'key' | 'ctrl' | 'shift' | 'alt'>): string {
  const parts: string[] = []
  if (s.ctrl) parts.push('ctrl')
  if (s.alt) parts.push('alt')
  if (s.shift) parts.push('shift')
  parts.push(normaliseKey(s.key))
  return parts.join('+')
}

// =============================================================================
// Registry — singleton, shared across hook instances
// =============================================================================

const registry = new Map<string, KeyboardShortcut>()

function registerShortcut(s: KeyboardShortcut): () => void {
  const id = shortcutId(s)
  if (registry.has(id) && import.meta.env.DEV) {
    console.warn(`[useKeyboardShortcuts] Shortcut conflict: "${id}" is already registered. Overwriting.`)
  }
  registry.set(id, s)
  return () => {
    registry.delete(id)
  }
}

// =============================================================================
// Global event handler
// =============================================================================

let globalHandlerAttached = false

function attachGlobalHandler() {
  if (globalHandlerAttached || typeof window === 'undefined') return
  globalHandlerAttached = true

  window.addEventListener('keydown', (e: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs/textareas
    const target = e.target as HTMLElement
    if (
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable
    ) return

    const id = shortcutId({
      key: e.key,
      ctrl: e.ctrlKey || e.metaKey,
      shift: e.shiftKey,
      alt: e.altKey,
    })

    const shortcut = registry.get(id)
    if (shortcut) {
      if (shortcut.preventDefault) e.preventDefault()
      shortcut.handler(e)
    }
  })
}

// =============================================================================
// Main hook — planning interface shortcuts
// =============================================================================

interface UsePlanningShortcutsOptions {
  /** Disable all shortcuts (e.g. when a modal is open) */
  disabled?: boolean
  /** Element to scope shortcuts to (null = global) */
  targetRef?: RefObject<HTMLElement>
  /** Called when 'R' is pressed (reset view) */
  onResetView?: () => void
  /** Called when 'F' is pressed (focus selection) */
  onFocusSelection?: () => void
}

/**
 * Register the full suite of planning workspace keyboard shortcuts.
 * Returns the active shortcut list for rendering a help overlay.
 */
export function usePlanningShortcuts(options: UsePlanningShortcutsOptions = {}): KeyboardShortcut[] {
  const { disabled = false, onResetView, onFocusSelection } = options

  const { undo, redo } = usePlanningStore()
  const { setActiveTool, setViewMode } = useViewerStore()

  // Attach the global handler once
  useEffect(() => { attachGlobalHandler() }, [])

  const shortcutsRef = useRef<Array<() => void>>([])

  const registerAll = useCallback(() => {
    if (disabled) return

    // Deregister previously registered shortcuts
    shortcutsRef.current.forEach(deregister => deregister())
    shortcutsRef.current = []

    const register = (s: KeyboardShortcut) => {
      shortcutsRef.current.push(registerShortcut(s))
    }

    // Undo
    register({
      key: 'z',
      ctrl: true,
      label: 'Undo',
      description: 'Undo the last fragment transform',
      contexts: ['planning'],
      preventDefault: true,
      handler: () => undo(),
    })

    // Redo
    register({
      key: 'z',
      ctrl: true,
      shift: true,
      label: 'Redo',
      description: 'Redo the last undone transform',
      contexts: ['planning'],
      preventDefault: true,
      handler: () => redo(),
    })

    // Toggle measurement mode
    register({
      key: ' ',
      label: 'Toggle Distance Measurement',
      description: 'Activate / deactivate the distance measurement tool',
      contexts: ['viewer', 'planning'],
      preventDefault: true,
      handler: () => {
        const { viewerState } = useViewerStore.getState()
        setActiveTool(viewerState.activeTool === 'measure_distance' ? 'none' : 'measure_distance')
      },
    })

    // Cancel / escape current tool
    register({
      key: 'Escape',
      label: 'Cancel Tool',
      description: 'Cancel the current active tool and return to selection mode',
      contexts: ['viewer', 'planning'],
      preventDefault: false,
      handler: () => setActiveTool('none'),
    })

    // Reset view
    register({
      key: 'r',
      label: 'Reset View',
      description: 'Reset the 3D camera to the default anterior view',
      contexts: ['viewer', 'planning'],
      handler: () => onResetView?.(),
    })

    // Focus selection
    register({
      key: 'f',
      label: 'Focus Selection',
      description: 'Frame the selected fragment in the 3D viewer',
      contexts: ['viewer', 'planning'],
      handler: () => onFocusSelection?.(),
    })

    // Switch view planes: 1 = axial, 2 = coronal, 3 = sagittal, 4 = 3D
    register({
      key: '1',
      label: 'Axial View',
      description: 'Switch to axial (transverse) cross-section view',
      contexts: ['viewer', 'planning'],
      handler: () => setViewMode('axial'),
    })

    register({
      key: '2',
      label: 'Coronal View',
      description: 'Switch to coronal (frontal) cross-section view',
      contexts: ['viewer', 'planning'],
      handler: () => setViewMode('coronal'),
    })

    register({
      key: '3',
      label: 'Sagittal View',
      description: 'Switch to sagittal (lateral) cross-section view',
      contexts: ['viewer', 'planning'],
      handler: () => setViewMode('sagittal'),
    })

    register({
      key: '4',
      label: '3D View',
      description: 'Switch back to 3D perspective view',
      contexts: ['viewer', 'planning'],
      handler: () => setViewMode('3d'),
    })

    // Angle measurement
    register({
      key: 'a',
      label: 'Angle Tool',
      description: 'Activate the angle measurement tool',
      contexts: ['viewer', 'planning'],
      handler: () => {
        const { viewerState } = useViewerStore.getState()
        setActiveTool(viewerState.activeTool === 'measure_angle' ? 'none' : 'measure_angle')
      },
    })

    // Select tool
    register({
      key: 's',
      label: 'Select Tool',
      description: 'Activate fragment selection mode',
      contexts: ['viewer', 'planning'],
      handler: () => setActiveTool('select'),
    })
  }, [disabled, undo, redo, setActiveTool, setViewMode, onResetView, onFocusSelection])

  useEffect(() => {
    registerAll()
    return () => {
      shortcutsRef.current.forEach(deregister => deregister())
      shortcutsRef.current = []
    }
  }, [registerAll])

  return buildShortcutList()
}

// =============================================================================
// useKeyboardShortcut — register a single shortcut
// =============================================================================

interface SingleShortcutOptions {
  key: string
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  handler: (event: KeyboardEvent) => void
  preventDefault?: boolean
  disabled?: boolean
}

/**
 * Register a single keyboard shortcut. Deregisters on unmount or when deps change.
 */
export function useKeyboardShortcut({
  key,
  ctrl,
  shift,
  alt,
  handler,
  preventDefault = false,
  disabled = false,
}: SingleShortcutOptions): void {
  useEffect(() => {
    attachGlobalHandler()
  }, [])

  useEffect(() => {
    if (disabled) return

    const deregister = registerShortcut({
      key,
      ctrl,
      shift,
      alt,
      label: key,
      description: '',
      handler,
      preventDefault,
    })

    return deregister
  }, [key, ctrl, shift, alt, handler, preventDefault, disabled])
}

// =============================================================================
// useEscapeKey — convenience hook for modals / overlays
// =============================================================================

export function useEscapeKey(handler: () => void, disabled = false): void {
  useKeyboardShortcut({
    key: 'Escape',
    handler,
    disabled,
  })
}

// =============================================================================
// Get all registered shortcuts (for help overlay)
// =============================================================================

function buildShortcutList(): KeyboardShortcut[] {
  return Array.from(registry.values())
}

export function useShortcutList(): KeyboardShortcut[] {
  // Re-read on each render — the registry is a singleton
  return buildShortcutList()
}

// =============================================================================
// Format shortcut key for display
// =============================================================================

export function formatShortcutKey(s: Pick<KeyboardShortcut, 'key' | 'ctrl' | 'shift' | 'alt'>): string {
  const isMac = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform)
  const parts: string[] = []

  if (s.ctrl) parts.push(isMac ? '⌘' : 'Ctrl')
  if (s.alt) parts.push(isMac ? '⌥' : 'Alt')
  if (s.shift) parts.push('⇧')

  const key = s.key === ' ' ? 'Space' : s.key.length === 1 ? s.key.toUpperCase() : s.key
  parts.push(key)

  return parts.join(isMac ? '' : '+')
}

// =============================================================================
// Shortcut cheat-sheet data (static for help overlay)
// =============================================================================

export interface ShortcutCheatsheetEntry {
  keys: string
  label: string
  context: string
}

export const SHORTCUT_CHEATSHEET: ShortcutCheatsheetEntry[] = [
  { keys: 'Ctrl+Z',       label: 'Undo fragment transform',     context: 'Planning' },
  { keys: 'Ctrl+⇧+Z',    label: 'Redo fragment transform',     context: 'Planning' },
  { keys: 'Space',        label: 'Toggle distance measurement',  context: 'Viewer'   },
  { keys: 'A',            label: 'Toggle angle measurement',     context: 'Viewer'   },
  { keys: 'S',            label: 'Select tool',                  context: 'Viewer'   },
  { keys: 'Escape',       label: 'Cancel current tool',          context: 'Viewer'   },
  { keys: 'R',            label: 'Reset camera view',            context: 'Viewer'   },
  { keys: 'F',            label: 'Focus selected fragment',      context: 'Viewer'   },
  { keys: '1',            label: 'Axial view',                   context: 'Viewer'   },
  { keys: '2',            label: 'Coronal view',                 context: 'Viewer'   },
  { keys: '3',            label: 'Sagittal view',                context: 'Viewer'   },
  { keys: '4',            label: '3D perspective view',          context: 'Viewer'   },
]

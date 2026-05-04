import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { ReductionPlan, FragmentTransform, TransformHistoryEntry, OcclusalConstraints, Transform3D } from '../types/medical'

interface PlanningState {
  currentPlan: ReductionPlan | null
  selectedFragmentId: string | null
  transformHistory: TransformHistoryEntry[]
  historyIndex: number
  isDirty: boolean
  isGenerating: boolean
  compareWithPlanId: string | null

  setPlan: (plan: ReductionPlan | null) => void
  selectFragment: (id: string | null) => void
  updateFragmentTransform: (fragmentId: string, transform: Transform3D, source: TransformHistoryEntry['source']) => void
  acceptAiSuggestion: (fragmentId: string) => void
  resetFragment: (fragmentId: string) => void
  undo: () => void
  redo: () => void
  setConstraints: (constraints: Partial<OcclusalConstraints>) => void
  setGenerating: (v: boolean) => void
  setCompareWith: (planId: string | null) => void
  markSaved: () => void
  toggleFragmentLock: (fragmentId: string) => void
}

function cloneTransform(transform: Transform3D): Transform3D {
  return {
    translation: { ...transform.translation },
    rotation: { ...transform.rotation },
    scale: { ...transform.scale },
  }
}

export const usePlanningStore = create<PlanningState>()(
  devtools(
    (set, get) => ({
      currentPlan: null,
      selectedFragmentId: null,
      transformHistory: [],
      historyIndex: -1,
      isDirty: false,
      isGenerating: false,
      compareWithPlanId: null,

      setPlan: (plan) => set({ currentPlan: plan, isDirty: false, transformHistory: [], historyIndex: -1 }),

      selectFragment: (id) => set({ selectedFragmentId: id }),

      updateFragmentTransform: (fragmentId, transform, source) => {
        const { currentPlan, transformHistory, historyIndex } = get()
        if (!currentPlan) return
        const fragment = currentPlan.fragmentTransforms.find((item: FragmentTransform) => item.fragmentId === fragmentId)
        if (!fragment) return

        const entry: TransformHistoryEntry = {
          id: `hist-${Date.now()}`,
          fragmentId,
          transform: cloneTransform(transform),
          previousTransform: cloneTransform(fragment.currentTransform),
          timestamp: new Date().toISOString(),
          source,
          description: source === 'ai_suggestion' ? 'AI suggestion applied' : source === 'reset' ? 'Fragment reset' : 'Manual transform',
        }

        // Trim future history if we've undone
        const newHistory = [...transformHistory.slice(0, historyIndex + 1), entry]

        const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
          f.fragmentId === fragmentId ? { ...f, currentTransform: cloneTransform(transform) } : f
        )

        set({
          currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments },
          transformHistory: newHistory,
          historyIndex: newHistory.length - 1,
          isDirty: true,
        })
      },

      acceptAiSuggestion: (fragmentId) => {
        const { currentPlan, updateFragmentTransform } = get()
        if (!currentPlan) return
        const fragment = currentPlan.fragmentTransforms.find((f: FragmentTransform) => f.fragmentId === fragmentId)
        if (fragment?.suggestedTransform) {
          updateFragmentTransform(fragmentId, fragment.suggestedTransform, 'ai_suggestion')
        }
      },

      resetFragment: (fragmentId) => {
        const { currentPlan, updateFragmentTransform } = get()
        if (!currentPlan) return
        const fragment = currentPlan.fragmentTransforms.find((f: FragmentTransform) => f.fragmentId === fragmentId)
        if (fragment) {
          updateFragmentTransform(fragmentId, fragment.baseTransform, 'reset')
        }
      },

      undo: () => {
        const { historyIndex, transformHistory, currentPlan } = get()
        if (historyIndex < 0 || !currentPlan) return
        const entry = transformHistory[historyIndex]
        const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
          f.fragmentId === entry.fragmentId
            ? { ...f, currentTransform: cloneTransform(entry.previousTransform ?? f.baseTransform) }
            : f
        )

        set({
          currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments },
          historyIndex: historyIndex - 1,
          isDirty: historyIndex - 1 >= 0,
        })
      },

      redo: () => {
        const { historyIndex, transformHistory, currentPlan } = get()
        if (historyIndex >= transformHistory.length - 1 || !currentPlan) return
        const nextIndex = historyIndex + 1
        const entry = transformHistory[nextIndex]

        const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
          f.fragmentId === entry.fragmentId ? { ...f, currentTransform: cloneTransform(entry.transform) } : f
        )
        set({ currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments }, historyIndex: nextIndex, isDirty: true })
      },

      setConstraints: (constraints) => {
        const { currentPlan } = get()
        if (!currentPlan) return
        set({ currentPlan: { ...currentPlan, constraints: { ...currentPlan.constraints, ...constraints } }, isDirty: true })
      },

      setGenerating: (v) => set({ isGenerating: v }),

      setCompareWith: (planId) => set({ compareWithPlanId: planId }),

      markSaved: () => set({ isDirty: false }),

      toggleFragmentLock: (fragmentId) => {
        const { currentPlan } = get()
        if (!currentPlan) return

        const updatedFragments = currentPlan.fragmentTransforms.map((fragment: FragmentTransform) =>
          fragment.fragmentId === fragmentId
            ? { ...fragment, isLocked: !fragment.isLocked }
            : fragment
        )

        set({ currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments } })
      },
    }),
    { name: 'PlanningStore' }
  )
)

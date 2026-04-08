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

        const entry: TransformHistoryEntry = {
          id: `hist-${Date.now()}`,
          fragmentId,
          transform,
          timestamp: new Date().toISOString(),
          source,
          description: source === 'ai_suggestion' ? 'AI suggestion applied' : source === 'reset' ? 'Fragment reset' : 'Manual transform',
        }

        // Trim future history if we've undone
        const newHistory = [...transformHistory.slice(0, historyIndex + 1), entry]

        const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
          f.fragmentId === fragmentId ? { ...f, currentTransform: transform } : f
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
        const prevIndex = historyIndex - 1
        const entry = prevIndex >= 0 ? transformHistory[prevIndex] : null

        if (entry) {
          const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
            f.fragmentId === entry.fragmentId ? { ...f, currentTransform: entry.transform } : f
          )
          set({ currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments }, historyIndex: prevIndex })
        } else {
          set({ historyIndex: -1 })
        }
      },

      redo: () => {
        const { historyIndex, transformHistory, currentPlan } = get()
        if (historyIndex >= transformHistory.length - 1 || !currentPlan) return
        const nextIndex = historyIndex + 1
        const entry = transformHistory[nextIndex]

        const updatedFragments = currentPlan.fragmentTransforms.map((f: FragmentTransform) =>
          f.fragmentId === entry.fragmentId ? { ...f, currentTransform: entry.transform } : f
        )
        set({ currentPlan: { ...currentPlan, fragmentTransforms: updatedFragments }, historyIndex: nextIndex })
      },

      setConstraints: (constraints) => {
        const { currentPlan } = get()
        if (!currentPlan) return
        set({ currentPlan: { ...currentPlan, constraints: { ...currentPlan.constraints, ...constraints } }, isDirty: true })
      },

      setGenerating: (v) => set({ isGenerating: v }),

      setCompareWith: (planId) => set({ compareWithPlanId: planId }),

      markSaved: () => set({ isDirty: false }),
    }),
    { name: 'PlanningStore' }
  )
)

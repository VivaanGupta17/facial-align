import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { SurgicalCase, Study } from '../types/medical'

interface CaseState {
  activeCaseId: string | null
  activeCase: SurgicalCase | null
  activeStudy: Study | null
  isCaseLoading: boolean
  caseError: string | null

  setActiveCase: (c: SurgicalCase | null) => void
  setActiveStudy: (s: Study | null) => void
  setActiveCaseId: (id: string | null) => void
  setCaseLoading: (loading: boolean) => void
  setCaseError: (error: string | null) => void
  clearCase: () => void
}

export const useCaseStore = create<CaseState>()(
  devtools(
    (set) => ({
      activeCaseId: null,
      activeCase: null,
      activeStudy: null,
      isCaseLoading: false,
      caseError: null,

      setActiveCase: (c) => set({ activeCase: c, activeCaseId: c?.id ?? null }),
      setActiveStudy: (s) => set({ activeStudy: s }),
      setActiveCaseId: (id) => set({ activeCaseId: id }),
      setCaseLoading: (loading) => set({ isCaseLoading: loading }),
      setCaseError: (error) => set({ caseError: error }),
      clearCase: () => set({ activeCaseId: null, activeCase: null, activeStudy: null, caseError: null }),
    }),
    { name: 'CaseStore' }
  )
)

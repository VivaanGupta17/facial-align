import { makeSurgicalCase } from '../test/factories'
import { useCaseStore } from './caseStore'

describe('caseStore', () => {
  beforeEach(() => {
    useCaseStore.setState(useCaseStore.getInitialState())
  })

  it('sets the active case and clears it again', () => {
    const store = useCaseStore.getState()
    const nextCase = makeSurgicalCase()

    store.setActiveCase(nextCase)
    expect(useCaseStore.getState().activeCaseId).toBe('case-001')
    expect(useCaseStore.getState().activeCase?.caseNumber).toBe('FA-2026-CASE01')

    store.clearCase()
    expect(useCaseStore.getState().activeCase).toBeNull()
    expect(useCaseStore.getState().activeCaseId).toBeNull()
  })
})

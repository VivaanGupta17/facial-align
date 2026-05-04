import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { casesApi, type CasesListParams } from '../lib/api'
import type { SurgicalCase } from '../types/medical'

export const caseKeys = {
  all: ['cases'] as const,
  lists: () => [...caseKeys.all, 'list'] as const,
  list: (params: CasesListParams) => [...caseKeys.lists(), params] as const,
  details: () => [...caseKeys.all, 'detail'] as const,
  detail: (id: string) => [...caseKeys.details(), id] as const,
  recent: (limit: number) => [...caseKeys.all, 'recent', limit] as const,
}

export function useCases(params: CasesListParams = {}) {
  return useQuery({
    queryKey: caseKeys.list(params),
    queryFn: () => casesApi.list(params),
    staleTime: 30_000,
  })
}

export function useCase(caseId: string) {
  return useQuery({
    queryKey: caseKeys.detail(caseId),
    queryFn: () => casesApi.get(caseId),
    enabled: !!caseId,
    staleTime: 20_000,
  })
}

export function useRecentCases(limit = 10) {
  return useQuery({
    queryKey: caseKeys.recent(limit),
    queryFn: () => casesApi.getRecent(limit),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

export function useCreateCase() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: Partial<SurgicalCase>) => casesApi.create(data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: caseKeys.lists() }) },
  })
}

export function useUpdateCase() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<SurgicalCase> }) =>
      casesApi.update(id, data),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: caseKeys.detail(id) })
      qc.invalidateQueries({ queryKey: caseKeys.lists() })
    },
  })
}

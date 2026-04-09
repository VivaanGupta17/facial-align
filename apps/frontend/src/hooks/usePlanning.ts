import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { planningApi, reviewApi } from '../lib/api'
import type { Transform3D } from '../types/medical'

export const planningKeys = {
  all: ['planning'] as const,
  plan: (planId: string) => [...planningKeys.all, 'plan', planId] as const,
  versions: (caseId: string) => [...planningKeys.all, 'versions', caseId] as const,
  review: (caseId: string) => [...planningKeys.all, 'review', caseId] as const,
}

export function usePlan(planId: string | undefined) {
  return useQuery({
    queryKey: planningKeys.plan(planId ?? ''),
    queryFn: () => planningApi.getPlan(planId!),
    enabled: !!planId,
    staleTime: 15_000,
  })
}

export function usePlanVersions(caseId: string) {
  return useQuery({
    queryKey: planningKeys.versions(caseId),
    queryFn: () => planningApi.listVersions(caseId),
    enabled: !!caseId,
    staleTime: 30_000,
  })
}

export function useGeneratePlan(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => planningApi.generatePlan(caseId),
    onSuccess: (data) => {
      // generatePlan now returns { jobId, planId } (202 async), not a plan.
      // Invalidate versions so the UI re-fetches once the plan is ready.
      // The component should poll /jobs/{jobId} or listen via WebSocket
      // for REDUCTION_COMPLETE, then refetch the plan.
      qc.invalidateQueries({ queryKey: planningKeys.versions(caseId) })
    },
  })
}

export function useUpdateFragmentTransform(planId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ fragmentId, transform }: { fragmentId: string; transform: Transform3D }) =>
      planningApi.updateFragmentTransform(planId, fragmentId, transform),
    onSuccess: (data) => {
      qc.setQueryData(planningKeys.plan(planId), data)
    },
  })
}

export function useAcceptAiSuggestion(planId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (fragmentId: string) => planningApi.acceptAiSuggestion(planId, fragmentId),
    onSuccess: (data) => { qc.setQueryData(planningKeys.plan(planId), data) },
  })
}

export function useSurgeonReview(caseId: string) {
  return useQuery({
    queryKey: planningKeys.review(caseId),
    queryFn: () => reviewApi.getReview(caseId),
    enabled: !!caseId,
    staleTime: 10_000,
  })
}

export function useApproveReview(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ reviewId, notes }: { reviewId: string; notes: string }) =>
      reviewApi.approve(reviewId, notes),
    onSuccess: () => { qc.invalidateQueries({ queryKey: planningKeys.review(caseId) }) },
  })
}

export function useRequestRevision(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ reviewId, notes }: { reviewId: string; notes: string }) =>
      reviewApi.requestRevision(reviewId, notes),
    onSuccess: () => { qc.invalidateQueries({ queryKey: planningKeys.review(caseId) }) },
  })
}

export function useRejectReview(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ reviewId, notes }: { reviewId: string; notes: string }) =>
      reviewApi.reject(reviewId, notes),
    onSuccess: () => { qc.invalidateQueries({ queryKey: planningKeys.review(caseId) }) },
  })
}

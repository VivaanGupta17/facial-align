import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { segmentationApi } from '../lib/api'

export const segmentationKeys = {
  all: ['segmentation'] as const,
  result: (caseId: string) => [...segmentationKeys.all, 'result', caseId] as const,
  job: (jobId: string) => [...segmentationKeys.all, 'job', jobId] as const,
}

export function useSegmentationResult(caseId: string) {
  return useQuery({
    queryKey: segmentationKeys.result(caseId),
    queryFn: () => segmentationApi.getResult(caseId),
    enabled: !!caseId,
    staleTime: 60_000,
  })
}

export function useSegmentationJob(jobId: string | undefined) {
  return useQuery({
    queryKey: segmentationKeys.job(jobId ?? ''),
    queryFn: () => segmentationApi.getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: 3_000,
  })
}

export function useSubmitSegmentationJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (caseId: string) => segmentationApi.submitJob(caseId),
    onSuccess: (_, caseId) => {
      qc.invalidateQueries({ queryKey: segmentationKeys.result(caseId) })
    },
  })
}

export function useApproveStructure(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (label: string) => segmentationApi.approveStructure(caseId, label),
    onSuccess: () => { qc.invalidateQueries({ queryKey: segmentationKeys.result(caseId) }) },
  })
}

export function useRejectStructure(caseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (label: string) => segmentationApi.rejectStructure(caseId, label),
    onSuccess: () => { qc.invalidateQueries({ queryKey: segmentationKeys.result(caseId) }) },
  })
}

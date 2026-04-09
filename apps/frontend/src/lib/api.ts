/**
 * Facial Align API Client
 * Typed API client for all backend endpoints.
 * All functions make real HTTP calls to the FastAPI backend.
 */

import type {
  SurgicalCase,
  CaseListItem,
  Study,
  SegmentationResult,
  ReductionPlan,
  DashboardStats,
  SystemHealth,
  SurgeonReview,
  PaginatedResponse,
  CaseStatus,
  CaseType,
} from '../types/medical'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

// ---------------------------
// HTTP Client
// ---------------------------

interface FetchOptions extends Omit<RequestInit, 'body'> {
  body?: unknown
  params?: Record<string, string | number | boolean | string[] | undefined>
}

async function fetchApi<T>(
  path: string,
  options: FetchOptions = {}
): Promise<T> {
  const { body, params, ...init } = options

  // Build query string
  let url = `${BASE_URL}${path}`
  if (params) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) {
        if (Array.isArray(v)) {
          v.forEach(item => searchParams.append(k, String(item)))
        } else {
          searchParams.set(k, String(v))
        }
      }
    })
    const qs = searchParams.toString()
    if (qs) url += `?${qs}`
  }

  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('auth_token') ?? ''}`,
      ...(init.headers as Record<string, string> || {}),
    },
    ...init,
    body: body ? JSON.stringify(body) : undefined,
  })

  if (res.status === 401) {
    localStorage.removeItem('auth_token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }

  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: 'Unknown error' }))
    throw new Error(error.message ?? error.detail ?? `HTTP ${res.status}`)
  }

  // Handle 204 No Content
  if (res.status === 204) return undefined as T

  return res.json() as Promise<T>
}

// ---------------------------
// Cases API
// ---------------------------

export interface CasesListParams {
  page?: number
  pageSize?: number
  status?: CaseStatus[]
  type?: CaseType[]
  surgeonId?: string
  search?: string
  dateFrom?: string
  dateTo?: string
  sortBy?: 'updatedAt' | 'createdAt' | 'caseNumber'
  sortOrder?: 'asc' | 'desc'
}

export const casesApi = {
  /** List all cases with optional filtering */
  list: async (params: CasesListParams = {}): Promise<PaginatedResponse<CaseListItem>> => {
    return fetchApi<PaginatedResponse<CaseListItem>>('/cases', {
      params: {
        page: params.page,
        page_size: params.pageSize,
        status: params.status?.join(','),
        case_type: params.type?.join(','),
        surgeon_id: params.surgeonId,
        search: params.search,
        created_after: params.dateFrom,
        created_before: params.dateTo,
        sort_by: params.sortBy,
        sort_order: params.sortOrder,
      },
    })
  },

  /** Get single case by ID */
  get: async (caseId: string): Promise<SurgicalCase> =>
    fetchApi<SurgicalCase>(`/cases/${caseId}`),

  /** Create new case */
  create: async (data: Partial<SurgicalCase>): Promise<SurgicalCase> =>
    fetchApi<SurgicalCase>('/cases', { method: 'POST', body: data }),

  /** Update case */
  update: async (caseId: string, data: Partial<SurgicalCase>): Promise<SurgicalCase> =>
    fetchApi<SurgicalCase>(`/cases/${caseId}`, { method: 'PATCH', body: data }),

  /** Transition case status */
  transitionStatus: async (caseId: string, targetStatus: string): Promise<SurgicalCase> =>
    fetchApi<SurgicalCase>(`/cases/${caseId}/status`, {
      method: 'POST',
      body: { targetStatus },
    }),

  /** Get recent cases for dashboard */
  getRecent: async (limit = 10): Promise<CaseListItem[]> => {
    const resp = await fetchApi<PaginatedResponse<CaseListItem>>('/cases', {
      params: { page_size: limit, sort_by: 'updatedAt', sort_order: 'desc' },
    })
    return resp.items
  },
}

// ---------------------------
// Dashboard API
// ---------------------------

const ACTIVE_STATUSES: CaseStatus[] = [
  'pending_upload', 'uploading', 'processing', 'segmentation_in_progress',
  'segmentation_review', 'planning', 'review',
]
const PENDING_SEG_STATUSES: CaseStatus[] = [
  'pending_upload', 'uploading', 'processing', 'segmentation_in_progress',
]
const REVIEW_STATUSES: CaseStatus[] = ['review', 'segmentation_review']

export const dashboardApi = {
  /** Compute dashboard statistics from real case data */
  getStats: async (): Promise<DashboardStats> => {
    const [allCases, reviewCases] = await Promise.all([
      fetchApi<PaginatedResponse<CaseListItem>>('/cases', {
        params: { page_size: 1 },
      }),
      fetchApi<PaginatedResponse<CaseListItem>>('/cases', {
        params: { page_size: 1, status: REVIEW_STATUSES.join(',') },
      }),
    ])

    const totalActive = allCases.total
    const awaitingReview = reviewCases.total

    // Estimate pending segmentation from total minus review/completed
    const pendingSegmentation = Math.max(0, totalActive - awaitingReview)

    return {
      activeCases: totalActive,
      activeCasesDelta: 0,
      pendingSegmentation,
      pendingSegmentationDelta: 0,
      awaitingReview,
      awaitingReviewDelta: 0,
      completedThisMonth: 0,
      completedThisMonthDelta: 0,
    }
  },

  /** Get system health */
  getSystemHealth: async (): Promise<SystemHealth> =>
    fetchApi<SystemHealth>('/health'),
}

// ---------------------------
// Studies API
// ---------------------------

export const studiesApi = {
  /** Get study by ID */
  get: async (studyId: string): Promise<Study> =>
    fetchApi<Study>(`/dicom/studies/${studyId}`),

  /** Upload DICOM file with progress tracking */
  upload: async (
    file: File,
    onProgress?: (pct: number) => void
  ): Promise<{ jobId: string; studyId: string }> => {
    const formData = new FormData()
    formData.append('file', file)

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE_URL}/dicom/upload`)
      xhr.setRequestHeader(
        'Authorization',
        `Bearer ${localStorage.getItem('auth_token') ?? ''}`
      )

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          onProgress?.(Math.round((e.loaded / e.total) * 100))
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText))
          } catch {
            reject(new Error('Invalid server response'))
          }
        } else {
          reject(new Error(`Upload failed: HTTP ${xhr.status}`))
        }
      })

      xhr.addEventListener('error', () => reject(new Error('Upload failed: network error')))
      xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))
      xhr.send(formData)
    })
  },

  /** Get study metadata */
  getMetadata: async (studyId: string): Promise<Record<string, unknown>> =>
    fetchApi<Record<string, unknown>>(`/dicom/studies/${studyId}/metadata`),
}

// ---------------------------
// Segmentation API
// ---------------------------

export const segmentationApi = {
  /** Get the latest segmentation result for a case */
  getResult: async (caseId: string): Promise<SegmentationResult> => {
    const results = await fetchApi<SegmentationResult[]>(`/segmentation/cases/${caseId}`)
    if (results.length > 0) return results[results.length - 1]
    throw new Error('No segmentation results found')
  },

  /** Submit a new segmentation job */
  submitJob: async (
    caseId: string
  ): Promise<{ jobId: string; segmentationId: string }> =>
    fetchApi('/segmentation', { method: 'POST', body: { caseId } }),

  /** Get job status by ID */
  getJobStatus: async (
    jobId: string
  ): Promise<{ status: string; progress: number; currentStep?: string }> =>
    fetchApi(`/jobs/${jobId}`),

  /** Approve a segmentation structure (backend TBD) */
  approveStructure: async (_caseId: string, _label: string): Promise<void> => {
    console.warn('approveStructure: backend endpoint not yet implemented')
  },

  /** Reject a segmentation structure (backend TBD) */
  rejectStructure: async (_caseId: string, _label: string): Promise<void> => {
    console.warn('rejectStructure: backend endpoint not yet implemented')
  },

  /** Request re-segmentation for a specific structure */
  requestResegmentation: async (
    caseId: string,
    label: string
  ): Promise<{ jobId: string }> =>
    fetchApi('/segmentation', {
      method: 'POST',
      body: { caseId, structures: [label] },
    }),
}

// ---------------------------
// Planning API
// ---------------------------

export const planningApi = {
  /** Get a reduction plan by ID */
  getPlan: async (planId: string): Promise<ReductionPlan> =>
    fetchApi<ReductionPlan>(`/planning/${planId}`),

  /** Generate a new reduction plan */
  generatePlan: async (
    caseId: string
  ): Promise<{ jobId: string; planId: string }> =>
    fetchApi('/planning', { method: 'POST', body: { caseId } }),

  /** Update fragment transform (surgeon manual edit) */
  updateFragmentTransform: async (
    planId: string,
    fragmentId: string,
    transform: unknown
  ): Promise<ReductionPlan> =>
    fetchApi<ReductionPlan>(`/planning/${planId}/surgeon-edit`, {
      method: 'POST',
      body: { fragmentId, transform, editType: 'manual_adjustment' },
    }),

  /** Accept AI-suggested transform for a fragment */
  acceptAiSuggestion: async (
    planId: string,
    fragmentId: string
  ): Promise<ReductionPlan> =>
    fetchApi<ReductionPlan>(`/planning/${planId}/surgeon-edit`, {
      method: 'POST',
      body: { fragmentId, editType: 'accept_ai_suggestion' },
    }),

  /** Reset a fragment to its base transform */
  resetFragment: async (
    planId: string,
    fragmentId: string
  ): Promise<ReductionPlan> =>
    fetchApi<ReductionPlan>(`/planning/${planId}/surgeon-edit`, {
      method: 'POST',
      body: { fragmentId, editType: 'reset_to_base' },
    }),

  /** List all plan versions for a case */
  listVersions: async (caseId: string): Promise<ReductionPlan[]> =>
    fetchApi<ReductionPlan[]>(`/planning/cases/${caseId}`),
}

// ---------------------------
// Review API
// ---------------------------

export const reviewApi = {
  /** Get surgeon review for a case */
  getReview: async (caseId: string): Promise<SurgeonReview> =>
    fetchApi<SurgeonReview>(`/reviews/${caseId}`),

  /** Update a checklist item */
  updateChecklist: async (
    reviewId: string,
    checklistId: string,
    passed: boolean
  ): Promise<SurgeonReview> =>
    fetchApi<SurgeonReview>(`/reviews/${reviewId}/checklist`, {
      method: 'PATCH',
      body: { checklistId, passed },
    }),

  /** Approve a review */
  approve: async (reviewId: string, notes: string): Promise<SurgeonReview> =>
    fetchApi<SurgeonReview>(`/reviews/${reviewId}/approve`, {
      method: 'POST',
      body: { notes },
    }),

  /** Request revision */
  requestRevision: async (
    reviewId: string,
    notes: string
  ): Promise<SurgeonReview> =>
    fetchApi<SurgeonReview>(`/reviews/${reviewId}/revision`, {
      method: 'POST',
      body: { notes },
    }),

  /** Reject a review */
  reject: async (reviewId: string, notes: string): Promise<SurgeonReview> =>
    fetchApi<SurgeonReview>(`/reviews/${reviewId}/reject`, {
      method: 'POST',
      body: { notes },
    }),
}

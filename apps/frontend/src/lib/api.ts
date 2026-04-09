/**
 * Facial Align API Client
 * Typed API client for all backend endpoints.
 * Currently returns mock data — replace implementations when backend is ready.
 */

import type {
  SurgicalCase,
  Study,
  SegmentationResult,
  ReductionPlan,
  DashboardStats,
  RecentCaseRow,
  SystemHealth,
  SurgeonReview,
  PaginatedResponse,
  CaseStatus,
  CaseType,
} from '../types/medical'

import {
  MOCK_CASES,
  MOCK_DASHBOARD_STATS,
  MOCK_RECENT_CASES,
  MOCK_SEGMENTATION_RESULT,
  MOCK_REDUCTION_PLAN,
  MOCK_SURGEON_REVIEW,
  MOCK_SYSTEM_HEALTH,
  MOCK_STUDIES,
} from './mockData'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

// Simulated network delay for development
const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

// ---------------------------
// HTTP Client (real backend)
// ---------------------------

async function fetchApi<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('auth_token') ?? ''}`,
      ...options.headers,
    },
    ...options,
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: 'Unknown error' }))
    throw new Error(error.message ?? `HTTP ${res.status}`)
  }

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
  sortBy?: 'updatedAt' | 'createdAt' | 'scheduledDate' | 'caseNumber'
  sortOrder?: 'asc' | 'desc'
}

export const casesApi = {
  /** List all cases with optional filtering */
  list: async (params: CasesListParams = {}): Promise<PaginatedResponse<SurgicalCase>> => {
    await delay(400)
    // Mock implementation — filter by status/type if provided
    let items = [...MOCK_CASES]
    if (params.status?.length) items = items.filter(c => params.status!.includes(c.status))
    if (params.type?.length) items = items.filter(c => params.type!.includes(c.type))
    if (params.search) {
      const q = params.search.toLowerCase()
      items = items.filter(c => c.caseNumber.toLowerCase().includes(q))
    }
    const page = params.page ?? 1
    const pageSize = params.pageSize ?? 20
    const start = (page - 1) * pageSize
    return { items: items.slice(start, start + pageSize), total: items.length, page, pageSize, hasMore: start + pageSize < items.length }
  },

  /** Get single case by ID */
  get: async (caseId: string): Promise<SurgicalCase> => {
    await delay(200)
    const c = MOCK_CASES.find(x => x.id === caseId)
    if (!c) throw new Error(`Case ${caseId} not found`)
    return c
  },

  /** Create new case */
  create: async (data: Partial<SurgicalCase>): Promise<SurgicalCase> => {
    await delay(600)
    return { ...MOCK_CASES[0], ...data, id: `case-${Date.now()}`, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() }
  },

  /** Update case */
  update: async (caseId: string, data: Partial<SurgicalCase>): Promise<SurgicalCase> => {
    await delay(300)
    const c = MOCK_CASES.find(x => x.id === caseId)
    if (!c) throw new Error(`Case ${caseId} not found`)
    return { ...c, ...data, updatedAt: new Date().toISOString() }
  },

  /** Get recent cases for dashboard */
  getRecent: async (limit = 10): Promise<RecentCaseRow[]> => {
    await delay(300)
    return MOCK_RECENT_CASES.slice(0, limit)
  },
}

// ---------------------------
// Dashboard API
// ---------------------------

export const dashboardApi = {
  getStats: async (): Promise<DashboardStats> => {
    await delay(250)
    return MOCK_DASHBOARD_STATS
  },

  getSystemHealth: async (): Promise<SystemHealth> => {
    await delay(150)
    return {
      ...MOCK_SYSTEM_HEALTH,
      lastChecked: new Date().toISOString(),
      // Simulate fluctuating GPU utilization
      gpus: MOCK_SYSTEM_HEALTH.gpus.map(g => ({
        ...g,
        utilizationPercent: Math.max(0, Math.min(100, g.utilizationPercent + Math.round((Math.random() - 0.5) * 10))),
      })),
    }
  },
}

// ---------------------------
// Studies API
// ---------------------------

export const studiesApi = {
  get: async (studyId: string): Promise<Study> => {
    await delay(200)
    const s = MOCK_STUDIES.find(x => x.id === studyId)
    if (!s) throw new Error(`Study ${studyId} not found`)
    return s
  },

  upload: async (_file: File, _onProgress?: (pct: number) => void): Promise<{ jobId: string; studyId: string }> => {
    // Simulate upload progress
    for (let i = 0; i <= 100; i += 10) {
      await delay(200)
      _onProgress?.(i)
    }
    return { jobId: `job-${Date.now()}`, studyId: `st-${Date.now()}` }
  },

  getMetadata: async (_studyId: string): Promise<Study> => {
    await delay(400)
    return MOCK_STUDIES[0]
  },
}

// ---------------------------
// Segmentation API
// ---------------------------

export const segmentationApi = {
  getResult: async (caseId: string): Promise<SegmentationResult> => {
    await delay(300)
    return { ...MOCK_SEGMENTATION_RESULT, caseId }
  },

  submitJob: async (caseId: string): Promise<{ jobId: string }> => {
    await delay(500)
    return { jobId: `job-${Date.now()}` }
  },

  getJobStatus: async (jobId: string): Promise<{ status: 'queued' | 'running' | 'completed' | 'failed'; progress: number }> => {
    await delay(100)
    return { status: 'running', progress: Math.random() * 100 }
  },

  approveStructure: async (_caseId: string, _label: string): Promise<void> => {
    await delay(200)
  },

  rejectStructure: async (_caseId: string, _label: string): Promise<void> => {
    await delay(200)
  },

  requestResegmentation: async (_caseId: string, _label: string): Promise<{ jobId: string }> => {
    await delay(300)
    return { jobId: `job-${Date.now()}` }
  },
}

// ---------------------------
// Planning API
// ---------------------------

export const planningApi = {
  getPlan: async (planId: string): Promise<ReductionPlan> => {
    await delay(250)
    return { ...MOCK_REDUCTION_PLAN, id: planId }
  },

  generatePlan: async (caseId: string): Promise<ReductionPlan> => {
    await delay(2000) // Simulate AI planning time
    return { ...MOCK_REDUCTION_PLAN, caseId, version: MOCK_REDUCTION_PLAN.version + 1, createdAt: new Date().toISOString() }
  },

  updateFragmentTransform: async (
    _planId: string,
    _fragmentId: string,
    _transform: unknown
  ): Promise<ReductionPlan> => {
    await delay(150)
    return MOCK_REDUCTION_PLAN
  },

  acceptAiSuggestion: async (_planId: string, _fragmentId: string): Promise<ReductionPlan> => {
    await delay(200)
    return MOCK_REDUCTION_PLAN
  },

  resetFragment: async (_planId: string, _fragmentId: string): Promise<ReductionPlan> => {
    await delay(200)
    return MOCK_REDUCTION_PLAN
  },

  listVersions: async (caseId: string): Promise<ReductionPlan[]> => {
    await delay(300)
    return [1, 2, 3].map(v => ({ ...MOCK_REDUCTION_PLAN, caseId, id: `plan-${v}`, version: v, name: `Plan v${v}` }))
  },
}

// ---------------------------
// Review API
// ---------------------------

export const reviewApi = {
  getReview: async (caseId: string): Promise<SurgeonReview> => {
    await delay(250)
    return { ...MOCK_SURGEON_REVIEW, caseId }
  },

  updateChecklist: async (_reviewId: string, _checklistId: string, _passed: boolean): Promise<SurgeonReview> => {
    await delay(200)
    return MOCK_SURGEON_REVIEW
  },

  approve: async (_reviewId: string, _notes: string): Promise<SurgeonReview> => {
    await delay(500)
    return { ...MOCK_SURGEON_REVIEW, decision: 'approved', signedAt: new Date().toISOString() }
  },

  requestRevision: async (_reviewId: string, _notes: string): Promise<SurgeonReview> => {
    await delay(400)
    return { ...MOCK_SURGEON_REVIEW, decision: 'revision_requested' }
  },

  reject: async (_reviewId: string, _notes: string): Promise<SurgeonReview> => {
    await delay(400)
    return { ...MOCK_SURGEON_REVIEW, decision: 'rejected' }
  },
}

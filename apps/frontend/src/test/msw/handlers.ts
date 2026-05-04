import { http, HttpResponse } from 'msw'

import { makeCapabilities, makeCaseListItem } from '../factories'

const cases = [
  makeCaseListItem(),
  makeCaseListItem({
    id: 'case-002',
    caseNumber: 'FA-2026-CASE02',
    status: 'review',
    caseType: 'midface_fracture',
    patientId: 'patient-002',
  }),
  makeCaseListItem({
    id: 'case-003',
    caseNumber: 'FA-2026-CASE03',
    status: 'approved',
    caseType: 'orthognathic',
    patientId: 'patient-003',
  }),
]

const authResponse = {
  access_token: 'test-access-token',
  refresh_token: 'test-refresh-token',
  token_type: 'bearer',
  expires_in: 3600,
}

export const handlers = [
  http.post('/api/v1/auth/login', async () => HttpResponse.json(authResponse)),
  http.post('/api/v1/auth/register', async () => HttpResponse.json(authResponse, { status: 201 })),
  http.get('/api/v1/auth/me', async () =>
    HttpResponse.json({
      id: 'user-001',
      email: 'surgeon@facialign.local',
      full_name: 'Dr. Test Surgeon',
      role: 'surgeon',
      institution: 'Test Hospital',
      specialty: 'OMFS',
      is_active: true,
      created_at: '2026-05-03T12:00:00.000Z',
    })
  ),
  http.get('/api/v1/cases', ({ request }) => {
    const url = new URL(request.url)
    const status = url.searchParams.get('status')
    const filtered = status
      ? cases.filter((item) => {
          if (status === 'REVIEWED') return item.status === 'review'
          if (status === 'APPROVED') return item.status === 'approved'
          return true
        })
      : cases

    return HttpResponse.json({
      items: filtered.map((item) => ({
        id: item.id,
        caseNumber: item.caseNumber,
        patientId: item.patientId,
        caseType: item.caseType === 'midface_fracture' ? 'TRAUMA' : item.caseType === 'orthognathic' ? 'ORTHOGNATHIC' : 'TRAUMA',
        status:
          item.status === 'review'
            ? 'REVIEWED'
            : item.status === 'approved'
            ? 'APPROVED'
            : 'PLANNING',
        surgeonId: item.surgeonId,
        fractureClassification: item.fractureClassification,
        latestSegmentationStatus: item.latestSegmentationStatus,
        latestPlanConfidence: item.latestPlanConfidence,
        createdAt: item.createdAt,
        updatedAt: item.updatedAt,
      })),
      total: filtered.length,
      page: Number(url.searchParams.get('page') ?? 1),
      pageSize: Number(url.searchParams.get('page_size') ?? 20),
      pages: 1,
      hasMore: false,
    })
  }),
  http.get('/api/v1/health', async () =>
    HttpResponse.json({
      status: 'healthy',
      version: '0.1.0',
      timestamp: '2026-05-03T12:00:00.000Z',
      gpuAvailable: true,
      gpuDevices: ['NVIDIA Test GPU'],
      components: [
        { name: 'postgresql', status: 'healthy', message: null },
        { name: 'celery', status: 'healthy', message: null },
      ],
    })
  ),
  http.get('/api/v1/capabilities', async () =>
    HttpResponse.json({
      generatedAt: '2026-05-03T12:00:00.000Z',
      capabilities: makeCapabilities().map((capability) => ({
        name: capability.name,
        category: capability.category,
        status: capability.status,
        baselineAvailable: capability.baselineAvailable,
        learnedAvailable: capability.learnedAvailable,
        artifactRequired: capability.artifactRequired,
        artifactReady: capability.artifactReady,
        validationTier: capability.validationTier,
        betaStatus: capability.betaStatus,
        modelVersion: capability.modelVersion,
        warnings: capability.warnings,
      })),
    })
  ),
]

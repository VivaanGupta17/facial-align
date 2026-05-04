import { afterEach, describe, expect, it, vi } from 'vitest'

import { authApi, casesApi, dashboardApi, planningApi } from './api'

describe('api client', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it('keeps login failures on the auth screen instead of forcing a session redirect', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Invalid email or password' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    )

    await expect(authApi.login('surgeon@example.com', 'wrong-password')).rejects.toThrow(
      'Invalid email or password'
    )
    expect(fetchSpy).toHaveBeenCalledOnce()
    expect(localStorage.getItem('auth_token')).toBeNull()
  })

  it('clears stored tokens when a protected endpoint returns 401', async () => {
    localStorage.setItem('auth_token', 'expired-token')
    localStorage.setItem('refresh_token', 'expired-refresh-token')

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    )

    await expect(authApi.me()).rejects.toThrow('Session expired')
    expect(localStorage.getItem('auth_token')).toBeNull()
    expect(localStorage.getItem('refresh_token')).toBeNull()
  })

  it('maps case list responses into the frontend workflow model', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [
            {
              id: 'case-001',
              caseNumber: 'FA-2026-0001',
              patientId: 'patient-001',
              caseType: 'TRAUMA',
              status: 'PLANNING',
              surgeonId: 'surgeon-001',
              fractureClassification: 'AO CMF: 91-B3.1 bilateral mandible',
              latestSegmentationStatus: 'complete',
              latestPlanConfidence: 0.88,
              createdAt: '2026-05-03T12:00:00.000Z',
              updatedAt: '2026-05-03T12:00:00.000Z',
            },
          ],
          total: 1,
          page: 1,
          pageSize: 20,
          pages: 1,
          hasMore: false,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    )

    const result = await casesApi.list({ status: ['planning'] })

    expect(result.total).toBe(1)
    expect(result.items[0]).toMatchObject({
      caseNumber: 'FA-2026-0001',
      caseType: 'mandible_fracture',
      status: 'planning',
    })
  })

  it('maps health and capability responses into the dashboard health model', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            status: 'healthy',
            version: '0.1.0',
            timestamp: '2026-05-03T12:00:00.000Z',
            gpuAvailable: true,
            gpuDevices: ['NVIDIA Test GPU'],
            components: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            generatedAt: '2026-05-03T12:00:00.000Z',
            capabilities: [
              {
                name: 'dental_segmentation',
                category: 'segmentation',
                status: 'degraded',
                baselineAvailable: false,
                learnedAvailable: false,
                artifactRequired: true,
                artifactReady: false,
                validationTier: 'beta_unavailable',
                betaStatus: 'beta_unavailable',
                modelVersion: null,
                warnings: ['Artifacts unavailable'],
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      )

    const health = await dashboardApi.getSystemHealth()

    expect(health.gpus[0].name).toBe('NVIDIA Test GPU')
    expect(health.models[0]).toMatchObject({
      name: 'dental_segmentation',
      status: 'degraded',
      validationTier: 'beta_unavailable',
    })
    expect(health.capabilities[0].warnings).toContain('Artifacts unavailable')
  })

  it('maps planning responses into fragment transforms and provenance', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 'plan-001',
          caseId: 'case-001',
          segmentationId: 'seg-001',
          planVersion: 1,
          status: 'approved',
          modelName: 'baseline_icp',
          modelVersion: 'baseline-v1',
          fragments: {
            'fragment-1': {
              fragmentId: 'fragment-1',
              fragmentLabel: 1,
              parentStructure: 'mandible',
              volumeCc: 8.4,
              centroidMm: [12, -18, 4],
            },
          },
          fragmentTransforms: [
            {
              fragmentId: 'fragment-1',
              fragmentLabel: 1,
              confidence: 0.83,
              transform: {
                rotationMatrix: [
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                ],
                translationMm: [2, 0, -1],
              },
            },
          ],
          occlusalConstraints: {
            targetOverjetMm: 2,
            targetOverbiteMm: 2,
            bilateralCondylarSeating: true,
          },
          occlusalMetrics: {
            overjetMm: 1.5,
            overbiteMm: 2,
            midlineDeviationMm: 0.7,
            cantDegrees: 0.5,
            molarRelationship: 'I',
          },
          confidenceScore: 0.83,
          provenance: {
            algorithmUsed: 'baseline_icp',
            validationTier: 'deterministic_baseline',
            betaStatus: 'not_beta',
            warnings: [],
            fallbackReason: null,
            modelVersion: 'baseline-v1',
          },
          surgeonApproved: true,
          approvedAt: '2026-05-03T12:10:00.000Z',
          approvedBy: 'surgeon-001',
          createdAt: '2026-05-03T12:00:00.000Z',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    )

    const plan = await planningApi.getPlan('plan-001')

    expect(plan.provenance?.algorithmUsed).toBe('baseline_icp')
    expect(plan.fragmentTransforms[0]).toMatchObject({
      fragmentId: 'fragment-1',
      structureLabel: 'mandible',
    })
  })
})

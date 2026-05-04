import type {
  CapabilityInfo,
  CaseListItem,
  DashboardStats,
  GpuStatus,
  OcclusalConstraints,
  OcclusalMetrics,
  ProvenanceInfo,
  ReductionPlan,
  ReviewChecklist,
  SurgeonReview,
  SurgicalCase,
  SystemHealth,
  Transform3D,
} from '../types/medical'

const now = '2026-05-03T12:00:00.000Z'

export function makeProvenance(overrides: Partial<ProvenanceInfo> = {}): ProvenanceInfo {
  return {
    algorithmUsed: 'baseline_icp',
    validationTier: 'deterministic_baseline',
    betaStatus: 'not_beta',
    warnings: [],
    fallbackReason: null,
    modelVersion: 'test-v1',
    ...overrides,
  }
}

export function makeCaseListItem(overrides: Partial<CaseListItem> = {}): CaseListItem {
  return {
    id: 'case-001',
    caseNumber: 'FA-2026-CASE01',
    patientId: 'patient-001',
    caseType: 'mandible_fracture',
    status: 'planning',
    surgeonId: 'surgeon-001',
    fractureClassification: 'AO CMF: 91-B3.1',
    latestSegmentationStatus: 'complete',
    latestPlanConfidence: 0.89,
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

export function makeSurgicalCase(overrides: Partial<SurgicalCase> = {}): SurgicalCase {
  return {
    ...makeCaseListItem(),
    studyId: 'study-001',
    reviewerId: null,
    plannedProcedure: 'ORIF',
    diagnosisCodes: ['S02.66XA'],
    targetSurgeryDate: null,
    teamIds: [],
    currentTaskId: null,
    lastError: null,
    approvedAt: null,
    createdBy: 'surgeon-001',
    latestSegmentation: 'seg-001',
    latestPlan: 'plan-001',
    segmentationCount: 1,
    planCount: 1,
    allowedTransitions: ['PLANNING', 'APPROVED'],
    studies: [],
    ...overrides,
  }
}

export function makeTransform(overrides: Partial<Transform3D> = {}): Transform3D {
  return {
    translation: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    ...overrides,
  }
}

export function makePlan(overrides: Partial<ReductionPlan> = {}): ReductionPlan {
  const metrics: OcclusalMetrics = {
    overjetMm: 1.5,
    overjetIdealMin: 1,
    overjetIdealMax: 3,
    overbitePercent: 25,
    overbiteIdealMin: 10,
    overbiteIdealMax: 40,
    midlineDeviationMm: 0.8,
    midlineDeviationThreshold: 2,
    occlusalCantDeg: 0.7,
    molarRelationshipLeft: 'I',
    molarRelationshipRight: 'I',
    canineRelationshipLeft: 'I',
    canineRelationshipRight: 'I',
  }
  const constraints: OcclusalConstraints = {
    enforceOverjet: true,
    enforceOverbite: true,
    enforceMidline: true,
    enforceSymmetry: true,
    enforceCondylarSeating: true,
    maxCondylarDeviationMm: 2,
  }

  return {
    id: 'plan-001',
    caseId: 'case-001',
    version: 1,
    name: 'Baseline Reduction Plan',
    description: 'Deterministic baseline alignment plan.',
    fragmentTransforms: [
      {
        fragmentId: 'fragment-1',
        structureLabel: 'fragment_1',
        displayName: 'Symphysis Fragment',
        baseTransform: makeTransform(),
        currentTransform: makeTransform(),
        suggestedTransform: makeTransform({
          translation: { x: 2, y: 0, z: -1 },
        }),
        suggestionConfidence: 0.82,
        isAligned: false,
        isLocked: false,
        volumeCm3: 8.4,
        centroid: { x: 12, y: -18, z: 4 },
      },
    ],
    occlusalMetrics: metrics,
    constraints,
    validations: [],
    aiConfidence: 0.82,
    aiRecommendation: 'Proceed with baseline alignment and clinical review.',
    isApproved: false,
    provenance: makeProvenance(),
    createdAt: now,
    updatedAt: now,
    createdBy: 'surgeon-001',
    ...overrides,
  }
}

export function makeReviewChecklist(overrides: Partial<ReviewChecklist> = {}): ReviewChecklist {
  return {
    id: 'seg-accuracy',
    category: 'Segmentation',
    label: 'Bone segmentation boundaries are accurate',
    passed: null,
    severity: 'required',
    ...overrides,
  }
}

export function makeSurgeonReview(overrides: Partial<SurgeonReview> = {}): SurgeonReview {
  return {
    id: 'review-001',
    caseId: 'case-001',
    planId: 'plan-001',
    reviewerId: 'surgeon-001',
    reviewerName: 'Dr. Test Surgeon',
    decision: 'pending',
    notes: '',
    checklist: [
      makeReviewChecklist(),
      makeReviewChecklist({
        id: 'reduction-occ',
        category: 'Reduction',
        label: 'Occlusion is within acceptable parameters',
      }),
    ],
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

export function makeCapabilities(): CapabilityInfo[] {
  return [
    {
      name: 'segmentation_baseline',
      category: 'segmentation',
      status: 'available',
      baselineAvailable: true,
      learnedAvailable: false,
      artifactRequired: false,
      artifactReady: true,
      validationTier: 'deterministic_baseline',
      betaStatus: 'not_beta',
      modelVersion: 'baseline-v1',
      warnings: [],
    },
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
      warnings: ['Dental segmentation artifacts are unavailable.'],
    },
  ]
}

export function makeSystemHealth(overrides: Partial<SystemHealth> = {}): SystemHealth {
  const gpus: GpuStatus[] = [
    {
      id: '0',
      name: 'NVIDIA Test GPU',
      utilizationPercent: 12,
      memoryUsedGb: 2,
      memoryTotalGb: 16,
      temperatureCelsius: 41,
      status: 'idle',
    },
  ]

  return {
    gpus,
    models: makeCapabilities().map((capability) => ({
      name: capability.name,
      version: capability.modelVersion ?? 'unknown',
      type: capability.category === 'planning' ? 'planning' : 'segmentation',
      status: capability.status,
      validationTier: capability.validationTier,
      betaStatus: capability.betaStatus,
      baselineAvailable: capability.baselineAvailable,
      learnedAvailable: capability.learnedAvailable,
      artifactReady: capability.artifactReady,
      warnings: capability.warnings,
      lastUpdated: now,
    })),
    capabilities: makeCapabilities(),
    queue: {
      depth: 0,
      estimatedWaitMinutes: 0,
      processingCount: 0,
    },
    apiLatencyMs: 12,
    storageUsedGb: 32,
    storageTotalGb: 128,
    lastChecked: now,
    ...overrides,
  }
}

export function makeDashboardStats(overrides: Partial<DashboardStats> = {}): DashboardStats {
  return {
    activeCases: 4,
    pendingSegmentation: 1,
    awaitingReview: 1,
    completedThisMonth: 2,
    activeCasesDelta: 0,
    pendingSegmentationDelta: 0,
    awaitingReviewDelta: 0,
    completedThisMonthDelta: 0,
    ...overrides,
  }
}

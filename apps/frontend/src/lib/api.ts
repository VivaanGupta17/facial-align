/**
 * Facial Align API Client
 * Normalizes backend `/api/v1` responses into the frontend's current UI model.
 */

import type {
  AngleClass,
  BoundingBox,
  CaseListItem,
  CaseStatus,
  CaseStudyInfo,
  CaseType,
  DashboardStats,
  FragmentTransform as UiFragmentTransform,
  OcclusalConstraints,
  PaginatedResponse,
  ReductionPlan,
  ReviewChecklist,
  SegmentationResult,
  StructureLabel,
  Study,
  SurgeonReview,
  SurgicalCase,
  SystemHealth,
  Transform3D as UiTransform3D,
  Vector3,
} from '../types/medical'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

// ---------------------------
// HTTP Client
// ---------------------------

interface FetchOptions extends Omit<RequestInit, 'body'> {
  body?: unknown
  params?: Record<string, string | number | boolean | string[] | undefined>
}

function buildQueryString(params?: FetchOptions['params']): string {
  if (!params) return ''

  const searchParams = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null) return
    if (Array.isArray(value)) {
      value.forEach((item) => searchParams.append(key, String(item)))
      return
    }
    searchParams.set(key, String(value))
  })

  const qs = searchParams.toString()
  return qs ? `?${qs}` : ''
}

function authHeaders(includeJson = false): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${localStorage.getItem('auth_token') ?? ''}`,
  }
  if (includeJson) {
    headers['Content-Type'] = 'application/json'
  }
  return headers
}

async function fetchApi<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { body, params, headers, ...init } = options
  const response = await fetch(`${BASE_URL}${path}${buildQueryString(params)}`, {
    ...init,
    headers: {
      ...authHeaders(body !== undefined),
      ...(headers as Record<string, string> | undefined),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  if (response.status === 401) {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('refresh_token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(
      error.detail ??
      error.message ??
      error.error ??
      `HTTP ${response.status}`
    )
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

// ---------------------------
// Shared helpers
// ---------------------------

const DEFAULT_VECTOR: Vector3 = { x: 0, y: 0, z: 0 }

const BACKEND_TO_FRONTEND_STATUS: Record<string, CaseStatus> = {
  CREATED: 'processing',
  DICOM_PROCESSING: 'processing',
  SEGMENTED: 'segmentation_review',
  PLANNING: 'planning',
  PLANNED: 'review',
  REVIEWED: 'review',
  APPROVED: 'approved',
  ARCHIVED: 'archived',
  FAILED: 'rejected',
}

const FRONTEND_TO_BACKEND_STATUS: Record<string, string> = {
  pending_upload: 'CREATED',
  uploading: 'DICOM_PROCESSING',
  processing: 'DICOM_PROCESSING',
  segmentation_in_progress: 'SEGMENTED',
  segmentation_review: 'SEGMENTED',
  planning: 'PLANNING',
  review: 'REVIEWED',
  approved: 'APPROVED',
  rejected: 'FAILED',
  completed: 'APPROVED',
  archived: 'ARCHIVED',
}

const STRUCTURE_LABEL_MAP: Record<string, StructureLabel> = {
  mandible: 'mandible',
  maxilla: 'maxilla',
  zygoma_l: 'zygoma_left',
  zygoma_r: 'zygoma_right',
  zygoma_left: 'zygoma_left',
  zygoma_right: 'zygoma_right',
  orbital_floor_l: 'orbit_left',
  orbital_floor_r: 'orbit_right',
  orbit_left: 'orbit_left',
  orbit_right: 'orbit_right',
  frontal_bone: 'frontal_bone',
  nasal_bones: 'nasal_bones',
  teeth_upper: 'teeth_upper',
  teeth_lower: 'teeth_lower',
  temporal_bone_l: 'temporal_left',
  temporal_bone_r: 'temporal_right',
  temporal_left: 'temporal_left',
  temporal_right: 'temporal_right',
  pterygoid_plates: 'sphenoid',
  sphenoid: 'sphenoid',
  skull_base: 'skull_base',
  fragment_1: 'fragment_1',
  fragment_2: 'fragment_2',
  fragment_3: 'fragment_3',
  fragment_4: 'fragment_4',
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function toColorHex(rgb?: number[] | null): string {
  if (!rgb || rgb.length !== 3) return '#64748b'
  return `#${rgb.map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, '0')).join('')}`
}

function toStructureLabel(name: string): StructureLabel {
  const normalized = name.trim().toLowerCase().replace(/[\s-]+/g, '_')
  return STRUCTURE_LABEL_MAP[normalized] ?? (normalized as StructureLabel)
}

function vectorFromList(values?: number[] | null): Vector3 {
  if (!values || values.length < 3) return DEFAULT_VECTOR
  return {
    x: safeNumber(values[0]),
    y: safeNumber(values[1]),
    z: safeNumber(values[2]),
  }
}

function makeBoundingBox(bounds?: BackendBoundingBox | null): BoundingBox {
  const min = {
    x: safeNumber(bounds?.minX),
    y: safeNumber(bounds?.minY),
    z: safeNumber(bounds?.minZ),
  }
  const max = {
    x: safeNumber(bounds?.maxX),
    y: safeNumber(bounds?.maxY),
    z: safeNumber(bounds?.maxZ),
  }

  return {
    min,
    max,
    center: {
      x: (min.x + max.x) / 2,
      y: (min.y + max.y) / 2,
      z: (min.z + max.z) / 2,
    },
  }
}

function frontendStatusFromBackend(status: string): CaseStatus {
  return BACKEND_TO_FRONTEND_STATUS[status] ?? 'processing'
}

function backendStatusFromFrontend(status: string): string {
  return FRONTEND_TO_BACKEND_STATUS[status] ?? status
}

function inferTraumaSubtype(text: string | null | undefined): CaseType {
  const normalized = (text ?? '').toLowerCase()
  if (normalized.includes('panfacial')) return 'panfacial_fracture'
  if (normalized.includes('orbital') || normalized.includes('orbit')) return 'orbital_fracture'
  if (normalized.includes('midface') || normalized.includes('le fort') || normalized.includes('zygoma')) {
    return 'midface_fracture'
  }
  if (normalized.includes('frontal')) return 'frontal_sinus_fracture'
  if (normalized.includes('mandible') || normalized.includes('mandib')) return 'mandible_fracture'
  return 'mandible_fracture'
}

function frontendCaseTypeFromBackend(
  caseType: string,
  fractureClassification?: string | null,
  plannedProcedure?: string | null,
): CaseType {
  switch (caseType) {
    case 'ORTHOGNATHIC':
      return 'orthognathic'
    case 'RECONSTRUCTION':
      return plannedProcedure?.toLowerCase().includes('tumor') ? 'tumor_resection' : 'reconstruction'
    case 'TMJ':
      return 'other'
    case 'OTHER':
      return 'other'
    case 'TRAUMA':
    default:
      return inferTraumaSubtype(fractureClassification ?? plannedProcedure)
  }
}

function backendCaseTypeFromFrontend(caseType: CaseType): string {
  switch (caseType) {
    case 'orthognathic':
      return 'ORTHOGNATHIC'
    case 'tumor_resection':
    case 'reconstruction':
      return 'RECONSTRUCTION'
    case 'other':
      return 'OTHER'
    default:
      return 'TRAUMA'
  }
}

function buildSyntheticPaginatedResponse<T>(
  items: T[],
  page: number,
  pageSize: number,
): PaginatedResponse<T> {
  const hasMore = items.length === pageSize
  const total = hasMore ? (page * pageSize) + 1 : ((page - 1) * pageSize) + items.length
  const pages = hasMore ? page + 1 : Math.max(page, 1)
  return {
    items,
    total,
    page,
    pageSize,
    pages,
    hasMore,
  }
}

function parseAngleClass(value?: string | null): AngleClass {
  const normalized = (value ?? 'Class_I').replace(/[^a-z0-9]/gi, '').toLowerCase()
  switch (normalized) {
    case 'classii':
    case 'classiidiv1':
      return 'II'
    case 'classiidiv2':
      return 'IIb'
    case 'classiii':
      return 'III'
    default:
      return 'I'
  }
}

function magnitude(transform: UiTransform3D): number {
  return Math.sqrt(
    transform.translation.x ** 2 +
    transform.translation.y ** 2 +
    transform.translation.z ** 2
  )
}

function matrixToEulerDegrees(matrix?: number[][] | null): Vector3 {
  if (!matrix || matrix.length !== 3 || matrix.some((row) => row.length !== 3)) {
    return DEFAULT_VECTOR
  }

  const sy = Math.sqrt((matrix[0][0] ** 2) + (matrix[1][0] ** 2))
  const singular = sy < 1e-6

  let x: number
  let y: number
  let z: number

  if (!singular) {
    x = Math.atan2(matrix[2][1], matrix[2][2])
    y = Math.atan2(-matrix[2][0], sy)
    z = Math.atan2(matrix[1][0], matrix[0][0])
  } else {
    x = Math.atan2(-matrix[1][2], matrix[1][1])
    y = Math.atan2(-matrix[2][0], sy)
    z = 0
  }

  const radToDeg = 180 / Math.PI
  return {
    x: x * radToDeg,
    y: y * radToDeg,
    z: z * radToDeg,
  }
}

function eulerDegreesToMatrix(rotation: Vector3): number[][] {
  const degToRad = Math.PI / 180
  const x = rotation.x * degToRad
  const y = rotation.y * degToRad
  const z = rotation.z * degToRad

  const cx = Math.cos(x)
  const sx = Math.sin(x)
  const cy = Math.cos(y)
  const sy = Math.sin(y)
  const cz = Math.cos(z)
  const sz = Math.sin(z)

  return [
    [(cz * cy), (cz * sy * sx) - (sz * cx), (cz * sy * cx) + (sz * sx)],
    [(sz * cy), (sz * sy * sx) + (cz * cx), (sz * sy * cx) - (cz * sx)],
    [(-sy), (cy * sx), (cy * cx)],
  ]
}

function frontendTransformFromBackend(transform?: BackendRigidTransform | null): UiTransform3D {
  return {
    translation: vectorFromList(transform?.translationMm),
    rotation: matrixToEulerDegrees(transform?.rotationMatrix),
    scale: { x: 1, y: 1, z: 1 },
  }
}

function backendTransformFromFrontend(transform: UiTransform3D): BackendRigidTransform {
  return {
    rotationMatrix: eulerDegreesToMatrix(transform.rotation),
    translationMm: [
      safeNumber(transform.translation.x),
      safeNumber(transform.translation.y),
      safeNumber(transform.translation.z),
    ],
  }
}

function mapStudyFromListItem(item: BackendStudyListItem): Study {
  return {
    id: item.id,
    patientId: item.patientId,
    accessionNumber: item.studyUid,
    studyDescription: `Study ${item.studyUid.slice(0, 12)}`,
    studyDate: item.acquisitionDate ?? item.createdAt,
    studyInstanceUid: item.studyUid,
    referringPhysician: '',
    institutionName: '',
    series: [
      {
        seriesInstanceUid: item.studyUid,
        modality: item.modality as 'CT' | 'CBCT' | 'MRI' | 'OPG',
        description: 'Primary Series',
        sliceCount: 0,
        sliceThicknessMm: 1.0,
        acquisitionDate: item.acquisitionDate ?? item.createdAt,
        bodyPart: 'Craniofacial',
        pixelSpacingMm: [1, 1],
        rows: 0,
        cols: 0,
        sopClassUid: '',
      },
    ],
    uploadedAt: item.createdAt,
    uploadedBy: '',
    fileSizeBytes: 0,
    fileCount: item.seriesCount,
    storageUri: '',
  }
}

function mapStudyFromMetadata(studyId: string, metadata: BackendStudyMetadata): Study {
  return {
    id: studyId,
    patientId: '',
    accessionNumber: metadata.studyUid,
    studyDescription: metadata.studyDescription ?? 'Imaging Study',
    studyDate: metadata.acquisitionDate ?? new Date().toISOString(),
    studyInstanceUid: metadata.studyUid,
    referringPhysician: '',
    institutionName: metadata.institutionName ?? '',
    series: metadata.series.map((series) => ({
      seriesInstanceUid: series.seriesInstanceUid,
      modality: series.modality as 'CT' | 'CBCT' | 'MRI' | 'OPG',
      description: series.seriesDescription ?? 'Series',
      sliceCount: series.sliceCount,
      sliceThicknessMm: series.sliceThicknessMm ?? 1.0,
      acquisitionDate: metadata.acquisitionDate ?? new Date().toISOString(),
      bodyPart: metadata.bodyPartExamined ?? 'Craniofacial',
      pixelSpacingMm: (series.pixelSpacingMm?.length === 2
        ? [series.pixelSpacingMm[0], series.pixelSpacingMm[1]]
        : [1, 1]) as [number, number],
      rows: 0,
      cols: 0,
      sopClassUid: '',
    })),
    uploadedAt: metadata.acquisitionDate ?? new Date().toISOString(),
    uploadedBy: '',
    fileSizeBytes: 0,
    fileCount: metadata.totalSliceCount,
    storageUri: '',
  }
}

function mapCaseStudy(info: BackendCaseStudyInfo): CaseStudyInfo {
  return {
    id: info.id,
    studyId: info.studyId,
    studyRole: info.studyRole as CaseStudyInfo['studyRole'],
    studyLabel: info.studyLabel,
    isPrimary: info.isPrimary,
    displayOrder: info.displayOrder,
    createdAt: info.createdAt,
    studyUid: info.studyUid,
    modality: info.modality,
    acquisitionDate: info.acquisitionDate,
    ingestionStatus: info.ingestionStatus,
  }
}

function mapCaseListItem(item: BackendCaseListItem): CaseListItem {
  return {
    id: item.id,
    caseNumber: item.caseNumber,
    patientId: item.patientId,
    caseType: frontendCaseTypeFromBackend(item.caseType, item.fractureClassification, null),
    status: frontendStatusFromBackend(item.status),
    surgeonId: item.surgeonId,
    fractureClassification: item.fractureClassification,
    latestSegmentationStatus: item.latestSegmentationStatus ?? null,
    latestPlanConfidence: item.latestPlanConfidence ?? null,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapCaseResponse(item: BackendCaseResponse): SurgicalCase {
  return {
    id: item.id,
    caseNumber: item.caseNumber,
    patientId: item.patientId,
    studyId: item.studyId,
    caseType: frontendCaseTypeFromBackend(item.caseType, item.fractureClassification, item.plannedProcedure),
    status: frontendStatusFromBackend(item.status),
    surgeonId: item.surgeonId,
    reviewerId: item.reviewerId,
    fractureClassification: item.fractureClassification,
    plannedProcedure: item.plannedProcedure,
    diagnosisCodes: item.diagnosisCodes,
    targetSurgeryDate: item.targetSurgeryDate,
    teamIds: item.teamIds,
    currentTaskId: item.currentTaskId,
    lastError: item.lastError,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
    approvedAt: item.approvedAt,
    createdBy: item.createdBy,
    latestSegmentation: item.latestSegmentation?.id ?? null,
    latestPlan: item.latestPlan?.id ?? null,
    segmentationCount: item.segmentationCount,
    planCount: item.planCount,
    allowedTransitions: (item.allowedTransitions ?? []).map(frontendStatusFromBackend),
    studies: (item.studies ?? []).map(mapCaseStudy),
  }
}

function mapSegmentationResult(result: BackendSegmentationResult): SegmentationResult {
  const confidenceByName = new Map(
    (result.confidenceMaps ?? []).map((entry) => [entry.structureName, entry])
  )
  const meshByName = new Map<string, string>()
  for (const mesh of result.meshes ?? []) {
    if (!meshByName.has(mesh.structureName)) {
      meshByName.set(mesh.structureName, mesh.path)
    }
  }

  const structures = (result.structureLabels ?? []).map((label) => {
    const review = result.structureReviews?.[label.name] ?? {}
    const confidenceMap = confidenceByName.get(label.name)
    const value = safeNumber(confidenceMap?.meanConfidence, result.overallConfidence ?? 0.5)
    const threshold = 0.75
    const boundingBox = makeBoundingBox(confidenceMap?.boundingBox)

    return {
      label: toStructureLabel(label.name),
      displayName: titleCase(label.name),
      confidence: {
        value,
        threshold,
        passesClinicalThreshold: value >= threshold,
      },
      volumeCm3: safeNumber(confidenceMap?.volumeCc),
      surfaceAreaCm2: 0,
      centroid: boundingBox.center,
      boundingBox,
      meshUri: meshByName.get(label.name) ?? '',
      color: toColorHex(label.colorRgb),
      opacity: 1,
      status: (review.status ?? (value >= threshold ? 'accepted' : 'flagged')) as SegmentationResult['structures'][number]['status'],
      fragmentCount: (result.fractureFragments ?? []).filter(
        (fragment) => fragment.parentStructure === label.name
      ).length || undefined,
    }
  })

  const overallConfidence = safeNumber(result.overallConfidence, 0)
  const warnings = result.provenance?.warnings ?? []

  return {
    id: result.id,
    caseId: result.caseId,
    jobId: '',
    modelName: result.modelName,
    modelVersion: result.modelVersion,
    inferenceTimeSeconds: safeNumber(result.inferenceTimeMs) / 1000,
    structures,
    overallConfidence: {
      value: overallConfidence,
      threshold: 0.75,
      passesClinicalThreshold: overallConfidence >= 0.75,
    },
    warnings,
    createdAt: result.createdAt,
    completedAt: result.completedAt ?? result.createdAt,
    gpuUsed: result.gpuDevice ?? 'Unavailable',
  }
}

function mapPlanConstraints(constraints?: BackendOcclusalConstraints | null): OcclusalConstraints {
  return {
    enforceOverjet: true,
    enforceOverbite: true,
    enforceMidline: true,
    enforceSymmetry: true,
    enforceCondylarSeating: constraints?.bilateralCondylarSeating ?? true,
    maxCondylarDeviationMm: 2,
  }
}

function mapPlanValidations(plan: BackendReductionPlanResponse): ReductionPlan['validations'] {
  const metrics = plan.occlusalMetrics
  const warnings = plan.provenance?.warnings ?? []
  const midline = safeNumber(metrics?.midlineDeviationMm)
  const cant = safeNumber(metrics?.cantDegrees)
  const overjet = safeNumber(metrics?.overjetMm)

  return [
    {
      name: 'Midline Alignment',
      description: 'Dental midlines remain within the clinical tolerance.',
      passed: midline <= safeNumber(plan.occlusalConstraints?.midlineToleranceMm, 1),
      value: midline,
      threshold: safeNumber(plan.occlusalConstraints?.midlineToleranceMm, 1),
      severity: 'warning',
    },
    {
      name: 'Occlusal Cant',
      description: 'Occlusal cant remains below the requested threshold.',
      passed: cant <= safeNumber(plan.occlusalConstraints?.cantToleranceDegrees, 2),
      value: cant,
      threshold: safeNumber(plan.occlusalConstraints?.cantToleranceDegrees, 2),
      severity: 'warning',
    },
    {
      name: 'Overjet',
      description: 'Anterior overjet remains close to the requested target.',
      passed: Math.abs(overjet - safeNumber(plan.occlusalConstraints?.targetOverjetMm, 2)) <= 2,
      value: overjet,
      threshold: safeNumber(plan.occlusalConstraints?.targetOverjetMm, 2),
      severity: 'info',
    },
    ...warnings.map((warning, index) => ({
      name: `Pipeline Warning ${index + 1}`,
      description: warning,
      passed: false,
      severity: 'warning' as const,
    })),
  ]
}

function mapPlanFragments(plan: BackendReductionPlanResponse): UiFragmentTransform[] {
  const fragmentInfo = plan.fragments ?? {}
  const transforms = plan.fragmentTransforms ?? []

  const transformed = transforms.length > 0
    ? transforms
    : Object.keys(fragmentInfo).map((fragmentId) => ({
        fragmentId,
        fragmentLabel: safeNumber(fragmentInfo[fragmentId]?.fragmentLabel),
        transform: {
          rotationMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          translationMm: [0, 0, 0],
        },
        confidence: safeNumber(plan.confidenceScore, 0.5),
        isReferenceFragment: false,
      }))

  return transformed.map((fragment, index) => {
    const info = fragmentInfo[fragment.fragmentId]
    const currentTransform = frontendTransformFromBackend(fragment.transform)
    const structureLabel = info?.parentStructure
      ? toStructureLabel(info.parentStructure)
      : (`fragment_${clamp(index + 1, 1, 4)}` as StructureLabel)

    const centroid = vectorFromList(info?.centroidMm)
    const displayBase = info?.parentStructure
      ? `${titleCase(info.parentStructure)} Fragment`
      : titleCase(fragment.fragmentId)

    return {
      fragmentId: fragment.fragmentId,
      structureLabel,
      displayName: `${displayBase} ${index + 1}`,
      baseTransform: currentTransform,
      currentTransform,
      suggestedTransform: currentTransform,
      suggestionConfidence: fragment.confidence ?? plan.confidenceScore ?? 0.5,
      isAligned: fragment.isReferenceFragment || magnitude(currentTransform) < 2,
      isLocked: false,
      volumeCm3: safeNumber(info?.volumeCc),
      centroid,
    }
  })
}

function mapPlanResponse(plan: BackendReductionPlanResponse): ReductionPlan {
  const overjetTarget = safeNumber(plan.occlusalConstraints?.targetOverjetMm, 2)
  const overbiteTargetMm = safeNumber(plan.occlusalConstraints?.targetOverbiteMm, 3)
  const midlineTolerance = safeNumber(plan.occlusalConstraints?.midlineToleranceMm, 1)
  const cantTolerance = safeNumber(plan.occlusalConstraints?.cantToleranceDegrees, 2)
  const metrics = plan.occlusalMetrics

  return {
    id: plan.id,
    caseId: plan.caseId,
    version: plan.planVersion,
    name: `Reduction Plan v${plan.planVersion}`,
    description: plan.provenance?.fallbackReason
      ? `${titleCase(plan.modelName ?? 'planner')} with explicit fallback handling`
      : `Generated with ${plan.modelName ?? 'baseline planner'}`,
    fragmentTransforms: mapPlanFragments(plan),
    occlusalMetrics: {
      overjetMm: safeNumber(metrics?.overjetMm, overjetTarget),
      overjetIdealMin: Math.max(0, overjetTarget - 1),
      overjetIdealMax: overjetTarget + 1,
      overbitePercent: clamp(safeNumber(metrics?.overbiteMm, overbiteTargetMm) * 10, 0, 100),
      overbiteIdealMin: 15,
      overbiteIdealMax: 35,
      midlineDeviationMm: safeNumber(metrics?.midlineDeviationMm),
      midlineDeviationThreshold: midlineTolerance,
      occlusalCantDeg: safeNumber(metrics?.cantDegrees),
      molarRelationshipLeft: parseAngleClass(metrics?.molarRelationship),
      molarRelationshipRight: parseAngleClass(metrics?.molarRelationship),
      canineRelationshipLeft: parseAngleClass(metrics?.molarRelationship),
      canineRelationshipRight: parseAngleClass(metrics?.molarRelationship),
    },
    constraints: mapPlanConstraints(plan.occlusalConstraints),
    validations: mapPlanValidations(plan),
    aiConfidence: safeNumber(plan.confidenceScore, 0.75),
    aiRecommendation: plan.provenance?.warnings?.[0] ?? 'Review the proposed reduction and confirm each fragment position.',
    isApproved: plan.surgeonApproved,
    createdAt: plan.createdAt,
    updatedAt: plan.approvedAt ?? plan.createdAt,
    createdBy: plan.approvedBy ?? 'system',
  }
}

function mapReviewResponse(review: BackendReviewResponse): SurgeonReview {
  return {
    id: review.id,
    caseId: review.caseId,
    planId: review.planId ?? '',
    reviewerId: review.reviewerId,
    reviewerName: review.reviewerName,
    decision: review.decision as SurgeonReview['decision'],
    notes: review.notes,
    checklist: review.checklist as ReviewChecklist[],
    signedAt: review.signedAt ?? undefined,
    createdAt: review.createdAt,
    updatedAt: review.updatedAt,
  }
}

async function getLatestSegmentationId(caseId: string): Promise<string> {
  const caseData = await fetchApi<BackendCaseResponse>(`/cases/${caseId}`)
  if (caseData.latestSegmentation?.id) {
    return caseData.latestSegmentation.id
  }

  const segmentations = await fetchApi<BackendSegmentationResult[]>(`/segmentation/cases/${caseId}`)
  if (!segmentations.length) {
    throw new Error('No completed segmentation is available for this case')
  }

  return segmentations[0].id
}

async function getFragmentTransformForAction(
  planId: string,
  fragmentId: string,
  type: 'suggested' | 'base',
): Promise<UiTransform3D> {
  const plan = await planningApi.getPlan(planId)
  const fragment = plan.fragmentTransforms.find((item) => item.fragmentId === fragmentId)
  if (!fragment) {
    throw new Error(`Fragment ${fragmentId} not found in plan ${planId}`)
  }

  return type === 'suggested'
    ? (fragment.suggestedTransform ?? fragment.currentTransform)
    : fragment.baseTransform
}

// ---------------------------
// Backend DTOs
// ---------------------------

interface BackendBoundingBox {
  minX: number
  minY: number
  minZ: number
  maxX: number
  maxY: number
  maxZ: number
}

interface BackendRigidTransform {
  rotationMatrix: number[][]
  translationMm: number[]
}

interface BackendCaseListItem {
  id: string
  caseNumber: string
  patientId: string
  caseType: string
  status: string
  surgeonId: string | null
  fractureClassification: string | null
  latestSegmentationStatus: string | null
  latestPlanConfidence: number | null
  createdAt: string
  updatedAt: string
}

interface BackendCaseSummary {
  id: string
}

interface BackendCaseStudyInfo {
  id: string
  studyId: string
  studyRole: string
  studyLabel: string | null
  isPrimary: boolean
  displayOrder: number
  createdAt: string
  studyUid: string | null
  modality: string | null
  acquisitionDate: string | null
  ingestionStatus: string | null
}

interface BackendCaseResponse extends BackendCaseListItem {
  studyId: string
  reviewerId: string | null
  plannedProcedure: string | null
  diagnosisCodes: string[] | null
  targetSurgeryDate: string | null
  teamIds: string[] | null
  currentTaskId: string | null
  lastError: string | null
  approvedAt: string | null
  createdBy: string | null
  latestSegmentation: BackendCaseSummary | null
  latestPlan: BackendCaseSummary | null
  segmentationCount: number
  planCount: number
  studies: BackendCaseStudyInfo[]
  allowedTransitions: string[]
}

interface BackendPaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  pages: number
  hasMore: boolean
}

interface BackendStudyListItem {
  id: string
  studyUid: string
  patientId: string
  modality: string
  acquisitionDate: string | null
  seriesCount: number
  ingestionStatus: string
  qualityScore: number | null
  createdAt: string
}

interface BackendSeriesInfo {
  seriesInstanceUid: string
  seriesDescription: string | null
  modality: string
  sliceCount: number
  sliceThicknessMm: number | null
  pixelSpacingMm?: number[] | null
}

interface BackendStudyMetadata {
  studyUid: string
  modality: string
  acquisitionDate: string | null
  bodyPartExamined: string | null
  studyDescription: string | null
  institutionName: string | null
  totalSliceCount: number
  series: BackendSeriesInfo[]
  [key: string]: unknown
}

interface BackendUploadResponse {
  studyId: string
  patientId: string
  ingestionJobId: string
}

interface BackendProvenanceInfo {
  algorithmUsed: string
  validationTier: string
  betaStatus: string
  warnings?: string[]
  fallbackReason?: string | null
  modelVersion?: string | null
}

interface BackendStructureLabel {
  name: string
  labelValue: number
  colorRgb?: number[] | null
}

interface BackendConfidenceMap {
  structureName: string
  meanConfidence: number
  volumeCc?: number | null
  boundingBox?: BackendBoundingBox | null
}

interface BackendMeshInfo {
  structureName: string
  format: string
  path: string
}

interface BackendSegmentationResult {
  id: string
  caseId: string
  status: string
  modelName: string
  modelVersion: string
  structureLabels?: BackendStructureLabel[] | null
  confidenceMaps?: BackendConfidenceMap[] | null
  structureReviews?: Record<string, { status?: string }> | null
  overallConfidence?: number | null
  provenance?: BackendProvenanceInfo | null
  meshes?: BackendMeshInfo[] | null
  fractureFragments?: Array<{ parentStructure?: string | null }> | null
  inferenceTimeMs?: number | null
  gpuDevice?: string | null
  createdAt: string
  completedAt?: string | null
}

interface BackendFragmentInfo {
  fragmentId: string
  fragmentLabel: number
  meshPath?: string | null
  volumeCc?: number | null
  centroidMm?: number[] | null
  parentStructure?: string | null
}

interface BackendFragmentTransform {
  fragmentId: string
  fragmentLabel: number
  transform: BackendRigidTransform
  confidence: number
  isReferenceFragment?: boolean
}

interface BackendOcclusalConstraints {
  targetOverjetMm?: number
  targetOverbiteMm?: number
  molarClassTarget?: string
  midlineToleranceMm?: number
  cantToleranceDegrees?: number
  bilateralCondylarSeating?: boolean
}

interface BackendOcclusalMetrics {
  overjetMm?: number | null
  overbiteMm?: number | null
  molarRelationship?: string | null
  midlineDeviationMm?: number | null
  cantDegrees?: number | null
}

interface BackendReductionPlanResponse {
  id: string
  caseId: string
  segmentationId?: string | null
  planVersion: number
  status: string
  modelName?: string | null
  modelVersion?: string | null
  fragments?: Record<string, BackendFragmentInfo> | null
  fragmentTransforms?: BackendFragmentTransform[] | null
  occlusalConstraints?: BackendOcclusalConstraints | null
  occlusalMetrics?: BackendOcclusalMetrics | null
  confidenceScore?: number | null
  provenance?: BackendProvenanceInfo | null
  surgeonApproved: boolean
  approvedAt?: string | null
  approvedBy?: string | null
  createdAt: string
}

interface BackendReviewResponse {
  id: string
  caseId: string
  planId?: string | null
  reviewerId: string
  reviewerName: string
  decision: string
  notes: string
  checklist: ReviewChecklist[]
  signedAt?: string | null
  createdAt: string
  updatedAt: string
}

interface BackendCapabilityEntry {
  name: string
  category: string
  modelVersion?: string | null
}

interface BackendCapabilitiesResponse {
  generatedAt: string
  capabilities: BackendCapabilityEntry[]
}

interface BackendHealthComponent {
  name: string
  status: string
  message?: string | null
}

interface BackendHealthResponse {
  status: string
  version: string
  timestamp: string
  gpuAvailable: boolean
  gpuDevices: string[]
  components: BackendHealthComponent[]
}

interface BackendJobStatusResponse {
  jobId: string
  status: string
  progress?: {
    percent?: number
    currentStep?: string | null
  } | null
}

// ---------------------------
// Auth types
// ---------------------------

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface UserProfile {
  id: string
  email: string
  full_name: string
  role: string
  institution: string | null
  specialty: string | null
  is_active: boolean
  created_at: string
}

export interface RegisterData {
  email: string
  password: string
  full_name: string
  role?: string
  institution?: string
  specialty?: string
}

// ---------------------------
// Auth API
// ---------------------------

export const authApi = {
  login: (email: string, password: string) =>
    fetchApi<TokenResponse>('/auth/login', { method: 'POST', body: { email, password } }),

  register: (data: RegisterData) =>
    fetchApi<TokenResponse>('/auth/register', { method: 'POST', body: data }),

  me: () => fetchApi<UserProfile>('/auth/me'),

  updateMe: (data: { full_name?: string; institution?: string; specialty?: string }) =>
    fetchApi<UserProfile>('/auth/me', { method: 'PUT', body: data }),

  refresh: (refreshToken: string) =>
    fetchApi<TokenResponse>('/auth/refresh', { method: 'POST', body: { refresh_token: refreshToken } }),

  changePassword: (currentPassword: string, newPassword: string) =>
    fetchApi<void>('/auth/change-password', {
      method: 'POST',
      body: { current_password: currentPassword, new_password: newPassword },
    }),
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
  list: async (params: CasesListParams = {}): Promise<PaginatedResponse<CaseListItem>> => {
    const response = await fetchApi<BackendPaginatedResponse<BackendCaseListItem>>('/cases', {
      params: {
        page: params.page,
        page_size: params.pageSize,
        status: params.status?.[0] ? backendStatusFromFrontend(params.status[0]) : undefined,
        case_type: params.type?.[0] ? backendCaseTypeFromFrontend(params.type[0]) : undefined,
        surgeon_id: params.surgeonId,
        created_after: params.dateFrom,
        created_before: params.dateTo,
      },
    })

    return {
      items: response.items.map(mapCaseListItem),
      total: response.total,
      page: response.page,
      pageSize: response.pageSize,
      pages: response.pages,
      hasMore: response.hasMore,
    }
  },

  get: async (caseId: string): Promise<SurgicalCase> =>
    mapCaseResponse(await fetchApi<BackendCaseResponse>(`/cases/${caseId}`)),

  create: async (data: Partial<SurgicalCase>): Promise<SurgicalCase> => {
    if (!data.patientId || !data.studyId || !data.caseType) {
      throw new Error('patientId, studyId, and caseType are required to create a case')
    }

    const created = await fetchApi<BackendCaseResponse>('/cases', {
      method: 'POST',
      body: {
        patientId: data.patientId,
        studyId: data.studyId,
        caseType: backendCaseTypeFromFrontend(data.caseType),
        fractureClassification: data.fractureClassification ?? undefined,
        plannedProcedure: data.plannedProcedure ?? undefined,
        diagnosisCodes: data.diagnosisCodes ?? undefined,
        targetSurgeryDate: data.targetSurgeryDate ?? undefined,
        surgeonId: data.surgeonId ?? undefined,
        teamIds: data.teamIds ?? undefined,
        uploadNotes: data.lastError ?? undefined,
      },
    })

    return mapCaseResponse(created)
  },

  update: async (caseId: string, data: Partial<SurgicalCase>): Promise<SurgicalCase> => {
    const updated = await fetchApi<BackendCaseResponse>(`/cases/${caseId}`, {
      method: 'PATCH',
      body: {
        surgeonId: data.surgeonId ?? undefined,
        reviewerId: data.reviewerId ?? undefined,
        fractureClassification: data.fractureClassification ?? undefined,
        plannedProcedure: data.plannedProcedure ?? undefined,
        diagnosisCodes: data.diagnosisCodes ?? undefined,
        targetSurgeryDate: data.targetSurgeryDate ?? undefined,
        teamIds: data.teamIds ?? undefined,
      },
    })

    return mapCaseResponse(updated)
  },

  transitionStatus: async (caseId: string, newStatus: string, notes?: string): Promise<SurgicalCase> => {
    const updated = await fetchApi<BackendCaseResponse>(`/cases/${caseId}/status`, {
      method: 'POST',
      body: { newStatus: backendStatusFromFrontend(newStatus), notes },
    })
    return mapCaseResponse(updated)
  },

  getRecent: async (limit = 10): Promise<CaseListItem[]> => {
    const response = await casesApi.list({ page: 1, pageSize: limit })
    return response.items
  },
}

// ---------------------------
// Dashboard API
// ---------------------------

export const dashboardApi = {
  getStats: async (): Promise<DashboardStats> => {
    const [allCases, reviewCases, approvedCases] = await Promise.all([
      fetchApi<BackendPaginatedResponse<BackendCaseListItem>>('/cases', { params: { page: 1, page_size: 1 } }),
      fetchApi<BackendPaginatedResponse<BackendCaseListItem>>('/cases', {
        params: { page: 1, page_size: 1, status: 'REVIEWED' },
      }),
      fetchApi<BackendPaginatedResponse<BackendCaseListItem>>('/cases', {
        params: { page: 1, page_size: 1, status: 'APPROVED' },
      }),
    ])

    return {
      activeCases: allCases.total,
      activeCasesDelta: 0,
      pendingSegmentation: Math.max(0, allCases.total - reviewCases.total - approvedCases.total),
      pendingSegmentationDelta: 0,
      awaitingReview: reviewCases.total,
      awaitingReviewDelta: 0,
      completedThisMonth: approvedCases.total,
      completedThisMonthDelta: 0,
    }
  },

  getSystemHealth: async (): Promise<SystemHealth> => {
    const [health, capabilities] = await Promise.all([
      fetchApi<BackendHealthResponse>('/health'),
      fetchApi<BackendCapabilitiesResponse>('/capabilities'),
    ])

    return {
      gpus: health.gpuDevices.map((name, index) => ({
        id: String(index),
        name,
        utilizationPercent: 0,
        memoryUsedGb: 0,
        memoryTotalGb: 0,
        temperatureCelsius: 0,
        status: health.gpuAvailable ? 'idle' : 'offline',
      })),
      models: capabilities.capabilities.map((capability) => ({
        name: capability.name,
        version: capability.modelVersion ?? 'unknown',
        type: capability.category === 'segmentation'
          ? 'segmentation'
          : capability.category === 'planning'
          ? 'planning'
          : 'occlusion',
        lastUpdated: health.timestamp,
        accuracy: 0,
      })),
      queue: {
        depth: 0,
        estimatedWaitMinutes: 0,
        processingCount: 0,
      },
      apiLatencyMs: 0,
      storageUsedGb: 0,
      storageTotalGb: 0,
      lastChecked: health.timestamp,
    }
  },
}

// ---------------------------
// Studies API
// ---------------------------

export const studiesApi = {
  list: async (
    params: { page?: number; pageSize?: number; modality?: string } = {}
  ): Promise<PaginatedResponse<Study>> => {
    const page = params.page ?? 1
    const pageSize = params.pageSize ?? 20
    const response = await fetchApi<BackendStudyListItem[]>('/dicom/studies', {
      params: {
        page,
        page_size: pageSize,
        modality: params.modality,
      },
    })

    return buildSyntheticPaginatedResponse(response.map(mapStudyFromListItem), page, pageSize)
  },

  get: async (studyId: string): Promise<Study> =>
    mapStudyFromMetadata(studyId, await fetchApi<BackendStudyMetadata>(`/dicom/studies/${studyId}`)),

  upload: async (
    file: File,
    patientMrn: string,
    onProgress?: (pct: number) => void,
  ): Promise<{ jobId: string; studyId: string; patientId: string }> => {
    if (!patientMrn.trim()) {
      throw new Error('Patient MRN is required before upload')
    }

    const formData = new FormData()
    formData.append('files', file)
    formData.append('patient_mrn', patientMrn.trim())

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE_URL}/dicom/upload`)
      xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_token') ?? ''}`)

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          onProgress?.(Math.round((event.loaded / event.total) * 100))
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText) as BackendUploadResponse
            resolve({
              jobId: response.ingestionJobId,
              studyId: response.studyId,
              patientId: response.patientId,
            })
          } catch {
            reject(new Error('Invalid server response'))
          }
          return
        }
        reject(new Error(`Upload failed: HTTP ${xhr.status}`))
      })

      xhr.addEventListener('error', () => reject(new Error('Upload failed: network error')))
      xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))
      xhr.send(formData)
    })
  },

  getMetadata: async (studyId: string): Promise<Record<string, unknown>> =>
    fetchApi<Record<string, unknown>>(`/dicom/studies/${studyId}`),

  uploadChunked: async (
    file: File,
    patientMrn: string,
    opts?: {
      onProgress?: (received: number, total: number, speedBps: number, etaSec: number) => void
      signal?: AbortSignal
      patientAge?: number
      patientSex?: string
      institutionCode?: string
      caseType?: string
    },
  ): Promise<{ jobId: string; studyId: string; patientId: string }> => {
    const chunkSize = 10 * 1024 * 1024

    const initForm = new FormData()
    initForm.append('filename', file.name)
    initForm.append('total_size', String(file.size))
    initForm.append('patient_mrn', patientMrn)
    if (opts?.patientAge != null) initForm.append('patient_age', String(opts.patientAge))
    if (opts?.patientSex) initForm.append('patient_sex', opts.patientSex)
    if (opts?.institutionCode) initForm.append('institution_code', opts.institutionCode)
    if (opts?.caseType) initForm.append('case_type', backendCaseTypeFromFrontend(opts.caseType as CaseType))

    const initRes = await fetch(`${BASE_URL}/dicom/upload/init`, {
      method: 'POST',
      headers: authHeaders(false),
      body: initForm,
      signal: opts?.signal,
    })
    if (!initRes.ok) throw new Error(`Init failed: HTTP ${initRes.status}`)

    const initPayload = await initRes.json() as {
      uploadId: string
      chunkCount: number
    }

    const uploadStartedAt = Date.now()
    for (let chunkIndex = 0; chunkIndex < initPayload.chunkCount; chunkIndex += 1) {
      if (opts?.signal?.aborted) throw new Error('Upload aborted')

      const start = chunkIndex * chunkSize
      const end = Math.min(start + chunkSize, file.size)
      const blob = file.slice(start, end)

      let attempt = 0
      while (attempt < 3) {
        try {
          const chunkForm = new FormData()
          chunkForm.append('chunk', blob, `chunk_${chunkIndex}`)
          const chunkRes = await fetch(`${BASE_URL}/dicom/upload/${initPayload.uploadId}/chunk/${chunkIndex}`, {
            method: 'PUT',
            headers: authHeaders(false),
            body: chunkForm,
            signal: opts?.signal,
          })
          if (!chunkRes.ok) throw new Error(`Chunk ${chunkIndex} failed: HTTP ${chunkRes.status}`)
          break
        } catch (error) {
          attempt += 1
          if (attempt >= 3) throw error
          await new Promise((resolve) => window.setTimeout(resolve, 1000 * attempt))
        }
      }

      const elapsedSeconds = (Date.now() - uploadStartedAt) / 1000
      const bytesUploaded = end
      const speedBps = elapsedSeconds > 0 ? bytesUploaded / elapsedSeconds : 0
      const remaining = file.size - bytesUploaded
      const etaSec = speedBps > 0 ? remaining / speedBps : 0
      opts?.onProgress?.(chunkIndex + 1, initPayload.chunkCount, speedBps, etaSec)
    }

    const completeRes = await fetch(`${BASE_URL}/dicom/upload/${initPayload.uploadId}/complete`, {
      method: 'POST',
      headers: authHeaders(false),
      signal: opts?.signal,
    })
    if (!completeRes.ok) throw new Error(`Complete failed: HTTP ${completeRes.status}`)

    const result = await completeRes.json() as BackendUploadResponse
    return {
      jobId: result.ingestionJobId,
      studyId: result.studyId,
      patientId: result.patientId,
    }
  },

  getUploadStatus: async (uploadId: string) =>
    fetchApi<{
      uploadId: string
      status: string
      receivedChunks: number[]
      chunkCount: number
      chunkSize: number
      totalSize: number
      filename: string
    }>(`/dicom/upload/${uploadId}/status`),
}

// ---------------------------
// Case Studies API
// ---------------------------

export const caseStudiesApi = {
  list: async (caseId: string): Promise<CaseStudyInfo[]> =>
    (await fetchApi<BackendCaseStudyInfo[]>(`/cases/${caseId}/studies`)).map(mapCaseStudy),

  attach: async (
    caseId: string,
    data: { studyId: string; studyRole?: string; studyLabel?: string; isPrimary?: boolean }
  ): Promise<CaseStudyInfo> =>
    mapCaseStudy(
      await fetchApi<BackendCaseStudyInfo>(`/cases/${caseId}/studies`, {
        method: 'POST',
        body: data,
      })
    ),

  detach: async (caseId: string, studyId: string): Promise<void> =>
    fetchApi<void>(`/cases/${caseId}/studies/${studyId}`, { method: 'DELETE' }),

  update: async (
    caseId: string,
    studyId: string,
    data: { studyRole?: string; studyLabel?: string; isPrimary?: boolean }
  ): Promise<CaseStudyInfo> =>
    mapCaseStudy(
      await fetchApi<BackendCaseStudyInfo>(`/cases/${caseId}/studies/${studyId}`, {
        method: 'PATCH',
        body: data,
      })
    ),
}

// ---------------------------
// Segmentation API
// ---------------------------

export const segmentationApi = {
  getResult: async (caseId: string): Promise<SegmentationResult> => {
    const results = await fetchApi<BackendSegmentationResult[]>(`/segmentation/cases/${caseId}`)
    if (!results.length) {
      throw new Error('No segmentation results found')
    }
    return mapSegmentationResult(results[0])
  },

  submitJob: async (caseId: string): Promise<{ jobId: string; segmentationId: string }> => {
    const response = await fetchApi<{ jobId: string; segmentationId: string }>('/segmentation', {
      method: 'POST',
      body: {
        caseId,
        modelName: 'totalsegmentator',
        identifyFragments: true,
        runDentalSegmentation: false,
        fastMode: false,
      },
    })
    return response
  },

  getJobStatus: async (
    jobId: string
  ): Promise<{ status: string; progress: number; currentStep?: string }> => {
    const response = await fetchApi<BackendJobStatusResponse>(`/jobs/${jobId}`)
    return {
      status: response.status,
      progress: safeNumber(response.progress?.percent),
      currentStep: response.progress?.currentStep ?? undefined,
    }
  },

  approveStructure: async (caseId: string, label: string): Promise<void> => {
    const segmentationId = await getLatestSegmentationId(caseId)
    await fetchApi(`/segmentation/${segmentationId}/structures/${label}/approve`, { method: 'POST' })
  },

  rejectStructure: async (caseId: string, label: string): Promise<void> => {
    const segmentationId = await getLatestSegmentationId(caseId)
    await fetchApi(`/segmentation/${segmentationId}/structures/${label}/reject`, { method: 'POST' })
  },

  requestResegmentation: async (caseId: string, label: string): Promise<{ jobId: string }> => {
    const segmentationId = await getLatestSegmentationId(caseId)
    const response = await fetchApi<{ jobId: string }>(
      `/segmentation/${segmentationId}/structures/${label}/resegment`,
      { method: 'POST' }
    )
    return response
  },
}

// ---------------------------
// Planning API
// ---------------------------

export const planningApi = {
  getPlan: async (planId: string): Promise<ReductionPlan> =>
    mapPlanResponse(await fetchApi<BackendReductionPlanResponse>(`/planning/${planId}`)),

  generatePlan: async (
    caseId: string
  ): Promise<{ jobId: string; planId?: string }> => {
    const segmentationId = await getLatestSegmentationId(caseId)
    const response = await fetchApi<{ jobId: string; result?: { planId?: string } }>(
      '/planning',
      {
        method: 'POST',
        body: {
          caseId,
          segmentationId,
          modelName: 'baseline_icp',
          useIntactReference: true,
          includeAlternativePlans: false,
        },
      }
    )

    return {
      jobId: response.jobId,
      planId: response.result?.planId,
    }
  },

  updateFragmentTransform: async (
    planId: string,
    fragmentId: string,
    transform: UiTransform3D
  ): Promise<ReductionPlan> =>
    mapPlanResponse(
      await fetchApi<BackendReductionPlanResponse>(`/planning/${planId}/surgeon-edit`, {
        method: 'POST',
        body: {
          fragmentId,
          newTransform: backendTransformFromFrontend(transform),
          notes: 'Manual adjustment from planning workspace',
          reOptimize: false,
        },
      })
    ),

  acceptAiSuggestion: async (
    planId: string,
    fragmentId: string
  ): Promise<ReductionPlan> => {
    const suggestedTransform = await getFragmentTransformForAction(planId, fragmentId, 'suggested')
    return planningApi.updateFragmentTransform(planId, fragmentId, suggestedTransform)
  },

  resetFragment: async (
    planId: string,
    fragmentId: string
  ): Promise<ReductionPlan> => {
    const baseTransform = await getFragmentTransformForAction(planId, fragmentId, 'base')
    return planningApi.updateFragmentTransform(planId, fragmentId, baseTransform)
  },

  listVersions: async (caseId: string): Promise<ReductionPlan[]> =>
    (await fetchApi<BackendReductionPlanResponse[]>(`/planning/cases/${caseId}`)).map(mapPlanResponse),

  overrideMetric: async (
    planId: string,
    metricName: string,
    targetValue: number,
    notes?: string
  ): Promise<ReductionPlan> =>
    mapPlanResponse(
      await fetchApi<BackendReductionPlanResponse>(`/planning/${planId}/metric-override`, {
        method: 'POST',
        body: { metricName, targetValue, notes },
      })
    ),

  exportSplint: async (planId: string, exportType: string): Promise<Blob> => {
    const response = await fetch(`${BASE_URL}/planning/${planId}/export`, {
      method: 'POST',
      headers: {
        ...authHeaders(true),
      },
      body: JSON.stringify({ exportType }),
    })
    if (!response.ok) {
      throw new Error(`Export failed: HTTP ${response.status}`)
    }
    return response.blob()
  },
}

// ---------------------------
// Export API
// ---------------------------

export interface ExportFileInfo {
  filename: string
  exportType: string
  downloadUrl: string
  vertexCount: number
  faceCount: number
  volumeMm3: number
  isWatertight: boolean
  isPrintable: boolean
}

export interface ExportResponse {
  planId: string
  caseId: string
  files: ExportFileInfo[]
  totalExportTimeSeconds: number
}

export const exportApi = {
  exportPlan: async (
    planId: string,
    exportType = 'full_assembly',
    stlFormat = 'binary',
    structureName?: string,
  ): Promise<ExportResponse> =>
    fetchApi<ExportResponse>(`/planning/${planId}/export`, {
      method: 'POST',
      body: { exportType, stlFormat, structureName },
    }),

  downloadStl: async (planId: string, filename: string): Promise<void> => {
    const response = await fetch(`${BASE_URL}/planning/${planId}/export/${filename}`, {
      headers: authHeaders(false),
    })
    if (!response.ok) {
      throw new Error(`Download failed: HTTP ${response.status}`)
    }

    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = objectUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(objectUrl)
  },
}

// ---------------------------
// Review API
// ---------------------------

export const reviewApi = {
  getReview: async (caseId: string): Promise<SurgeonReview> =>
    mapReviewResponse(await fetchApi<BackendReviewResponse>(`/reviews/${caseId}`)),

  updateChecklist: async (
    reviewId: string,
    checklistId: string,
    passed: boolean
  ): Promise<SurgeonReview> =>
    mapReviewResponse(
      await fetchApi<BackendReviewResponse>(`/reviews/${reviewId}/checklist`, {
        method: 'PATCH',
        body: { checklistId, passed },
      })
    ),

  approve: async (
    reviewId: string,
    notes: string,
    signature?: string
  ): Promise<SurgeonReview> =>
    mapReviewResponse(
      await fetchApi<BackendReviewResponse>(`/reviews/${reviewId}/approve`, {
        method: 'POST',
        body: { notes, signature },
      })
    ),

  requestRevision: async (
    reviewId: string,
    notes: string
  ): Promise<SurgeonReview> =>
    mapReviewResponse(
      await fetchApi<BackendReviewResponse>(`/reviews/${reviewId}/revision`, {
        method: 'POST',
        body: { notes },
      })
    ),

  reject: async (reviewId: string, notes: string): Promise<SurgeonReview> =>
    mapReviewResponse(
      await fetchApi<BackendReviewResponse>(`/reviews/${reviewId}/reject`, {
        method: 'POST',
        body: { notes },
      })
    ),
}

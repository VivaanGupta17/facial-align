// =============================================================================
// Core Medical Entity Types — Facial Align Platform
// =============================================================================

// ---------------------------
// Enumerations
// ---------------------------

export type CaseStatus =
  | 'pending_upload'
  | 'uploading'
  | 'processing'
  | 'segmentation_in_progress'
  | 'segmentation_review'
  | 'planning'
  | 'review'
  | 'approved'
  | 'rejected'
  | 'completed'
  | 'archived'

export type CaseType =
  | 'mandible_fracture'
  | 'midface_fracture'
  | 'panfacial_fracture'
  | 'orbital_fracture'
  | 'frontal_sinus_fracture'
  | 'orthognathic'
  | 'tumor_resection'
  | 'reconstruction'
  | 'other'

export type StructureLabel =
  | 'mandible'
  | 'maxilla'
  | 'zygoma_left'
  | 'zygoma_right'
  | 'orbit_left'
  | 'orbit_right'
  | 'frontal_bone'
  | 'nasal_bones'
  | 'teeth_upper'
  | 'teeth_lower'
  | 'sphenoid'
  | 'temporal_left'
  | 'temporal_right'
  | 'skull_base'
  | 'fragment_1'
  | 'fragment_2'
  | 'fragment_3'
  | 'fragment_4'

export type SegmentationStatus = 'accepted' | 'rejected' | 'pending' | 'flagged'

export type AngleClass = 'I' | 'II' | 'III' | 'IIa' | 'IIb'

export type ReviewDecision = 'approved' | 'revision_requested' | 'rejected' | 'pending'

// ---------------------------
// Primitives
// ---------------------------

export interface Vector3 {
  x: number
  y: number
  z: number
}

export interface Transform3D {
  translation: Vector3
  rotation: Vector3 // Euler angles in degrees
  scale: Vector3
}

export interface BoundingBox {
  min: Vector3
  max: Vector3
  center: Vector3
}

// ---------------------------
// Patient & Study
// ---------------------------

export interface Patient {
  id: string
  anonymizedId: string
  age: number
  sex: 'M' | 'F' | 'O'
  weightKg?: number
  heightCm?: number
  createdAt: string
}

export interface DicomSeries {
  seriesInstanceUid: string
  modality: 'CT' | 'CBCT' | 'MRI' | 'OPG'
  description: string
  sliceCount: number
  sliceThicknessMm: number
  kvp?: number
  mAs?: number
  acquisitionDate: string
  bodyPart: string
  pixelSpacingMm: [number, number]
  rows: number
  cols: number
  sopClassUid: string
}

export interface Study {
  id: string
  patientId: string
  accessionNumber: string
  studyDescription: string
  studyDate: string
  studyInstanceUid: string
  referringPhysician: string
  institutionName: string
  series: DicomSeries[]
  uploadedAt: string
  uploadedBy: string
  fileSizeBytes: number
  fileCount: number
  storageUri: string
}

// ---------------------------
// Study Role & Case-Study junction
// ---------------------------

export type StudyRole = 'pre_op' | 'post_op' | 'follow_up' | 'intra_op'

export interface CaseStudyInfo {
  id: string
  studyId: string
  studyRole: StudyRole
  studyLabel: string | null
  isPrimary: boolean
  displayOrder: number
  createdAt: string
  studyUid: string | null
  modality: string | null
  acquisitionDate: string | null
  ingestionStatus: string | null
}

export interface ChunkedUploadProgress {
  uploadId: string
  chunkSize: number
  chunkCount: number
  receivedCount: number
  totalSize: number
  speedBytesPerSec: number
  etaSeconds: number
}

// ---------------------------
// Surgical Case — matches backend CaseResponse
// ---------------------------

export interface SurgicalCase {
  id: string
  caseNumber: string
  patientId: string
  studyId: string
  caseType: CaseType
  status: CaseStatus
  surgeonId: string | null
  reviewerId: string | null
  fractureClassification: string | null
  plannedProcedure: string | null
  diagnosisCodes: string[] | null
  targetSurgeryDate: string | null
  teamIds: string[] | null
  currentTaskId: string | null
  lastError: string | null
  createdAt: string
  updatedAt: string
  approvedAt: string | null
  createdBy: string | null
  latestSegmentation: string | null
  latestPlan: string | null
  segmentationCount: number
  planCount: number
  allowedTransitions: string[]
  studies: CaseStudyInfo[]
}

/** Lightweight item returned by the list endpoint */
export interface CaseListItem {
  id: string
  caseNumber: string
  patientId: string
  caseType: CaseType
  status: CaseStatus
  surgeonId: string | null
  fractureClassification: string | null
  latestSegmentationStatus: string | null
  latestPlanConfidence: number | null
  createdAt: string
  updatedAt: string
}

// ---------------------------
// Segmentation
// ---------------------------

export interface ConfidenceScore {
  value: number // 0–1
  threshold: number
  passesClinicalThreshold: boolean
}

export interface SegmentedStructure {
  label: StructureLabel
  displayName: string
  confidence: ConfidenceScore
  volumeCm3: number
  surfaceAreaCm2: number
  centroid: Vector3
  boundingBox: BoundingBox
  meshUri: string
  color: string // hex
  opacity: number // 0–1
  status: SegmentationStatus
  fragmentCount?: number
}

export interface SegmentationResult {
  id: string
  caseId: string
  jobId: string
  modelName: string
  modelVersion: string
  inferenceTimeSeconds: number
  structures: SegmentedStructure[]
  overallConfidence: ConfidenceScore
  warnings: string[]
  createdAt: string
  completedAt: string
  gpuUsed: string
}

// ---------------------------
// Fragment & Transform
// ---------------------------

export interface FragmentTransform {
  fragmentId: string
  structureLabel: StructureLabel
  displayName: string
  baseTransform: Transform3D
  currentTransform: Transform3D
  suggestedTransform?: Transform3D
  suggestionConfidence?: number
  isAligned: boolean
  isLocked: boolean
  volumeCm3: number
  centroid: Vector3
}

export interface TransformHistoryEntry {
  id: string
  fragmentId: string
  transform: Transform3D
  timestamp: string
  source: 'manual' | 'ai_suggestion' | 'reset'
  description: string
}

// ---------------------------
// Planning
// ---------------------------

export interface OcclusalMetrics {
  overjetMm: number
  overjetIdealMin: number
  overjetIdealMax: number
  overbitePercent: number
  overbiteIdealMin: number
  overbiteIdealMax: number
  midlineDeviationMm: number
  midlineDeviationThreshold: number
  occlusalCantDeg: number
  molarRelationshipLeft: AngleClass
  molarRelationshipRight: AngleClass
  canineRelationshipLeft: AngleClass
  canineRelationshipRight: AngleClass
}

export interface OcclusalConstraints {
  enforceOverjet: boolean
  enforceOverbite: boolean
  enforceMidline: boolean
  enforceSymmetry: boolean
  enforceCondylarSeating: boolean
  maxCondylarDeviationMm: number
}

export interface ConstraintValidation {
  name: string
  description: string
  passed: boolean
  value?: number
  threshold?: number
  severity: 'error' | 'warning' | 'info'
}

export interface ReductionPlan {
  id: string
  caseId: string
  version: number
  name: string
  description: string
  fragmentTransforms: FragmentTransform[]
  occlusalMetrics: OcclusalMetrics
  constraints: OcclusalConstraints
  validations: ConstraintValidation[]
  aiConfidence: number
  aiRecommendation: string
  isApproved: boolean
  createdAt: string
  updatedAt: string
  createdBy: string
}

// ---------------------------
// Review
// ---------------------------

export interface ReviewChecklist {
  id: string
  category: string
  label: string
  passed: boolean | null
  notes?: string
  severity: 'required' | 'recommended' | 'optional'
}

export interface SurgeonReview {
  id: string
  caseId: string
  planId: string
  reviewerId: string
  reviewerName: string
  decision: ReviewDecision
  notes: string
  checklist: ReviewChecklist[]
  signedAt?: string
  createdAt: string
  updatedAt: string
}

// ---------------------------
// Viewer State
// ---------------------------

export interface StructureVisibility {
  label: StructureLabel
  visible: boolean
  opacity: number
  color: string
  wireframe: boolean
  selected: boolean
}

export interface CameraPreset {
  name: string
  label: string
  position: Vector3
  target: Vector3
  up: Vector3
}

export interface MeshAsset {
  uri: string
  format: 'glb' | 'stl' | 'obj'
  label: StructureLabel
  loaded: boolean
  error?: string
}

export interface MeasurementAnnotation {
  id: string
  type: 'distance' | 'angle' | 'point'
  points: Vector3[]
  value: number
  unit: 'mm' | 'deg'
  label: string
  color: string
  visible: boolean
}

export interface ViewerState {
  viewMode: '3d' | 'axial' | 'coronal' | 'sagittal'
  structureVisibility: Record<StructureLabel, StructureVisibility>
  selectedFragmentId: string | null
  cameraPreset: string
  showGrid: boolean
  showAxes: boolean
  showMeasurements: boolean
  activeTool: 'none' | 'measure_distance' | 'measure_angle' | 'annotate' | 'select'
  measurements: MeasurementAnnotation[]
  showStructuresPanel: boolean
}

// ---------------------------
// System Health
// ---------------------------

export interface GpuStatus {
  id: string
  name: string
  utilizationPercent: number
  memoryUsedGb: number
  memoryTotalGb: number
  temperatureCelsius: number
  status: 'idle' | 'busy' | 'error' | 'offline'
}

export interface ModelInfo {
  name: string
  version: string
  type: 'segmentation' | 'planning' | 'occlusion'
  lastUpdated: string
  accuracy: number
}

export interface QueueStatus {
  depth: number
  estimatedWaitMinutes: number
  processingCount: number
}

export interface SystemHealth {
  gpus: GpuStatus[]
  models: ModelInfo[]
  queue: QueueStatus
  apiLatencyMs: number
  storageUsedGb: number
  storageTotalGb: number
  lastChecked: string
}

// ---------------------------
// API Response Shapes
// ---------------------------

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  pages: number
  hasMore: boolean
}

export interface ApiError {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface ApiResponse<T> {
  data: T
  error?: ApiError
  requestId: string
  timestamp: string
}

// ---------------------------
// Dashboard Summary
// ---------------------------

export interface DashboardStats {
  activeCases: number
  pendingSegmentation: number
  awaitingReview: number
  completedThisMonth: number
  activeCasesDelta: number
  pendingSegmentationDelta: number
  awaitingReviewDelta: number
  completedThisMonthDelta: number
}

export interface RecentCaseRow {
  id: string
  caseNumber: string
  patientId: string
  caseType: CaseType
  status: CaseStatus
  surgeonId: string | null
  updatedAt: string
  targetSurgeryDate?: string | null
}

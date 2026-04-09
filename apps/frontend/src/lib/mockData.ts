/**
 * Mock data for development — replace with real API responses when backend is connected.
 * All patient identifiers are synthetic/anonymized.
 */

import type {
  SurgicalCase,
  Patient,
  Study,
  SegmentationResult,
  ReductionPlan,
  DashboardStats,
  RecentCaseRow,
  SystemHealth,
  SurgeonReview,
  FragmentTransform,
} from '../types/medical'

// ---------------------------
// Patients (anonymized)
// ---------------------------

export const MOCK_PATIENTS: Patient[] = [
  { id: 'pt-001', anonymizedId: 'FA-2024-0047', age: 34, sex: 'M', weightKg: 82, heightCm: 178, createdAt: '2024-11-15T08:22:00Z' },
  { id: 'pt-002', anonymizedId: 'FA-2024-0048', age: 52, sex: 'F', weightKg: 64, heightCm: 165, createdAt: '2024-11-18T11:05:00Z' },
  { id: 'pt-003', anonymizedId: 'FA-2024-0049', age: 28, sex: 'M', weightKg: 77, heightCm: 182, createdAt: '2024-11-20T09:44:00Z' },
  { id: 'pt-004', anonymizedId: 'FA-2024-0051', age: 41, sex: 'F', weightKg: 59, heightCm: 162, createdAt: '2024-11-22T14:18:00Z' },
  { id: 'pt-005', anonymizedId: 'FA-2024-0053', age: 19, sex: 'M', weightKg: 70, heightCm: 175, createdAt: '2024-11-25T07:55:00Z' },
  { id: 'pt-006', anonymizedId: 'FA-2024-0055', age: 63, sex: 'F', weightKg: 68, heightCm: 160, createdAt: '2024-11-28T16:30:00Z' },
]

// ---------------------------
// Studies
// ---------------------------

export const MOCK_STUDIES: Study[] = [
  {
    id: 'st-001', patientId: 'pt-001', accessionNumber: 'ACC-2024-9841',
    studyDescription: 'CT Maxillofacial w/o contrast', studyDate: '2024-11-15',
    studyInstanceUid: '1.2.840.113619.2.55.3.1271', referringPhysician: 'Dr. Chen',
    institutionName: 'Metro General Hospital', uploadedAt: '2024-11-15T10:30:00Z',
    uploadedBy: 'admin', fileSizeBytes: 892043264, fileCount: 412,
    storageUri: 's3://facial-align-dicom/st-001/', 
    series: [{
      seriesInstanceUid: '1.2.840.113619.2.55.3.1271.1', modality: 'CT',
      description: 'Axial 0.625mm', sliceCount: 412, sliceThicknessMm: 0.625,
      kvp: 120, mAs: 280, acquisitionDate: '2024-11-15', bodyPart: 'HEAD',
      pixelSpacingMm: [0.488, 0.488], rows: 512, cols: 512,
      sopClassUid: '1.2.840.10008.5.1.4.1.1.2',
    }]
  },
  {
    id: 'st-002', patientId: 'pt-002', accessionNumber: 'ACC-2024-9852',
    studyDescription: 'CT Orbits/Facial Bones', studyDate: '2024-11-18',
    studyInstanceUid: '1.2.840.113619.2.55.3.1283', referringPhysician: 'Dr. Park',
    institutionName: 'University Medical Center', uploadedAt: '2024-11-18T13:45:00Z',
    uploadedBy: 'admin', fileSizeBytes: 1134567890, fileCount: 521,
    storageUri: 's3://facial-align-dicom/st-002/',
    series: [{
      seriesInstanceUid: '1.2.840.113619.2.55.3.1283.1', modality: 'CT',
      description: 'Axial 0.5mm', sliceCount: 521, sliceThicknessMm: 0.5,
      kvp: 120, mAs: 300, acquisitionDate: '2024-11-18', bodyPart: 'HEAD',
      pixelSpacingMm: [0.488, 0.488], rows: 512, cols: 512,
      sopClassUid: '1.2.840.10008.5.1.4.1.1.2',
    }]
  },
]

// ---------------------------
// Surgical Cases
// ---------------------------

export const MOCK_CASES: SurgicalCase[] = [
  {
    id: 'case-001', caseNumber: 'FA-2024-0047', studyId: 'st-001', patientId: 'pt-001',
    type: 'mandible_fracture', status: 'planning', priority: 'urgent',
    assignments: [
      { surgeonId: 'surg-001', surgeonName: 'Dr. Emily Chen', role: 'primary', assignedAt: '2024-11-15T11:00:00Z' },
      { surgeonId: 'surg-002', surgeonName: 'Dr. Marcus Reid', role: 'reviewer', assignedAt: '2024-11-15T11:00:00Z' },
    ],
    notes: [
      { id: 'note-001', authorId: 'surg-001', authorName: 'Dr. Emily Chen', content: 'Parasymphysis fracture + condylar head fracture right side. Patient presents with limited mouth opening and malocclusion. Priority reduction needed.', createdAt: '2024-11-15T11:30:00Z', tags: ['fracture', 'condyle', 'malocclusion'] },
    ],
    timeline: [
      { id: 'tl-001', event: 'case_created', description: 'Case created from DICOM upload', performedBy: 'Dr. Emily Chen', performedAt: '2024-11-15T11:00:00Z' },
      { id: 'tl-002', event: 'segmentation_started', description: 'Segmentation job submitted', performedBy: 'system', performedAt: '2024-11-15T11:05:00Z' },
      { id: 'tl-003', event: 'segmentation_completed', description: 'Segmentation completed — 8 structures identified', performedBy: 'system', performedAt: '2024-11-15T11:23:00Z' },
      { id: 'tl-004', event: 'segmentation_approved', description: 'Segmentation results reviewed and approved', performedBy: 'Dr. Emily Chen', performedAt: '2024-11-15T14:12:00Z' },
      { id: 'tl-005', event: 'planning_started', description: 'Reduction planning initiated', performedBy: 'Dr. Emily Chen', performedAt: '2024-11-15T14:15:00Z' },
    ],
    createdAt: '2024-11-15T11:00:00Z', updatedAt: '2024-11-16T09:22:00Z',
    scheduledDate: '2024-11-20T08:00:00Z', segmentationJobId: 'job-001', currentPlanId: 'plan-001',
  },
  {
    id: 'case-002', caseNumber: 'FA-2024-0048', studyId: 'st-002', patientId: 'pt-002',
    type: 'panfacial_fracture', status: 'review', priority: 'urgent',
    assignments: [
      { surgeonId: 'surg-001', surgeonName: 'Dr. Emily Chen', role: 'primary', assignedAt: '2024-11-18T14:00:00Z' },
      { surgeonId: 'surg-003', surgeonName: 'Dr. Aisha Okonkwo', role: 'assistant', assignedAt: '2024-11-18T14:00:00Z' },
    ],
    notes: [],
    timeline: [
      { id: 'tl-006', event: 'case_created', description: 'Case created', performedBy: 'Dr. Emily Chen', performedAt: '2024-11-18T14:00:00Z' },
      { id: 'tl-007', event: 'segmentation_completed', description: 'Segmentation complete', performedBy: 'system', performedAt: '2024-11-18T14:35:00Z' },
      { id: 'tl-008', event: 'planning_completed', description: 'Plan v2.1 finalized', performedBy: 'Dr. Emily Chen', performedAt: '2024-11-19T16:00:00Z' },
    ],
    createdAt: '2024-11-18T14:00:00Z', updatedAt: '2024-11-20T10:15:00Z',
    scheduledDate: '2024-11-24T07:30:00Z', segmentationJobId: 'job-002', currentPlanId: 'plan-002',
  },
  {
    id: 'case-003', caseNumber: 'FA-2024-0049', studyId: 'st-001', patientId: 'pt-003',
    type: 'orbital_fracture', status: 'segmentation_review', priority: 'routine',
    assignments: [{ surgeonId: 'surg-002', surgeonName: 'Dr. Marcus Reid', role: 'primary', assignedAt: '2024-11-20T10:00:00Z' }],
    notes: [], timeline: [],
    createdAt: '2024-11-20T10:00:00Z', updatedAt: '2024-11-20T11:30:00Z',
    segmentationJobId: 'job-003',
  },
  {
    id: 'case-004', caseNumber: 'FA-2024-0051', studyId: 'st-002', patientId: 'pt-004',
    type: 'orthognathic', status: 'approved', priority: 'routine',
    assignments: [{ surgeonId: 'surg-003', surgeonName: 'Dr. Aisha Okonkwo', role: 'primary', assignedAt: '2024-11-10T09:00:00Z' }],
    notes: [], timeline: [],
    createdAt: '2024-11-10T09:00:00Z', updatedAt: '2024-11-22T15:00:00Z',
    currentPlanId: 'plan-003',
  },
  {
    id: 'case-005', caseNumber: 'FA-2024-0053', studyId: 'st-001', patientId: 'pt-005',
    type: 'midface_fracture', status: 'segmentation_in_progress', priority: 'stat',
    assignments: [{ surgeonId: 'surg-001', surgeonName: 'Dr. Emily Chen', role: 'primary', assignedAt: '2024-11-25T08:00:00Z' }],
    notes: [], timeline: [],
    createdAt: '2024-11-25T08:00:00Z', updatedAt: '2024-11-25T08:12:00Z',
    segmentationJobId: 'job-005',
  },
  {
    id: 'case-006', caseNumber: 'FA-2024-0055', studyId: 'st-002', patientId: 'pt-006',
    type: 'frontal_sinus_fracture', status: 'completed', priority: 'routine',
    assignments: [{ surgeonId: 'surg-002', surgeonName: 'Dr. Marcus Reid', role: 'primary', assignedAt: '2024-11-20T13:00:00Z' }],
    notes: [], timeline: [],
    createdAt: '2024-11-20T13:00:00Z', updatedAt: '2024-11-26T09:00:00Z', completedAt: '2024-11-26T09:00:00Z',
    currentPlanId: 'plan-004',
  },
]

// ---------------------------
// Dashboard Stats
// ---------------------------

export const MOCK_DASHBOARD_STATS: DashboardStats = {
  activeCases: 4,
  pendingSegmentation: 2,
  awaitingReview: 1,
  completedThisMonth: 8,
  activeCasesDelta: 1,
  pendingSegmentationDelta: -1,
  awaitingReviewDelta: 0,
  completedThisMonthDelta: 3,
}

// ---------------------------
// Recent Cases Table
// ---------------------------

export const MOCK_RECENT_CASES: RecentCaseRow[] = MOCK_CASES.map(c => ({
  id: c.id,
  caseNumber: c.caseNumber,
  anonymizedPatientId: MOCK_PATIENTS.find(p => p.id === c.patientId)?.anonymizedId ?? 'UNKNOWN',
  type: c.type,
  status: c.status,
  priority: c.priority,
  primarySurgeon: c.assignments.find(a => a.role === 'primary')?.surgeonName ?? 'Unassigned',
  updatedAt: c.updatedAt,
  scheduledDate: c.scheduledDate,
}))

// ---------------------------
// Segmentation Result
// ---------------------------

export const MOCK_SEGMENTATION_RESULT: SegmentationResult = {
  id: 'seg-001', caseId: 'case-001', jobId: 'job-001',
  modelName: 'FacialSeg-v3', modelVersion: '3.2.1',
  inferenceTimeSeconds: 47.3,
  overallConfidence: { value: 0.92, threshold: 0.85, passesClinicalThreshold: true },
  warnings: ['Right condyle fragment confidence below threshold (0.79) — manual review recommended'],
  createdAt: '2024-11-15T11:05:00Z', completedAt: '2024-11-15T11:23:00Z',
  gpuUsed: 'NVIDIA A100 80GB',
  structures: [
    {
      label: 'mandible', displayName: 'Mandible (main body)',
      confidence: { value: 0.97, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 42.8, surfaceAreaCm2: 184.3,
      centroid: { x: 0.2, y: -12.4, z: -8.1 },
      boundingBox: { min: { x: -45, y: -30, z: -25 }, max: { x: 45, y: 5, z: 10 }, center: { x: 0, y: -12.5, z: -7.5 } },
      meshUri: '/mock-meshes/mandible.glb', color: '#eab308', opacity: 1.0, status: 'accepted',
      fragmentCount: 3,
    },
    {
      label: 'maxilla', displayName: 'Maxilla',
      confidence: { value: 0.95, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 28.4, surfaceAreaCm2: 142.7,
      centroid: { x: 0.1, y: 2.3, z: -5.4 },
      boundingBox: { min: { x: -38, y: -8, z: -20 }, max: { x: 38, y: 18, z: 5 }, center: { x: 0, y: 5, z: -7.5 } },
      meshUri: '/mock-meshes/maxilla.glb', color: '#3b82f6', opacity: 1.0, status: 'accepted',
    },
    {
      label: 'zygoma_left', displayName: 'Zygoma (Left)',
      confidence: { value: 0.91, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 8.2, surfaceAreaCm2: 62.1,
      centroid: { x: -35.2, y: 5.1, z: -2.3 },
      boundingBox: { min: { x: -55, y: -5, z: -15 }, max: { x: -20, y: 15, z: 10 }, center: { x: -37.5, y: 5, z: -2.5 } },
      meshUri: '/mock-meshes/zygoma_l.glb', color: '#8b5cf6', opacity: 1.0, status: 'accepted',
    },
    {
      label: 'zygoma_right', displayName: 'Zygoma (Right)',
      confidence: { value: 0.89, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 8.4, surfaceAreaCm2: 63.8,
      centroid: { x: 35.1, y: 5.0, z: -2.1 },
      boundingBox: { min: { x: 20, y: -5, z: -15 }, max: { x: 55, y: 15, z: 10 }, center: { x: 37.5, y: 5, z: -2.5 } },
      meshUri: '/mock-meshes/zygoma_r.glb', color: '#8b5cf6', opacity: 1.0, status: 'accepted',
    },
    {
      label: 'teeth_upper', displayName: 'Upper Dentition',
      confidence: { value: 0.93, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 14.2, surfaceAreaCm2: 88.6,
      centroid: { x: 0.3, y: -4.1, z: -3.2 },
      boundingBox: { min: { x: -32, y: -12, z: -18 }, max: { x: 32, y: 2, z: 8 }, center: { x: 0, y: -5, z: -5 } },
      meshUri: '/mock-meshes/teeth_upper.glb', color: '#f8fafc', opacity: 1.0, status: 'accepted',
    },
    {
      label: 'teeth_lower', displayName: 'Lower Dentition',
      confidence: { value: 0.94, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 12.8, surfaceAreaCm2: 81.4,
      centroid: { x: 0.1, y: -18.2, z: -3.8 },
      boundingBox: { min: { x: -30, y: -25, z: -18 }, max: { x: 30, y: -12, z: 8 }, center: { x: 0, y: -18.5, z: -5 } },
      meshUri: '/mock-meshes/teeth_lower.glb', color: '#f8fafc', opacity: 1.0, status: 'accepted',
    },
    {
      label: 'orbit_left', displayName: 'Left Orbit',
      confidence: { value: 0.88, threshold: 0.85, passesClinicalThreshold: true },
      volumeCm3: 22.4, surfaceAreaCm2: 108.3,
      centroid: { x: -28.4, y: 22.1, z: -4.8 },
      boundingBox: { min: { x: -50, y: 12, z: -20 }, max: { x: -12, y: 35, z: 12 }, center: { x: -31, y: 23.5, z: -4 } },
      meshUri: '/mock-meshes/orbit_l.glb', color: '#f97316', opacity: 0.85, status: 'accepted',
    },
    {
      label: 'orbit_right', displayName: 'Right Orbit',
      confidence: { value: 0.79, threshold: 0.85, passesClinicalThreshold: false },
      volumeCm3: 19.1, surfaceAreaCm2: 98.7,
      centroid: { x: 28.2, y: 21.9, z: -5.1 },
      boundingBox: { min: { x: 12, y: 12, z: -20 }, max: { x: 50, y: 35, z: 12 }, center: { x: 31, y: 23.5, z: -4 } },
      meshUri: '/mock-meshes/orbit_r.glb', color: '#f97316', opacity: 0.85, status: 'flagged',
    },
  ],
}

// ---------------------------
// Fragment Transforms (for planning)
// ---------------------------

export const MOCK_FRAGMENT_TRANSFORMS: FragmentTransform[] = [
  {
    fragmentId: 'frag-001', structureLabel: 'mandible', displayName: 'Mandible Body',
    baseTransform: { translation: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
    currentTransform: { translation: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
    suggestedTransform: { translation: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
    suggestionConfidence: 0.97,
    isAligned: true, isLocked: true, volumeCm3: 28.4, centroid: { x: 0.2, y: -12.4, z: -8.1 },
  },
  {
    fragmentId: 'frag-002', structureLabel: 'mandible', displayName: 'Right Parasymphysis Fragment',
    baseTransform: { translation: { x: 4.2, y: -3.1, z: 2.0 }, rotation: { x: 5, y: 12, z: -3 }, scale: { x: 1, y: 1, z: 1 } },
    currentTransform: { translation: { x: 2.1, y: -1.5, z: 0.9 }, rotation: { x: 2.5, y: 6.1, z: -1.4 }, scale: { x: 1, y: 1, z: 1 } },
    suggestedTransform: { translation: { x: 0.3, y: -0.2, z: 0.1 }, rotation: { x: 0.4, y: 0.8, z: -0.2 }, scale: { x: 1, y: 1, z: 1 } },
    suggestionConfidence: 0.84,
    isAligned: false, isLocked: false, volumeCm3: 8.7, centroid: { x: 18.4, y: -15.2, z: -7.8 },
  },
  {
    fragmentId: 'frag-003', structureLabel: 'mandible', displayName: 'Right Condylar Head',
    baseTransform: { translation: { x: 6.8, y: -2.4, z: 8.1 }, rotation: { x: -8, y: 22, z: 5 }, scale: { x: 1, y: 1, z: 1 } },
    currentTransform: { translation: { x: 3.4, y: -1.2, z: 4.0 }, rotation: { x: -4, y: 11, z: 2.5 }, scale: { x: 1, y: 1, z: 1 } },
    suggestedTransform: { translation: { x: 0.5, y: -0.3, z: 0.6 }, rotation: { x: -1.2, y: 2.1, z: 0.4 }, scale: { x: 1, y: 1, z: 1 } },
    suggestionConfidence: 0.71,
    isAligned: false, isLocked: false, volumeCm3: 5.7, centroid: { x: 42.1, y: 8.4, z: 4.2 },
  },
]

// ---------------------------
// Reduction Plan
// ---------------------------

export const MOCK_REDUCTION_PLAN: ReductionPlan = {
  id: 'plan-001', caseId: 'case-001', version: 3, name: 'Plan v3 — AI-Optimized',
  description: 'Anatomic reduction of parasymphysis + condylar subcondylar fractures with attention to occlusal restoration',
  fragmentTransforms: MOCK_FRAGMENT_TRANSFORMS,
  occlusalMetrics: {
    overjetMm: 3.8, overjetIdealMin: 1.0, overjetIdealMax: 4.0,
    overbitePercent: 24, overbiteIdealMin: 15, overbiteIdealMax: 35,
    midlineDeviationMm: 1.2, midlineDeviationThreshold: 2.0,
    occlusalCantDeg: 0.8,
    molarRelationshipLeft: 'I', molarRelationshipRight: 'I',
    canineRelationshipLeft: 'I', canineRelationshipRight: 'I',
  },
  constraints: {
    enforceOverjet: true, enforceOverbite: true, enforceMidline: true,
    enforceSymmetry: true, enforceCondylarSeating: true, maxCondylarDeviationMm: 2.0,
  },
  validations: [
    { name: 'Occlusal Restoration', description: 'Class I occlusion achieved bilaterally', passed: true, severity: 'error' },
    { name: 'Overjet', description: 'Overjet within ideal range (1–4mm)', passed: true, value: 3.8, threshold: 4.0, severity: 'error' },
    { name: 'Overbite', description: 'Overbite within normal range (15–35%)', passed: true, value: 24, threshold: 35, severity: 'warning' },
    { name: 'Midline', description: 'Midline deviation < 2mm', passed: true, value: 1.2, threshold: 2.0, severity: 'warning' },
    { name: 'Condylar Seating', description: 'Condyles properly seated in fossa', passed: false, severity: 'error' },
    { name: 'Facial Symmetry', description: 'Bilateral facial dimensions within 2mm', passed: true, severity: 'warning' },
    { name: 'Fragment Alignment', description: 'All fragments within 1mm of reduced position', passed: false, severity: 'error' },
  ],
  aiConfidence: 0.81,
  aiRecommendation: 'Anatomic reduction achievable with MMF application. Right condylar fragment requires careful manual positioning — AI confidence is moderate (71%). Consider ORIF for condylar subcondylar component. Parasymphysis fragment shows excellent reduction potential.',
  isApproved: false,
  createdAt: '2024-11-15T14:15:00Z', updatedAt: '2024-11-16T08:44:00Z', createdBy: 'Dr. Emily Chen',
}

// ---------------------------
// Surgeon Review
// ---------------------------

export const MOCK_SURGEON_REVIEW: SurgeonReview = {
  id: 'rev-001', caseId: 'case-002', planId: 'plan-002',
  reviewerId: 'surg-002', reviewerName: 'Dr. Marcus Reid',
  decision: 'pending', notes: '',
  checklist: [
    { id: 'chk-001', category: 'Anatomy', label: 'Fracture fragments correctly identified', passed: true, severity: 'required' },
    { id: 'chk-002', category: 'Anatomy', label: 'Segmentation boundaries accurate', passed: true, severity: 'required' },
    { id: 'chk-003', category: 'Occlusion', label: 'Pre-injury occlusal relationship restored', passed: true, severity: 'required' },
    { id: 'chk-004', category: 'Occlusion', label: 'Molar relationship Class I bilaterally', passed: true, severity: 'required' },
    { id: 'chk-005', category: 'Biomechanics', label: 'Condylar seating verified', passed: null, severity: 'required' },
    { id: 'chk-006', category: 'Biomechanics', label: 'Masticatory muscle vectors considered', passed: true, severity: 'recommended' },
    { id: 'chk-007', category: 'Symmetry', label: 'Facial midline aligned', passed: true, severity: 'required' },
    { id: 'chk-008', category: 'Symmetry', label: 'Orbital rims symmetric', passed: null, severity: 'recommended' },
    { id: 'chk-009', category: 'Safety', label: 'No neurovascular compromise identified', passed: true, severity: 'required' },
    { id: 'chk-010', category: 'Safety', label: 'IAN canal path clear of planned hardware', passed: true, severity: 'required' },
  ],
  createdAt: '2024-11-20T10:00:00Z', updatedAt: '2024-11-20T10:00:00Z',
}

// ---------------------------
// System Health
// ---------------------------

export const MOCK_SYSTEM_HEALTH: SystemHealth = {
  gpus: [
    { id: 'gpu-0', name: 'NVIDIA A100 80GB (0)', utilizationPercent: 73, memoryUsedGb: 52.4, memoryTotalGb: 80, temperatureCelsius: 68, status: 'busy' },
    { id: 'gpu-1', name: 'NVIDIA A100 80GB (1)', utilizationPercent: 12, memoryUsedGb: 8.1, memoryTotalGb: 80, temperatureCelsius: 42, status: 'idle' },
    { id: 'gpu-2', name: 'NVIDIA A100 80GB (2)', utilizationPercent: 0, memoryUsedGb: 0.4, memoryTotalGb: 80, temperatureCelsius: 38, status: 'idle' },
    { id: 'gpu-3', name: 'NVIDIA A100 80GB (3)', utilizationPercent: 91, memoryUsedGb: 74.2, memoryTotalGb: 80, temperatureCelsius: 74, status: 'busy' },
  ],
  models: [
    { name: 'FacialSeg-v3', version: '3.2.1', type: 'segmentation', lastUpdated: '2024-11-01', accuracy: 0.94 },
    { name: 'ReductionPlanner', version: '2.1.0', type: 'planning', lastUpdated: '2024-10-15', accuracy: 0.88 },
    { name: 'OcclusionNet', version: '1.4.2', type: 'occlusion', lastUpdated: '2024-11-10', accuracy: 0.91 },
  ],
  queue: { depth: 3, estimatedWaitMinutes: 14, processingCount: 2 },
  apiLatencyMs: 42,
  storageUsedGb: 8421,
  storageTotalGb: 20000,
  lastChecked: new Date().toISOString(),
}

/**
 * Client-side validation utilities for Facial Align.
 * Validates transforms, DICOM files, occlusal constraints, fragment IDs,
 * and planning workspace form fields.
 */

import type { Transform3D, OcclusalConstraints, Vector3 } from '../types/medical'

// =============================================================================
// Result type
// =============================================================================

export interface ValidationResult {
  valid: boolean
  errors: ValidationFieldError[]
}

export interface ValidationFieldError {
  field: string
  message: string
  code: string
}

function ok(): ValidationResult {
  return { valid: true, errors: [] }
}

function fail(field: string, message: string, code: string): ValidationResult {
  return { valid: false, errors: [{ field, message, code }] }
}

function merge(...results: ValidationResult[]): ValidationResult {
  const errors = results.flatMap(r => r.errors)
  return { valid: errors.length === 0, errors }
}

// =============================================================================
// Transform3D Validation
// =============================================================================

const ORTHONORMALITY_TOLERANCE = 1e-4

/**
 * Validate that a Transform3D has sane Euler angles and scale.
 * Full rotation-matrix orthonormality is checked on the derived matrix.
 */
export function validateTransform3D(t: Transform3D, fieldPrefix = 'transform'): ValidationResult {
  const results: ValidationResult[] = []

  // Rotation angle ranges: allow ±360° in each axis
  for (const axis of ['x', 'y', 'z'] as const) {
    const angle = t.rotation[axis]
    if (!isFinite(angle)) {
      results.push(fail(`${fieldPrefix}.rotation.${axis}`, `Rotation ${axis} must be a finite number`, 'ROTATION_NOT_FINITE'))
    }
    if (Math.abs(angle) > 360) {
      results.push(fail(`${fieldPrefix}.rotation.${axis}`, `Rotation ${axis} (${angle.toFixed(1)}°) exceeds ±360°`, 'ROTATION_OUT_OF_RANGE'))
    }
  }

  // Translation: must be finite
  for (const axis of ['x', 'y', 'z'] as const) {
    const val = t.translation[axis]
    if (!isFinite(val)) {
      results.push(fail(`${fieldPrefix}.translation.${axis}`, `Translation ${axis} must be a finite number`, 'TRANSLATION_NOT_FINITE'))
    }
    // Sanity check: craniofacial translations rarely exceed 200mm
    if (Math.abs(val) > 200) {
      results.push(fail(`${fieldPrefix}.translation.${axis}`, `Translation ${axis} (${val.toFixed(1)} mm) is unusually large (>200 mm)`, 'TRANSLATION_LARGE'))
    }
  }

  // Scale: must be positive and close to 1 (rigid body, no anisotropic scaling)
  for (const axis of ['x', 'y', 'z'] as const) {
    const s = t.scale[axis]
    if (!isFinite(s) || s <= 0) {
      results.push(fail(`${fieldPrefix}.scale.${axis}`, `Scale ${axis} must be a positive finite number`, 'SCALE_INVALID'))
    }
    if (s < 0.5 || s > 2.0) {
      results.push(fail(`${fieldPrefix}.scale.${axis}`, `Scale ${axis} (${s.toFixed(3)}) is outside the expected range [0.5, 2.0]`, 'SCALE_OUT_OF_RANGE'))
    }
  }

  // Check rotation matrix orthonormality derived from Euler angles
  if (results.length === 0) {
    const matResult = validateRotationMatrixOrthonormality(
      eulerDegreesToRotationMatrix(t.rotation),
      fieldPrefix
    )
    results.push(matResult)
  }

  return merge(...results)
}

/**
 * Verify a 3x3 rotation matrix (stored as row-major [r0, r1, r2, r3, r4, r5, r6, r7, r8])
 * is orthonormal: R^T R = I and det(R) = +1.
 */
export function validateRotationMatrixOrthonormality(
  mat: number[],
  fieldPrefix = 'rotation'
): ValidationResult {
  if (mat.length !== 9) {
    return fail(fieldPrefix, 'Rotation matrix must have exactly 9 elements (3x3)', 'MATRIX_WRONG_SIZE')
  }

  // Check each column is a unit vector and columns are mutually orthogonal
  const col0: [number, number, number] = [mat[0], mat[3], mat[6]]
  const col1: [number, number, number] = [mat[1], mat[4], mat[7]]
  const col2: [number, number, number] = [mat[2], mat[5], mat[8]]

  const len0 = vecMag(col0)
  const len1 = vecMag(col1)
  const len2 = vecMag(col2)

  if (Math.abs(len0 - 1) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix column 0 is not unit length (|col|=${len0.toFixed(6)})`, 'MATRIX_NOT_ORTHONORMAL')
  }
  if (Math.abs(len1 - 1) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix column 1 is not unit length (|col|=${len1.toFixed(6)})`, 'MATRIX_NOT_ORTHONORMAL')
  }
  if (Math.abs(len2 - 1) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix column 2 is not unit length (|col|=${len2.toFixed(6)})`, 'MATRIX_NOT_ORTHONORMAL')
  }

  const dot01 = vecDot(col0, col1)
  const dot02 = vecDot(col0, col2)
  const dot12 = vecDot(col1, col2)

  if (Math.abs(dot01) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix columns 0 and 1 are not orthogonal (dot=${dot01.toFixed(6)})`, 'MATRIX_NOT_ORTHOGONAL')
  }
  if (Math.abs(dot02) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix columns 0 and 2 are not orthogonal (dot=${dot02.toFixed(6)})`, 'MATRIX_NOT_ORTHOGONAL')
  }
  if (Math.abs(dot12) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix columns 1 and 2 are not orthogonal (dot=${dot12.toFixed(6)})`, 'MATRIX_NOT_ORTHOGONAL')
  }

  // Determinant must be +1 (not -1, which would be a reflection)
  const det = mat3Det(mat)
  if (Math.abs(det - 1) > ORTHONORMALITY_TOLERANCE) {
    return fail(fieldPrefix, `Rotation matrix determinant is ${det.toFixed(6)} — must be +1 (proper rotation)`, 'MATRIX_NOT_PROPER_ROTATION')
  }

  return ok()
}

// =============================================================================
// DICOM File Validation
// =============================================================================

const MAX_DICOM_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024 // 2 GB per file
const MAX_DICOM_TOTAL_SIZE_BYTES = 10 * 1024 * 1024 * 1024 // 10 GB total upload

/** Accepted DICOM file extensions */
const DICOM_EXTENSIONS = new Set(['.dcm', '.dicom', '.ima', '.img', ''])

export function validateDicomFile(file: File): ValidationResult {
  const results: ValidationResult[] = []
  const ext = getFileExtension(file.name).toLowerCase()

  // Extension check — some DICOM files have no extension
  if (!DICOM_EXTENSIONS.has(ext) && ext !== '') {
    results.push(fail('file', `"${file.name}" does not appear to be a DICOM file (extension: "${ext || 'none'}")`, 'INVALID_DICOM_EXTENSION'))
  }

  // Size check
  if (file.size === 0) {
    results.push(fail('file', `"${file.name}" is empty`, 'EMPTY_FILE'))
  } else if (file.size > MAX_DICOM_FILE_SIZE_BYTES) {
    results.push(fail('file', `"${file.name}" exceeds the 2 GB per-file limit`, 'FILE_TOO_LARGE'))
  }

  return merge(...results)
}

export function validateDicomFileSet(files: File[]): ValidationResult {
  if (files.length === 0) {
    return fail('files', 'No files selected. Please select a DICOM series.', 'NO_FILES')
  }

  if (files.length > 2000) {
    return fail('files', `Too many files selected (${files.length}). Maximum is 2000 DICOM slices per upload.`, 'TOO_MANY_FILES')
  }

  const totalSize = files.reduce((acc, f) => acc + f.size, 0)
  if (totalSize > MAX_DICOM_TOTAL_SIZE_BYTES) {
    const totalGb = (totalSize / 1024 ** 3).toFixed(1)
    return fail('files', `Total upload size (${totalGb} GB) exceeds the 10 GB limit.`, 'TOTAL_SIZE_TOO_LARGE')
  }

  // Per-file validation — collect first 10 errors
  const errors: ValidationFieldError[] = []
  for (const file of files) {
    const r = validateDicomFile(file)
    if (!r.valid) {
      errors.push(...r.errors)
      if (errors.length >= 10) break
    }
  }

  return errors.length > 0 ? { valid: false, errors } : ok()
}

// =============================================================================
// Occlusal Constraint Validation
// =============================================================================

/** Clinical ranges for surgical planning parameters */
const OCCLUSAL_RANGES = {
  maxCondylarDeviationMm: { min: 0, max: 10, label: 'Maximum condylar deviation' },
} as const

export function validateOcclusalConstraints(constraints: OcclusalConstraints): ValidationResult {
  const results: ValidationResult[] = []

  const { min, max, label } = OCCLUSAL_RANGES.maxCondylarDeviationMm
  const val = constraints.maxCondylarDeviationMm

  if (!isFinite(val) || val < min) {
    results.push(fail('maxCondylarDeviationMm', `${label} must be ≥ ${min} mm`, 'CONDYLAR_DEVIATION_BELOW_MIN'))
  } else if (val > max) {
    results.push(fail('maxCondylarDeviationMm', `${label} (${val} mm) exceeds maximum of ${max} mm`, 'CONDYLAR_DEVIATION_ABOVE_MAX'))
  }

  // Validate that at least one constraint is enforced when any boolean flag is true
  const enforcedFlags = [
    constraints.enforceOverjet,
    constraints.enforceOverbite,
    constraints.enforceMidline,
    constraints.enforceSymmetry,
    constraints.enforceCondylarSeating,
  ]
  if (enforcedFlags.every(f => f === false)) {
    results.push(fail('constraints', 'At least one occlusal constraint should be enforced for a valid surgical plan.', 'NO_CONSTRAINTS_ENFORCED'))
  }

  return merge(...results)
}

// =============================================================================
// Overjet / Overbite Range Validation
// =============================================================================

export interface OcclusalMetricInput {
  overjetMm: number
  overbitePercent: number
  midlineDeviationMm: number
  occlusalCantDeg: number
}

export function validateOcclusalMetrics(metrics: OcclusalMetricInput): ValidationResult {
  const results: ValidationResult[] = []

  if (!isFinite(metrics.overjetMm) || metrics.overjetMm < -10 || metrics.overjetMm > 20) {
    results.push(fail('overjetMm', `Overjet must be between -10 mm and +20 mm (got ${metrics.overjetMm} mm)`, 'OVERJET_OUT_OF_RANGE'))
  }

  if (!isFinite(metrics.overbitePercent) || metrics.overbitePercent < -50 || metrics.overbitePercent > 100) {
    results.push(fail('overbitePercent', `Overbite must be between -50% and 100% (got ${metrics.overbitePercent}%)`, 'OVERBITE_OUT_OF_RANGE'))
  }

  if (!isFinite(metrics.midlineDeviationMm) || Math.abs(metrics.midlineDeviationMm) > 15) {
    results.push(fail('midlineDeviationMm', `Midline deviation must be ≤ ±15 mm (got ${metrics.midlineDeviationMm} mm)`, 'MIDLINE_DEVIATION_LARGE'))
  }

  if (!isFinite(metrics.occlusalCantDeg) || Math.abs(metrics.occlusalCantDeg) > 20) {
    results.push(fail('occlusalCantDeg', `Occlusal cant must be ≤ ±20° (got ${metrics.occlusalCantDeg}°)`, 'OCCLUSAL_CANT_LARGE'))
  }

  return merge(...results)
}

// =============================================================================
// Fragment ID Format Validation
// =============================================================================

/**
 * Fragment IDs must match: frag-<alphanumeric+hyphens>, e.g. "frag-001", "frag-condyle-r"
 */
const FRAGMENT_ID_PATTERN = /^frag-[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*$/

export function validateFragmentId(id: string): ValidationResult {
  if (!id || id.trim().length === 0) {
    return fail('fragmentId', 'Fragment ID cannot be empty', 'FRAGMENT_ID_EMPTY')
  }
  if (!FRAGMENT_ID_PATTERN.test(id)) {
    return fail('fragmentId', `Fragment ID "${id}" is invalid. Must match pattern: frag-<alphanumeric> (e.g. "frag-001", "frag-condyle-r")`, 'FRAGMENT_ID_INVALID_FORMAT')
  }
  if (id.length > 64) {
    return fail('fragmentId', `Fragment ID is too long (${id.length} characters, max 64)`, 'FRAGMENT_ID_TOO_LONG')
  }
  return ok()
}

// =============================================================================
// Planning Workspace Form Helpers
// =============================================================================

export interface PlanNameInput {
  name: string
  description?: string
}

export function validatePlanName(input: PlanNameInput): ValidationResult {
  const results: ValidationResult[] = []

  if (!input.name || input.name.trim().length === 0) {
    results.push(fail('name', 'Plan name is required', 'PLAN_NAME_EMPTY'))
  } else if (input.name.trim().length < 3) {
    results.push(fail('name', 'Plan name must be at least 3 characters', 'PLAN_NAME_TOO_SHORT'))
  } else if (input.name.length > 120) {
    results.push(fail('name', `Plan name is too long (${input.name.length} chars, max 120)`, 'PLAN_NAME_TOO_LONG'))
  }

  if (input.description && input.description.length > 500) {
    results.push(fail('description', `Description is too long (${input.description.length} chars, max 500)`, 'DESCRIPTION_TOO_LONG'))
  }

  return merge(...results)
}

export interface CaseNoteInput {
  content: string
  tags?: string[]
}

export function validateCaseNote(input: CaseNoteInput): ValidationResult {
  const results: ValidationResult[] = []

  if (!input.content || input.content.trim().length === 0) {
    results.push(fail('content', 'Note content cannot be empty', 'NOTE_EMPTY'))
  } else if (input.content.length > 5000) {
    results.push(fail('content', `Note is too long (${input.content.length} chars, max 5000)`, 'NOTE_TOO_LONG'))
  }

  if (input.tags) {
    if (input.tags.length > 10) {
      results.push(fail('tags', `Too many tags (${input.tags.length}, max 10)`, 'TOO_MANY_TAGS'))
    }
    for (const tag of input.tags) {
      if (tag.length > 32) {
        results.push(fail('tags', `Tag "${tag.slice(0, 20)}..." is too long (max 32 characters)`, 'TAG_TOO_LONG'))
        break
      }
    }
  }

  return merge(...results)
}

/** Validate a point array has the correct count and all finite coordinates */
export function validatePointArray(points: Vector3[], expectedCount?: number, fieldName = 'points'): ValidationResult {
  if (expectedCount != null && points.length !== expectedCount) {
    return fail(fieldName, `Expected ${expectedCount} point(s), got ${points.length}`, 'WRONG_POINT_COUNT')
  }

  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    if (!isFinite(p.x) || !isFinite(p.y) || !isFinite(p.z)) {
      return fail(fieldName, `Point ${i} has non-finite coordinates`, 'POINT_NOT_FINITE')
    }
  }

  return ok()
}

// =============================================================================
// Internal helpers
// =============================================================================

function getFileExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf('.')
  return dotIndex === -1 ? '' : filename.slice(dotIndex)
}

function vecMag(v: [number, number, number]): number {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
}

function vecDot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function mat3Det(m: number[]): number {
  // Row-major: [m0,m1,m2, m3,m4,m5, m6,m7,m8]
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  )
}

/**
 * Convert Euler angles (degrees, XYZ intrinsic order) to a 3x3 rotation matrix (row-major).
 */
export function eulerDegreesToRotationMatrix(euler: Vector3): number[] {
  const toRad = Math.PI / 180
  const cx = Math.cos(euler.x * toRad), sx = Math.sin(euler.x * toRad)
  const cy = Math.cos(euler.y * toRad), sy = Math.sin(euler.y * toRad)
  const cz = Math.cos(euler.z * toRad), sz = Math.sin(euler.z * toRad)

  // Rz * Ry * Rx (row-major)
  return [
    cy * cz,                   cy * sz,                   -sy,
    sx * sy * cz - cx * sz,    sx * sy * sz + cx * cz,     sx * cy,
    cx * sy * cz + sx * sz,    cx * sy * sz - sx * cz,     cx * cy,
  ]
}
